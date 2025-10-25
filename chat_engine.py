import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode


# ----------------------------
# Logging (minimal)
# ----------------------------
LOG_FILENAME = f"agentTutorial-{time.strftime('%m-%d-%H%M%S')}.log"


def get_logger() -> logging.Logger:
    logger = logging.getLogger("chat_engine")
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILENAME, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    return logger


logger = get_logger()


# ----------------------------
# Examples (read from JSON next to this file)
# ----------------------------

def _load_examples_text() -> str:
    data = json.loads((Path(__file__).parent / "mimic_examples.json").read_text(encoding="utf-8"))
    return "\n\n".join(
        f"Q: {ex['q']}\nSQL:\n```sql\n{ex['sql']}\n```" for ex in data["examples"]
    )


# ----------------------------
# Tools
# ----------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression with digits, + - * / and parentheses."""
    logger.info("calculator:start expr=%s", expression)
    try:
        if not re.fullmatch(r"[0-9\s+\-*/().]+", expression):
            return "Error: invalid characters. Only digits, +-*/ and parentheses are allowed."
        result = eval(expression, {"__builtins__": {}}, {})
        logger.info("calculator:success result=%s", result)
        return str(result)
    except Exception as e:
        logger.exception("calculator:error")
        return f"Error: {e}"


@tool
def mimic(question: str = "", top_n: int = 50, dry_run: bool = False) -> str:
    """LLM-only SQL over MIMIC-IV ED (BigQuery StandardSQL). Returns SQL + CSV preview.

    Assumes a file named mimic_examples.json is in the same directory as this module.
    """
    from google.cloud import bigquery

    started = time.monotonic()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("SQL_GEN_MODEL", "gemini-2.5-flash")

    def terr(kind: str, msg: str) -> str:
        return f"TOOL_ERROR[mimic]: {kind}: {msg}"

    if not question.strip():
        return terr("ValueError", "question is required.")
    if not api_key:
        return terr("RuntimeError", "Missing GEMINI_API_KEY/GOOGLE_API_KEY.")

    limit_cap = int(top_n or 50)

    def extract_sql(text: str) -> str:
        m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
        sql = (m.group(1) if m else text).strip()
        return re.sub(r";\s*$", "", sql)

    def csv_preview(df, n: int) -> str:
        try:
            return df.head(n).to_csv(index=False)
        except Exception:
            return "(preview unavailable)"

    # --- generate SQL with LLM ---
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)
        rules = (
            "You are a BigQuery SQL assistant for MIMIC-IV emergency department (ED) data.\n"
            "Return ONE StandardSQL query only in a fenced ```sql``` block, no prose.\n"
            "Use fully-qualified tables from physionet-data (e.g., mimiciv_ed.edstays, diagnosis, triage, vitalsign;\n"
            "and mimiciv_3_1_hosp.d_icd_diagnoses for ICD titles). ED visit key is edstays.stay_id.\n"
            "Group only by scalar fields; when counting ED visits per subject, group by subject_id.\n"
            f"Include a LIMIT ≤ {limit_cap}. Use StandardSQL.\n"
        )
        examples_txt = _load_examples_text()
        examples = f"\nEXAMPLES (guidance only; do not echo):\n{examples_txt}\n"
        sys_msg = SystemMessage(content=rules + examples)
        user_msg = HumanMessage(content=f"User question: {question}\nReturn only the SQL in a fenced code block.")
        reply = llm.invoke([sys_msg, user_msg])
        sql = extract_sql((getattr(reply, "content", "") or "").strip())
        if not sql:
            return terr("RuntimeError", "LLM failed to produce SQL.")
        if dry_run:
            return f"SQL (dry run):\n```sql\n{sql}\n```"
    except Exception as e:
        return terr(type(e).__name__, str(e))

    # --- execute in BigQuery ---
    try:
        client = bigquery.Client(project=os.getenv("GCP_PROJECT_ID", "indigo-proxy-472718-m1"))
        job_cfg = bigquery.QueryJobConfig(
            use_legacy_sql=False,
            maximum_bytes_billed=int(os.getenv("BQ_MAX_BYTES_BILLED", "1000000000")),
        )
        df = client.query(sql, job_config=job_cfg).to_dataframe()
        preview = csv_preview(df, n=min(20, limit_cap))
        elapsed = time.monotonic() - started
        return (
            f"SQL:\n```sql\n{sql}\n```\n"
            f"Rows returned: {len(df)}\n"
            f"Preview (CSV):\n```csv\n{preview}\n```\n"
            f"(elapsed {elapsed:.2f}s)"
        )
    except Exception as e:
        logger.exception("mimic:bq_error")
        return terr(type(e).__name__, str(e))


# ----------------------------
# Agent
# ----------------------------
class LCAgent:
    """Lean LangGraph agent using Gemini + two tools."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        logger.info("LCAgent:init model=%s temp=%s", model, temperature)
        base_llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)
        self.tools = [calculator, mimic]
        self.llm = base_llm.bind_tools(self.tools)
        self.system_text = (
            "You are a careful assistant. "
            "Use calculator for arithmetic; use mimic for MIMIC/ED analytics. "
            "If any tool returns a string starting with 'TOOL_ERROR[', return it verbatim. "
            "Keep responses concise."
        )

        graph = StateGraph(MessagesState)
        graph.add_node("llm", self._llm_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", self._route, {"tools": "tools", END: END})
        graph.add_edge("tools", "llm")
        self.graph = graph.compile()
        logger.info("LCAgent:graph_compiled")

    # --- graph nodes / routing ---
    def _llm_node(self, state: MessagesState):
        reply = self.llm.invoke(state["messages"])
        if getattr(reply, "tool_calls", None):
            logger.info("llm_node:tool_calls=%s", [(tc.get("name"), tc.get("args")) for tc in reply.tool_calls])  # type: ignore[attr-defined]
        else:
            logger.info("llm_node:reply=%s", (getattr(reply, "content", "") or "").strip()[:200])
        return {"messages": [reply]}

    @staticmethod
    def _route(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and getattr(last, "tool_calls", None) else END

    @staticmethod
    def _to_lc_history(history: List[Dict[str, str]]):
        role_map = {"user": HumanMessage, "assistant": AIMessage}
        return [role_map[m.get("role")](m.get("content", "")) for m in history if m.get("role") in role_map]

    def ask(self, history: List[Dict[str, str]], user_input: str) -> str:
        messages = [SystemMessage(self.system_text)] + self._to_lc_history(history)
        if not messages or not isinstance(messages[-1], HumanMessage) or messages[-1].content != user_input:
            messages.append(HumanMessage(user_input))
        state = self.graph.invoke({"messages": messages})
        out = state.get("messages", [])
        if not out:
            return "(no response)"

        # Surface TOOL_ERRORs verbatim
        for m in reversed(out):
            text = getattr(m, "content", "")
            if isinstance(text, str) and text.startswith("TOOL_ERROR["):
                return text

        # Prefer last AI message; fall back to latest ToolMessage
        last_text = getattr(out[-1], "content", "") or ""
        if isinstance(last_text, str) and last_text.strip():
            return last_text
        for m in reversed(out):
            if isinstance(m, ToolMessage):
                t = m.content
                if isinstance(t, str) and t.strip():
                    return t
        return "(no response)"