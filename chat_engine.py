# chat_engine.py
import json
import logging
import os
import re
import time
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

LOG_FILENAME = f"agentTutorial-{time.strftime('%m-%d-%H%M%S')}.log"

def get_logger() -> logging.Logger:
    logger = logging.getLogger("chat_engine")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILENAME, encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

logger = get_logger()

def _load_examples_text() -> str:
    data = json.loads((Path(__file__).parent / "mimic_examples.json").read_text(encoding="utf-8"))
    return "\n\n".join(
        f"Q: {ex['q']}\nSQL:\n```sql\n{ex['sql']}\n```" for ex in data["examples"]
    )

# ---- simple helpers for mimic (flattened) ----

def tool_error(msg: str) -> str:
    return f"TOOL_ERROR[mimic]: {msg}"

def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    sql = (m.group(1) if m else text).strip()
    return re.sub(r";\s*$", "", sql)

def csv_preview(df, n: int) -> str:
    try:
        return df.head(n).to_csv(index=False)
    except Exception:
        return "(preview unavailable)"

# ---- tools ----

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    logger.info("calculator expr=%s", expression)
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"calc error: {e}"

@tool
def mimic(question: str = "", top_n: int = 50, dry_run: bool = False) -> str:
    """LLM-only SQL over MIMIC-IV ED (BigQuery). Returns SQL and CSV preview."""
    from google.cloud import bigquery

    logger.info("mimic start q=%s", question)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("SQL_GEN_MODEL", "gemini-2.5-flash")
    if not question.strip():
        return tool_error("question required")
    if not api_key:
        return tool_error("missing GEMINI_API_KEY or GOOGLE_API_KEY")

    limit_cap = int(top_n or 50)
    rules = (
        "You are a BigQuery SQL assistant for MIMIC-IV ED.\n"
        f"Return ONE StandardSQL query only in ```sql``` with LIMIT <= {limit_cap}.\n"
        "Use tables: physionet-data.mimiciv_ed.* and physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses.\n"
        "ED visit key is edstays.stay_id.\n"
    )
    examples = _load_examples_text()
    sys_msg = SystemMessage(content=rules + "\n" + examples)
    user_msg = HumanMessage(content=f"User question: {question}\nReturn only SQL in a fenced block.")
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)
    reply = llm.invoke([sys_msg, user_msg])
    sql = extract_sql((getattr(reply, "content", "") or "").strip())
    if not sql:
        return tool_error("no SQL produced")

    if dry_run:
        logger.info("mimic dry run")
        return f"SQL (dry run):\n```sql\n{sql}\n```"

    client = bigquery.Client(project=os.getenv("GCP_PROJECT_ID"))
    job_cfg = bigquery.QueryJobConfig(
        use_legacy_sql=False,
        maximum_bytes_billed=int(os.getenv("BQ_MAX_BYTES_BILLED", "1000000000")),
    )
    logger.info("mimic sql=%s", sql)
    df = client.query(sql, job_config=job_cfg).to_dataframe()
    preview = csv_preview(df, n=min(20, limit_cap))
    out = (
        f"SQL:\n```sql\n{sql}\n```\n"
        f"Rows returned: {len(df)}\n"
        f"Preview (CSV):\n```csv\n{preview}\n```"
    )
    logger.info("mimic done rows=%d", len(df))
    return out

# ---- agent ----

class LCAgent:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        logger.info("agent init")
        self.tools = [calculator, mimic]
        self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature).bind_tools(self.tools)
        self.system_text = (
            "You are a concise assistant. "
            "Use calculator for arithmetic. Use mimic for MIMIC ED analytics."
        )
        graph = StateGraph(MessagesState)
        graph.add_node("llm", self._llm_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", self._route, {"tools": "tools", END: END})
        graph.add_edge("tools", "llm")
        self.graph = graph.compile()
        logger.info("agent graph ready")

    def _llm_node(self, state: MessagesState):
        reply = self.llm.invoke(state["messages"])
        if getattr(reply, "tool_calls", None):
            logger.info("llm tool_calls")
        else:
            logger.info("llm reply: %s", (getattr(reply, "content", "") or "")[:200])
        return {"messages": [reply]}

    @staticmethod
    def _route(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and getattr(last, "tool_calls", None) else END

    @staticmethod
    def _to_lc_history(history):
        role_map = {"user": HumanMessage, "assistant": AIMessage}
        return [role_map[m.get("role")](m.get("content", "")) for m in history if m.get("role") in role_map]

    def ask(self, history, user_input: str) -> str:
        messages = [SystemMessage(self.system_text)] + self._to_lc_history(history)
        state = self.graph.invoke({"messages": messages})
        out = state.get("messages", [])
        return (getattr(out[-1], "content", "") or "").strip() if out else "(no response)"