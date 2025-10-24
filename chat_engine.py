# chat_engine.py
import logging
import re
import os
import time
from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# LangGraph pieces
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

# --- Added: initialize file logging with startup timestamp ---
logging.basicConfig(filename=f"agentTutorial-{time.strftime('%m-%d-%H%S')}.log", level=logging.INFO)

logger = logging.getLogger("chat_engine")  # inherits root handlers


def _excerpt(s: str, n: int = 400) -> str:
    try:
        s = str(s)
    except Exception:
        return "<non-string>"
    return s if len(s) <= n else s[:n] + "…"


def _content_to_text(content) -> str:
    # Gemini sometimes returns a list of parts; flatten text parts.
    if isinstance(content, list):
        try:
            return "".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        except Exception:
            return "<unprintable parts>"
    return str(content)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression containing digits, + - * / and parentheses.

    Parameters
    ----------
    expression : str
        Example: "(3 + 5) * 12"

    Returns
    -------
    str
        The numeric result as a string, or "Error: ..." if invalid.
    """
    logger.info("calculator:start expr=%s", _excerpt(expression, 120))
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)]+", expression):
        logger.warning("calculator:unsupported_chars expr=%s", _excerpt(expression, 120))
        return "Error: unsupported characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        logger.info("calculator:success result=%s", result)
        return str(result)
    except Exception as e:
        logger.exception("calculator:error")
        return f"Error: {e}"



@tool
def mimic(question: str = "", top_n: int = 50, dry_run: bool = False) -> str:
    """
    LLM-only SQL over MIMIC-IV ED (BigQuery StandardSQL), with guardrails:
    - physionet-data.* only
    - aggregate-only (no raw identifiers in SELECT/GROUP BY)
    - LIMIT ≤ top_n

    Returns SQL + a small CSV preview (or SQL only if dry_run).
    On error: 'TOOL_ERROR[mimic]: <Type>: <message>'
    """
    import os, re, time
    from google.cloud import bigquery
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    started = time.monotonic()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("SQL_GEN_MODEL", "gemini-2.5-flash")

    if not question.strip():
        return "TOOL_ERROR[mimic]: ValueError: question is required."
    if not api_key:
        return "TOOL_ERROR[mimic]: RuntimeError: Missing GEMINI_API_KEY/GOOGLE_API_KEY."

    limit_cap = int(top_n or 50)

    # ---------- tiny helpers ----------
    def _extract_sql(text: str) -> str:
        m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
        sql = (m.group(1) if m else text).strip()
        return re.sub(r";\s*$", "", sql)

    def _enforce_limit(sql: str, cap: int) -> str:
        return re.sub(r"\bLIMIT\s+(\d+)\b",
                      lambda m: f"LIMIT {cap}" if int(m.group(1)) > cap else m.group(0),
                      sql, flags=re.I) if re.search(r"\bLIMIT\b", sql, re.I) else f"{sql}\nLIMIT {cap}"

    def _sanitize(sql: str, cap: int) -> str:
        # Block DDL/DML
        if re.search(r"\b(CREATE|ALTER|DROP|TRUNCATE|INSERT|DELETE|UPDATE|MERGE|GRANT|REVOKE|EXECUTE\s+IMMEDIATE)\b", sql, re.I):
            raise RuntimeError("Unsafe SQL: DDL/DML not allowed.")

        # Project/table restriction
        for _, tok in re.findall(r"\b(FROM|JOIN|INTO|UPDATE|TABLE)\s+([`a-zA-Z0-9_.-]+)", sql, flags=re.I):
            name = tok.strip("`").strip()
            if name and not name.startswith("(") and not name.upper().startswith("UNNEST"):
                if not name.startswith("physionet-data."):
                    raise RuntimeError(f"Cross-project or unqualified reference not allowed: {name}")

        # Aggregate-only checks
        sel_m = re.search(r"(?is)\bSELECT\b(.*?)\bFROM\b", sql)
        if not sel_m:
            raise RuntimeError("Malformed SQL: missing SELECT ... FROM.")
        sel = sel_m.group(1)

        # No SELECT * or table.* (except inside aggregates)
        if re.search(r"(?is)\bSELECT\s+(?:DISTINCT\s+)?\*", sql) or re.search(r"(?is)\bSELECT\b.*\.\*", sql):
            raise RuntimeError("Aggregate-only policy: wildcard SELECT is not allowed.")

        if not re.search(r"\b(COUNT|SUM|AVG|MIN|MAX|APPROX_COUNT_DISTINCT|ARRAY_AGG|STRING_AGG)\s*\(", sel, re.I):
            raise RuntimeError("Aggregate-only policy: SELECT must include an aggregate.")

        id_cols = r"(subject_id|stay_id|hadm_id|icustay_id|patient_id|anchor_year|anchor_age|anchor_year_group)"
        sel_stripped = re.sub(r"COUNT\s*\(\s*(?:DISTINCT\s+)?[^)]*\)", "", sel, flags=re.I)
        if re.search(id_cols, sel_stripped, re.I):
            raise RuntimeError("Aggregate-only policy: identifiers may appear only inside COUNT(...).")

        gb_m = re.search(r"(?is)\bGROUP\s+BY\b(.*?)(?:\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|$)", sql)
        if gb_m and re.search(id_cols, gb_m.group(1), re.I):
            raise RuntimeError("Aggregate-only policy: GROUP BY identifiers is not allowed.")

        return _enforce_limit(sql, cap)

    def _csv_preview(df, n: int) -> str:
        try:
            return df.head(n).to_csv(index=False)
        except Exception:
            return "(preview unavailable)"

    # ---------- generate SQL with LLM ----------
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)
        sys_msg = SystemMessage(
            content=(
                "You are a BigQuery SQL assistant for MIMIC-IV emergency department (ED) data.\n"
                "Return ONE StandardSQL query only, inside ```sql ...``` with no explanation.\n"
                "- Use fully-qualified tables ONLY in physionet-data.\n"
                "  Prefer: physionet-data.mimiciv_ed.edstays, diagnosis, triage, vitalsign;\n"
                "          physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses.\n"
                "- Aggregate-only: do not SELECT identifiers (subject_id, stay_id, hadm_id, icustay_id, etc.)\n"
                "  except inside COUNT(...) / COUNT(DISTINCT ...). Do not GROUP BY identifiers.\n"
                f"- Include a LIMIT ≤ {limit_cap}. Use StandardSQL."
            )
        )
        user_msg = HumanMessage(content=f"User question: {question}\nReturn only the SQL in a fenced code block.")
        reply = llm.invoke([sys_msg, user_msg])
        sql = _extract_sql(_content_to_text(getattr(reply, "content", "")))  # _content_to_text is defined above in your module

        # --- Added: log generated SQL BEFORE sanitization (1/2) ---
        logger.info("mimic:generated_sql_before_sanitize sql=%s", sql)

        if not sql:
            raise RuntimeError("LLM failed to produce SQL.")
        sql = _sanitize(sql, limit_cap)
        if dry_run:
            return f"SQL (dry run):\n```sql\n{sql}\n```"
    except Exception as e:
        return f"TOOL_ERROR[mimic]: {type(e).__name__}: {e}"

    # ---------- execute ----------
    try:
        client = bigquery.Client(project=os.getenv("GCP_PROJECT_ID", "indigo-proxy-472718-m1"))
        job_cfg = bigquery.QueryJobConfig(
            use_legacy_sql=False,
            maximum_bytes_billed=int(os.getenv("BQ_MAX_BYTES_BILLED", "1000000000")),  # 1GB default
        )

        # --- Added: log SQL AFTER sanitization, right before execution (2/2) ---
        logger.info("mimic:generated_sql_after_sanitize sql=%s", sql)

        df = client.query(sql, job_config=job_cfg).to_dataframe()
        preview = _csv_preview(df, n=min(20, limit_cap))
        elapsed = time.monotonic() - started
        return (
            f"SQL:\n```sql\n{sql}\n```\n"
            f"Rows returned: {len(df)}\n"
            f"Preview (CSV):\n```csv\n{preview}\n```\n"
            f"(elapsed {elapsed:.2f}s)"
        )
    except Exception as e:
        return f"TOOL_ERROR[mimic]: {type(e).__name__}: {e}"


class LCAgent:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        logger.info("LCAgent:init model=%s temp=%s", model, temperature)

        base_llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
        )

        self.tools = [calculator, mimic]
        logger.info("LCAgent:tools %s", [t.name for t in self.tools])
        self.llm = base_llm.bind_tools(self.tools)

        self.system_text = (
            "You are a careful assistant. "
            "If a calculation is needed, call the calculator tool. "
            "If the user asks about patients or MIMIC analytics, call the mimic tool (aggregate-only). "
            "If ANY tool returns a string starting with 'TOOL_ERROR[', respond with that string verbatim. "
            "Keep responses concise."
        )

        graph = StateGraph(MessagesState)

        def llm_node(state: MessagesState):
            logger.info("llm_node:enter history_len=%d", len(state["messages"]))
            reply = self.llm.invoke(state["messages"])

            tool_calls = getattr(reply, "tool_calls", None)
            if tool_calls:
                try:
                    summary = [(tc.get("name"), tc.get("args")) for tc in tool_calls]  # type: ignore[index]
                except Exception:
                    summary = "<unserializable tool_calls>"
                logger.info("llm_node:tool_calls %s", summary)
            else:
                logger.info(
                    "llm_node:no_tool_call text=%s",
                    _excerpt(_content_to_text(getattr(reply, "content", "")), 300),
                )
            return {"messages": [reply]}

        tool_node = ToolNode(self.tools)

        def route(state: MessagesState):
            last = state["messages"][-1]
            go = isinstance(last, AIMessage) and getattr(last, "tool_calls", None)
            logger.info("router:go_tools=%s", bool(go))
            return "tools" if go else END

        graph.add_node("llm", llm_node)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", route, {"tools": "tools", END: END})
        graph.add_edge("tools", "llm")

        self.graph = graph.compile()
        logger.info("LCAgent:graph_compiled")

    @staticmethod
    def _to_lc_history(history: List[Dict[str, str]]):
        msgs = []
        for m in history:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content))
            elif role == "assistant":
                msgs.append(AIMessage(content))
        logger.info("to_lc_history:converted=%d", len(msgs))
        return msgs

    @staticmethod
    def _log_tail(messages):
        logger.info("trace:tail_messages=%d", len(messages))
        for i, m in enumerate(messages[-6:], start=max(0, len(messages) - 6)):
            kind = m.__class__.__name__
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                logger.info("trace[%d]:%s tool_calls=%s", i, kind, [(tc.get("name"), tc.get("args")) for tc in m.tool_calls])  # type: ignore[attr-defined]
            elif isinstance(m, ToolMessage):
                logger.info("trace[%d]:ToolMessage name=%s content=%s", i, getattr(m, "name", "?"), _excerpt(_content_to_text(m.content), 300))
            else:
                logger.info("trace[%d]:%s content=%s", i, kind, _excerpt(_content_to_text(getattr(m, "content", "")), 300))

    def ask(self, history: List[Dict[str, str]], user_input: str) -> str:
        logger.info("ask:start prompt=%s", _excerpt(user_input, 240))
        messages = self._to_lc_history(history)

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(self.system_text)] + messages
            logger.info("ask:system_message_prepended")

        if not messages or not isinstance(messages[-1], HumanMessage) or messages[-1].content != user_input:
            messages.append(HumanMessage(user_input))
            logger.info("ask:user_turn_appended")

        logger.info("ask:graph_invoke history_len=%d", len(messages) - 1)
        state = self.graph.invoke({"messages": messages})

        out = state.get("messages", [])
        self._log_tail(out)

        if not out:
            logger.warning("ask:no_messages_returned")
            return "(no response)"

        # If any tool emitted TOOL_ERROR, surface it verbatim to the UI.
        for m in reversed(out):
            text = _content_to_text(getattr(m, "content", ""))
            if isinstance(text, str) and text.startswith("TOOL_ERROR["):
                logger.info("ask:returning_tool_error=%s", _excerpt(text, 300))
                return text

        last = out[-1]
        text = _content_to_text(getattr(last, "content", ""))
        logger.info("ask:returning_ai_text=%s", _excerpt(text, 300))
        return text
