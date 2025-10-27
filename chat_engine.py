# chat_engine.py
import json
import logging
import os
import re
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from google.cloud import bigquery

from agent_logger import get_logger

logger = get_logger()
logger.setLevel(logging.INFO)

def _load_examples_text() -> str:
    data = json.loads((Path(__file__).parent / "mimic_examples.json").read_text(encoding="utf-8"))
    return "\n\n".join(
        f"Q: {ex['q']}\nSQL:\n```sql\n{ex['sql']}\n```" for ex in data["examples"]
    )

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

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    logger.info("calculator expr=%s", expression)
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"calc error: {e}"

@tool
def mimic(question: str = "", top_n: int = 50) -> str:
    """LLM only SQL over MIMIC IV ED (BigQuery). Returns SQL and CSV preview."""
    logger.info("mimic start q=%s", question)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("SQL_GEN_MODEL", "gemini-2.5-flash")
    limit_cap = int(top_n or 50)
    rules = (
        "You are a BigQuery SQL assistant for MIMIC IV ED.\n"
        f"Return one Standard SQL query in ```sql``` with LIMIT <= {limit_cap}.\n"
        "Use tables: physionet-data.mimiciv_ed.* and physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses.\n"
        "ED visit key is edstays.stay_id.\n"
    )
    examples = _load_examples_text()
    sys_msg = SystemMessage(content=rules + "\n" + examples)
    user_msg = HumanMessage(content=f"User question: {question}\nReturn only SQL in a fenced block.")
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)
    reply = llm.invoke([sys_msg, user_msg])
    sql = extract_sql((getattr(reply, "content", "") or "").strip())

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
    logger.info(out[preview])
    logger.info("mimic done rows=%d", len(df))
    return out


class LCAgent:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        logger.info("agent init")
        self.tools = [calculator, mimic]
        self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature).bind_tools(self.tools)
        self.system_text = (
            "You are a concise assistant. "
            "Use calculator for arithmetic. Use mimic for MIMIC-IV ED analytics."
        )
        graph = StateGraph(MessagesState)
        graph.add_node("llm", self._llm_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_edge(START, "llm")
       
        graph.add_conditional_edges(
            "llm",
            lambda s: "tools" if isinstance(s["messages"][-1], AIMessage) and getattr(s["messages"][-1], "tool_calls", None) else END,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", "llm")
        self.graph = graph.compile()
        logger.info("agent graph ready")

    def _llm_node(self, state: MessagesState):
        reply = self.llm.invoke(state["messages"])
        if getattr(reply, "tool_calls", None):
            for tc in reply.tool_calls:
                logger.info("llm tool_call name=%s args=%s", tc.get("name"), tc.get("args"))
        else:
            logger.info("llm text reply=%r", reply.content)
        return {"messages": [reply]}

    def ask(self, history, user_input: str) -> str:
        msgs = [SystemMessage(self.system_text)]
        for m in history:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content))
            elif role == "assistant":
                msgs.append(AIMessage(content))
        msgs.append(HumanMessage(user_input))
        out = self.graph.invoke({"messages": msgs}).get("messages", [])
        return (getattr(out[-1], "content", "") or "").strip() if out else "(no response)"
