# chat_engine.py
import logging
import re
import inspect
from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger("chat_engine")  # inherits handlers from root

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression. Supports digits, + - * / and parentheses.
    Example: "(3 + 5) * 12"
    """
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)]+", expression):
        return "Error: unsupported characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

class LCAgent:
    """
    Minimal LangChain agent wrapper using a ReAct agent with one tool (calculator).
    Adapts to different langgraph.create_react_agent signatures.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        logger.info("Building model %s", model)
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
        )
        self.tools = [calculator]
        self.system_text = (
            "You are a careful assistant. "
            "If a calculation is needed, call the calculator tool with the exact arithmetic expression. "
            "Otherwise answer directly. Keep responses concise."
        )

        # Detect which keyword the local langgraph supports
        try:
            sig = inspect.signature(create_react_agent)
            if "prompt" in sig.parameters:
                logger.info("create_react_agent supports prompt=...")
                self.agent = create_react_agent(model=self.llm, tools=self.tools, prompt=self.system_text)
                self._mode = "prompt"
            elif "state_modifier" in sig.parameters:
                logger.info("create_react_agent supports state_modifier=...")
                self.agent = create_react_agent(model=self.llm, tools=self.tools, state_modifier=self.system_text)
                self._mode = "state_modifier"
            else:
                logger.info("No prompt/state_modifier in signature. Will prepend SystemMessage at runtime.")
                self.agent = create_react_agent(model=self.llm, tools=self.tools)
                self._mode = "system_message"
        except Exception as e:
            logger.info("Signature check failed (%s). Will prepend SystemMessage at runtime.", e)
            self.agent = create_react_agent(model=self.llm, tools=self.tools)
            self._mode = "system_message"

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
        return msgs

    def ask(self, history: List[Dict[str, str]], user_input: str) -> str:
        messages = self._to_lc_history(history)

        # If we could not inject guidance via create_react_agent, prepend SystemMessage here
        if self._mode == "system_message":
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(self.system_text)] + messages

        # Ensure the last turn is the current user input
        if not messages or not isinstance(messages[-1], HumanMessage) or messages[-1].content != user_input:
            messages.append(HumanMessage(user_input))

        logger.info("Invoking agent with %d prior messages", len(messages) - 1)
        state = self.agent.invoke({"messages": messages})

        out = state.get("messages", [])
        if not out:
            logger.warning("No messages returned from agent")
            return "(no response)"
        last = out[-1]
        return getattr(last, "content", str(last))
