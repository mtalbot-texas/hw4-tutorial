# app.py
import os
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent_logger import get_logger

# Create or reuse the shared logger first, regardless of entry point.
logger = get_logger()

def _current_logfile(logger: logging.Logger) -> str:
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            return getattr(h, "baseFilename", "")
    return ""

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

st.title("Minimal Chat: calculator + mimic")

# Save and show the logfile path
st.session_state["logfile"] = _current_logfile(logger)
st.caption(f"Logs: {st.session_state['logfile']}  Set GEMINI_API_KEY in .env or paste below.")

api_key = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or st.text_input("Gemini API key", placeholder="AIza...", type="password")
)
if not api_key:
    st.write("Enter a valid Gemini API key or set GEMINI_API_KEY in .env.")
    st.stop()

from chat_engine import LCAgent 

if "engine" not in st.session_state:
    st.session_state.engine = LCAgent(api_key=api_key)

if "history" not in st.session_state:
    st.session_state.history = []

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type a message, e.g., What is (3+5)*12? or Use mimic(top_n=100)")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        answer = st.session_state.engine.ask(st.session_state.history, prompt)
    except Exception as e:
        logger.exception("agent error")
        answer = f"Error: {e}"
    st.session_state.history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)