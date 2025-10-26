# app.py
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

def setup_logging():
    if st.session_state.get("logging_configured"):
        return
    ts = datetime.now().strftime("%m-%d-%H%M")
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"tutorial-{ts}.log"

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(str(logfile), encoding="utf-8")
    ch = logging.StreamHandler(stream=sys.stdout)
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)
    root.setLevel(logging.INFO)

    st.session_state["logfile"] = str(logfile)
    st.session_state["logging_configured"] = True
    logging.info("logging file: %s", logfile)

setup_logging()

from chat_engine import LCAgent  

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

st.title("Minimal Chat: calculator + mimic")
st.caption(f"Logs: {st.session_state['logfile']}  Set GEMINI_API_KEY in .env or paste below.")

api_key = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or st.text_input("Gemini API key", placeholder="AIza...", type="password")
)
if not api_key:
    st.write("Enter a valid Gemini API key or set GEMINI_API_KEY in .env.")
    st.stop()

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
        logging.exception("agent error")
        answer = f"Error: {e}"
    st.session_state.history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)