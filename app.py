# app.py
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

def _configure_logging_once():
    if "logfile" in st.session_state and st.session_state.get("logging_configured"):
        return

    ts = datetime.now().strftime("%m-%d-%H:%M")
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"tutorial-{ts}.log"

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(str(logfile), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    root.setLevel(logging.DEBUG)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    st.session_state["logfile"] = str(logfile)
    st.session_state["logging_configured"] = True
    logging.info("Logging initialized. File: %s", logfile)

_configure_logging_once()

from chat_engine import LCAgent  # noqa: E402
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

st.title("Minimal Chat (calculator + mimic tool)")
st.caption(
    f"Logs: {st.session_state['logfile']}  "
    "Paste your Gemini API key from Google AI Studio (looks like AIza...).  "
    "You can also set GEMINI_API_KEY in a .env file."
)

api_key = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or st.text_input("Gemini API key (Google AI Studio)", placeholder="AIza... paste here", type="password")
)
if not api_key:
    st.write("Enter a valid Gemini API key or set GEMINI_API_KEY in .env.")
    st.stop()

if "engine" not in st.session_state:
    logging.info("Creating LCAgent instance")
    st.session_state.engine = LCAgent(api_key=api_key)

if "history" not in st.session_state:
    st.session_state.history = []

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type a message (e.g., Use mimic(top_n=2055)  or  What is (3+5)*12?)")
if prompt:
    logging.info("UI:prompt=%s", prompt)
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        answer = st.session_state.engine.ask(st.session_state.history, prompt)
    except Exception as e:
        logging.exception("Agent error")
        answer = f"Error: {e}"
    logging.info("UI:answer=%s", answer)
    st.session_state.history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)