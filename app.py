# app.py
import os
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Create logs directory and a single logfile name once
if "logfile" not in st.session_state:
    Path("logs").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%m_%d_%H_%M")  # MM_DD_HH_MM
    st.session_state["logfile"] = str(Path("logs") / f"agent_{ts}.log")

# Initialize root logging only once per session
if not st.session_state.get("logging_configured"):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # remove existing handlers (avoid duplicates on rerun)
    for h in list(root.handlers):
        root.removeHandler(h)

    # file handler
    fh = logging.FileHandler(st.session_state["logfile"], mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)

    # console handler (helps when something breaks very early)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(ch)

    logging.info("Logging initialized. File: %s", st.session_state["logfile"])
    st.session_state["logging_configured"] = True

# Load .env (OS env still wins if already set)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

# Import after logging is configured so the module inherits handlers
from chat_engine import LCAgent

st.title("Minimal Chat (calculator tool)")
st.caption(
    f"Logs: {st.session_state['logfile']}  "
    "Paste your Gemini API key from Google AI Studio (looks like AIza...).  "
    "You can also set GEMINI_API_KEY in a .env file."
)

# Read key from env or prompt
api_key = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or st.text_input("Gemini API key (Google AI Studio)", placeholder="AIza... paste here", type="password")
)
if not api_key:
    st.write("Enter a valid Gemini API key or set GEMINI_API_KEY in .env.")
    st.stop()

# Build or keep the agent in session
if "engine" not in st.session_state:
    logging.info("Creating LCAgent instance")
    st.session_state.engine = LCAgent(api_key=api_key)

# Simple chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Render prior turns
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Handle a new turn
prompt = st.chat_input("Type a message (for example: What is (3+5)*12?)")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        answer = st.session_state.engine.ask(st.session_state.history, prompt)
    except Exception as e:
        logging.exception("Agent error")
        answer = f"Error: {e}"
    st.session_state.history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
