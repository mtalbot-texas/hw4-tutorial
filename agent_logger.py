# agent_logger.py
import logging
import time

LOG_FILENAME = f"agentTutorial-{time.strftime('%m-%d-%H%M%S')}.log"

def get_logger() -> logging.Logger:
    logger = logging.getLogger("agent_logger")
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
    logger.propagate = False
    return logger