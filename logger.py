import logging
import os

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Configure and return a logger for the project."""
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler("logs/auto_testgen.log", mode="a", encoding="utf-8")
        console = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        console.setFormatter(fmt)
        logger.addHandler(handler)
        logger.addHandler(console)
        logger.setLevel(level)
    return logger
