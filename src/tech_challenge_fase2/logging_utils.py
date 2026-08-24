"""Logging local sem dados em nivel de paciente."""

import logging
from pathlib import Path


def configure_logging(log_path: Path) -> logging.Logger:
    """Configura console e arquivo, substituindo handlers de execucoes anteriores."""

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tech_challenge_fase2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger

