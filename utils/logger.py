"""
utils/logger.py
---------------
Centralised logging configuration for PayPilot AI.

Purpose
-------
Give the application one clean, predictable console log format without
touching Streamlit's own logging or the root logger.

Design rules
------------
* Configures ONLY the "paypilot" logger namespace.  Third-party loggers
  (streamlit, tornado, PIL, …) keep their default behaviour.
* Idempotent — calling configure_logging() repeatedly never duplicates
  handlers (Streamlit reruns scripts constantly; import-time side effects
  must be safe).
* Never logs secrets or credentials.

Public API
----------
    configure_logging(level)   -> logging.Logger   (the "paypilot" logger)
    get_logger(name)            -> logging.Logger   (child of "paypilot")
    set_log_level(level)        -> None
    remove_handlers()           -> None            (mainly for tests)
    LOG_FORMAT / DATE_FORMAT    -> module constants
"""

from __future__ import annotations

import logging
from typing import Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGGER_NAME: str = "paypilot"

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

LogLevel = Union[int, str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(level: LogLevel = logging.INFO) -> logging.Logger:
    """
    Set up (or return the already-configured) application logger.

    Safe to call on every script run: existing handlers on the "paypilot"
    logger are reused, never duplicated.

    Parameters
    ----------
    level : logging level constant ("DEBUG", "INFO", logging.WARNING, …)

    Returns
    -------
    The configured "paypilot" logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_coerce_level(level))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)

    # Keep our messages out of the root logger so they never appear twice
    # (Streamlit already installs its own root handlers).
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger of the "paypilot" namespace.

    >>> get_logger("pages").name
    'paypilot.pages'

    Names already starting with "paypilot" are passed through unchanged.
    """
    if name.startswith(LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def set_log_level(level: LogLevel) -> None:
    """Change the application log level at runtime."""
    logging.getLogger(LOGGER_NAME).setLevel(_coerce_level(level))


def remove_handlers() -> None:
    """Detach all handlers from the "paypilot" logger (used by tests)."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _coerce_level(level: LogLevel) -> int:
    """
    Accept either a logging constant (logging.INFO) or its string name
    ("INFO") and always return an int level.
    """
    if isinstance(level, str):
        return logging.getLevelName(level.upper())
    return int(level)
