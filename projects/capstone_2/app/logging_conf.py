"""Logging configuration.

Call configure_logging() exactly once, from the process entry point - the CLI,
the FastAPI startup hook, or a test fixture. Library modules only ever call
logging.getLogger(__name__); they never configure handlers.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

from app.config import get_settings

_FORMAT: Final[str] = "%(asctime)s  %(levelname)-7s  %(name)-24s  %(message)s"
_DATEFMT: Final[str] = "%H:%M:%S"

# Module-level guard: configure_logging() may be imported by several entry points
# (CLI, API, tests) but must only install handlers once.
_configured: bool = False


def configure_logging(level: str | None = None) -> None:
    """Install a single stdout handler on the root logger. Safe to call twice."""
    global _configured
    if _configured:
        return

    logging.basicConfig(
        level=(level or get_settings().log_level).upper(),
        format=_FORMAT,
        datefmt=_DATEFMT,
        stream=sys.stdout,
    )
    # These libraries are chatty at INFO and drown out our own messages.
    for noisy in ("httpx", "httpcore", "openai", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, named for the calling module via __name__."""
    return logging.getLogger(name)
