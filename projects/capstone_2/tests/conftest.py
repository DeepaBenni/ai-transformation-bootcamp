"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.audit.context import RunContext, run_context
from app.audit.trace import Tracer, new_run_id


@pytest.fixture
def run() -> Iterator[RunContext]:
    """Bind a throwaway run context with tracing to file only (never the DB)."""
    context = RunContext(run_id=new_run_id(), tracer=Tracer(new_run_id(), to_db=False))
    with run_context(context) as bound:
        yield bound
