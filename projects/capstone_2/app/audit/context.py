"""The ambient run context.

Tools need to know which run they belong to. Threading a run_id through every
signature would pollute the tool schemas the model sees, so it lives in a
context variable instead - set once per run, read anywhere.

Boundary: this module only stores and retrieves the current run. It does not
create run ids or tracers (that is app/audit/trace.py).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.audit.trace import Tracer


@dataclass(frozen=True)
class RunContext:
    """Identity and tracing for one ticket run."""

    run_id: str
    tracer: Tracer


# A ContextVar (not a global) so concurrent runs, including async tasks, each see
# their own context rather than racing on a shared value.
_CURRENT: ContextVar[RunContext | None] = ContextVar("opsmate_run", default=None)


@contextmanager
def run_context(context: RunContext) -> Iterator[RunContext]:
    """Bind a run context for the duration of the block."""
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


def current_run() -> RunContext:
    """Return the active run context.

    Raises:
        RuntimeError: if called outside a run. A tool that reaches this has been
            invoked without a run, which is a bug worth failing loudly on.
    """
    context = _CURRENT.get()
    if context is None:
        raise RuntimeError("No active run context. Wrap the call in run_context(RunContext(...)).")
    return context
