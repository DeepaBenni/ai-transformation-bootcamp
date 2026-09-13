"""The append-only record of a run.

Every event is written twice on purpose: to traces/<run_id>.jsonl, which is the
machine-readable artefact R6 asks for, and to the audit_events table, which is
queryable and drives the UI and the cost report.

A failure to write a trace row must never fail the run it is describing, so
every write is wrapped.

Boundary: audit_events and runs are written here with raw SQL only. Estate and
approval writes go through the ORM elsewhere; this module never touches them.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.audit.cost import usd
from app.config import get_settings
from app.db.session import db_session
from app.logging_conf import get_logger

LOGGER = get_logger(__name__)

_INSERT_EVENT = text("""
    INSERT INTO audit_events
        (run_id, seq, ts, agent, event, tool, args_json, observation_json, model,
         prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms, error)
    VALUES
        (:run_id, :seq, :ts, :agent, :event, :tool, :args_json, :observation_json, :model,
         :prompt_tokens, :completion_tokens, :total_tokens, :cost_usd, :latency_ms, :error)
    """)

_MS_PER_SECOND: int = 1000


def new_run_id() -> str:
    """Return a short, sortable, unique run identifier."""
    return f"r_{datetime.now(UTC):%Y%m%d%H%M%S}_{uuid.uuid4().hex[:6]}"


class Tracer:
    """Records one run's events. One Tracer per run, never shared."""

    def __init__(self, run_id: str, *, to_db: bool = True) -> None:
        """Open a trace for `run_id`, creating its JSONL file path."""
        self.run_id = run_id
        self.to_db = to_db
        self.events: list[dict[str, Any]] = []
        self._seq = 0
        self._t0 = time.perf_counter()
        self._path: Path = get_settings().trace_dir / f"{run_id}.jsonl"

    # ---------------------------------------------------------------- write --
    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one event row. Never raises."""
        self._seq += 1
        prompt_tokens = int(fields.pop("prompt_tokens", 0))
        completion_tokens = int(fields.pop("completion_tokens", 0))

        row: dict[str, Any] = {
            "run_id": self.run_id,
            "seq": self._seq,
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "elapsed_ms": round((time.perf_counter() - self._t0) * _MS_PER_SECOND, 1),
            "event": event,
            "agent": fields.pop("agent", None),
            "tool": fields.pop("tool", None),
            "args": fields.pop("args", None),
            "observation": fields.pop("observation", None),
            "model": fields.pop("model", None),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": usd(prompt_tokens, completion_tokens),
            "latency_ms": int(fields.pop("latency_ms", 0)),
            "error": fields.pop("error", None),
        }
        row.update(fields)

        self.events.append(row)
        self._write_jsonl(row)
        if self.to_db:
            self._write_db(row)
        return row

    def _write_jsonl(self, row: dict[str, Any]) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        except OSError as exc:
            LOGGER.warning("Trace file write failed for %s: %s", self.run_id, exc)

    def _write_db(self, row: dict[str, Any]) -> None:
        try:
            with db_session() as session:
                session.execute(
                    _INSERT_EVENT,
                    {
                        "run_id": row["run_id"],
                        "seq": row["seq"],
                        # MySQL DATETIME(3) is naive; strip the tzinfo we added above.
                        "ts": datetime.now(UTC).replace(tzinfo=None),
                        "agent": row["agent"],
                        "event": row["event"],
                        "tool": row["tool"],
                        "args_json": (
                            json.dumps(row["args"], default=str)
                            if row["args"] is not None
                            else None
                        ),
                        "observation_json": (
                            json.dumps(row["observation"], default=str)
                            if row["observation"] is not None
                            else None
                        ),
                        "model": row["model"],
                        "prompt_tokens": row["prompt_tokens"],
                        "completion_tokens": row["completion_tokens"],
                        "total_tokens": row["total_tokens"],
                        "cost_usd": row["cost_usd"],
                        "latency_ms": row["latency_ms"],
                        "error": str(row["error"]) if row["error"] else None,
                    },
                )
        except Exception as exc:  # broad on purpose: tracing must never break the run
            LOGGER.warning(
                "Trace DB write failed for %s seq %s: %s", row["run_id"], row["seq"], exc
            )

    # ----------------------------------------------------------------- read --
    @property
    def totals(self) -> dict[str, Any]:
        """Aggregate tokens, cost and elapsed time for the run."""
        return {
            "events": len(self.events),
            "total_tokens": sum(event["total_tokens"] for event in self.events),
            "cost_usd": round(sum(event["cost_usd"] for event in self.events), 6),
            "elapsed_ms": round((time.perf_counter() - self._t0) * _MS_PER_SECOND, 1),
        }
