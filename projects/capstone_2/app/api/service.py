"""Orchestrates graph runs across HTTP requests.

A run that pauses for approval spans two requests. The graph's own state is
restored by the checkpointer; the Tracer is not checkpointable, so it is held in
a small in-process registry keyed by run_id.

Consequence, stated plainly rather than hidden: this service must run with a
single worker, and a process restart loses the tracer for any run that is
mid-approval. The MySQL record of that run survives; its remaining trace rows do
not. This is listed as residual risk in the governance note.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from sqlalchemy import text

from app.agents.conversation import TriageResult, triage
from app.agents.graph import build_graph
from app.agents.state import TicketIn, initial_state
from app.api import sessions
from app.audit.context import RunContext, run_context
from app.audit.trace import Tracer, new_run_id
from app.config import ROOT, get_settings
from app.db.session import db_session
from app.logging_conf import get_logger
from app.render import visible_message

LOGGER = get_logger(__name__)

_BODY_PREVIEW_CHARS: int = 400
_STOP_REASON_CHARS: int = 500
_HISTORY_TURNS: int = 20
_CHECKPOINT_FILE = ROOT / "checkpoints.sqlite"


@dataclass
class RunHandle:
    """Everything about an in-flight run that the checkpointer does not hold."""

    run_id: str
    ticket_id: str
    tracer: Tracer
    started: float = field(default_factory=time.perf_counter)
    pending: dict[str, Any] | None = None


@dataclass
class Conversation:
    """One chat session's memory: what was said, and what to carry forward.

    In-process, like the run registry - lost on a service restart. The MySQL
    record of any ticket started from the conversation survives; the chit-chat
    around it does not. Listed as residual risk in the governance note.
    """

    history: list[dict[str, str]] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)
    tickets: int = 0

    def add(self, role: str, content: str) -> None:
        """Append one turn and cap the transcript length."""
        self.history.append({"role": role, "content": content})
        del self.history[:-_HISTORY_TURNS]


class OpsMateService:
    """Owns the compiled graph, the checkpointer and the run registry."""

    def __init__(self) -> None:
        """Compile the graph once, with a durable SQLite checkpointer."""
        connection = sqlite3.connect(str(_CHECKPOINT_FILE), check_same_thread=False)
        self._saver = SqliteSaver(connection)
        self._saver.setup()
        self._graph = build_graph(checkpointer=self._saver)
        self._runs: dict[str, RunHandle] = {}
        self._conversations: dict[str, Conversation] = {}
        LOGGER.info("Graph compiled with SQLite checkpointer at %s", _CHECKPOINT_FILE)

    # ------------------------------------------------------------- helpers --
    @property
    def active_runs(self) -> int:
        """Return the number of runs currently suspended awaiting a human."""
        return sum(1 for handle in self._runs.values() if handle.pending is not None)

    def _config(self, run_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": run_id},
            "recursion_limit": get_settings().recursion_limit,
        }

    def _handle(self, run_id: str) -> RunHandle:
        handle = self._runs.get(run_id)
        if handle is None:
            raise KeyError(run_id)
        return handle

    # ------------------------------------------------------------- chat --
    def _conversation(self, session_id: str) -> Conversation:
        """Return the in-process conversation, hydrating it from MySQL if needed."""
        convo = self._conversations.get(session_id)
        if convo is not None:
            return convo

        sessions.ensure_session(session_id)
        stored = sessions.get_messages(session_id)
        convo = Conversation(
            history=[{"role": row["role"], "content": row["content"]} for row in stored][
                -_HISTORY_TURNS:
            ],
            facts=(sessions.get_session(session_id) or {}).get("facts") or {},
            tickets=sum(1 for row in stored if row["kind"] == "run"),
        )
        self._conversations[session_id] = convo
        return convo

    def chat(self, session_id: str, message: str) -> dict[str, Any]:
        """Route one chat message: small talk, a scope refusal, or a ticket run.

        Every turn is persisted to MySQL (chat_sessions / chat_messages) so the
        session history side panel can reload it after a restart. The user's name
        and the recent turns are threaded into whichever path is taken.
        """
        convo = self._conversation(session_id)

        # The classifier call needs a run context, but a greeting is not a run:
        # give it a throwaway tracer that writes nothing to MySQL.
        scratch = Tracer(new_run_id(), to_db=False)
        with run_context(RunContext(run_id=scratch.run_id, tracer=scratch)):
            verdict: TriageResult = triage(message, convo.history, convo.facts)

        if verdict.remember:
            convo.facts.update(verdict.remember)
            sessions.save_facts(session_id, convo.facts)
        convo.add("user", message)
        sessions.append_message(session_id, "user", message, kind="text")

        if verdict.kind == "ticket":
            convo.tickets += 1
            run = self.start(
                message,
                ticket_id=f"CHAT-{convo.tickets:03d}",
                reporter=convo.facts.get("user_name"),
                history=list(convo.history),
            )
            visible = run.get("user_visible_message") or run.get("message") or ""
            convo.add("assistant", visible)
            response = {
                "session_id": session_id,
                "kind": "ticket",
                "reply": visible,
                "run": run,
            }
            sessions.append_message(
                session_id,
                "assistant",
                visible,
                kind="run",
                run_id=run.get("run_id"),
                payload=response,
            )
            return response

        convo.add("assistant", verdict.reply)
        response = {
            "session_id": session_id,
            "kind": verdict.kind,
            "reply": verdict.reply,
            "run": None,
        }
        sessions.append_message(
            session_id, "assistant", verdict.reply, kind="chat", payload=response
        )
        return response

    # -------------------------------------------------------- session api --
    @staticmethod
    def new_session() -> dict[str, Any]:
        """Create a fresh, empty session and return its metadata."""
        return sessions.ensure_session(new_run_id())

    @staticmethod
    def list_sessions() -> list[dict[str, Any]]:
        """List every chat session, most recently used first."""
        return sessions.list_sessions()

    @staticmethod
    def session_detail(session_id: str) -> dict[str, Any]:
        """Return a session's metadata and its full message list.

        Raises:
            KeyError: if the session does not exist.
        """
        meta = sessions.get_session(session_id)
        if meta is None:
            raise KeyError(session_id)
        return {**meta, "messages": sessions.get_messages(session_id)}

    @staticmethod
    def rename_session(session_id: str, title: str) -> dict[str, Any]:
        """Rename a session.

        Raises:
            KeyError: if the session does not exist.
        """
        updated = sessions.rename_session(session_id, title)
        if updated is None:
            raise KeyError(session_id)
        return updated

    # --------------------------------------------------------------- runs --
    def start(
        self,
        body: str,
        ticket_id: str,
        reporter: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Begin a run. Returns as soon as it finishes or suspends for approval."""
        settings = get_settings()
        run_id = new_run_id()
        handle = RunHandle(run_id=run_id, ticket_id=ticket_id, tracer=Tracer(run_id))
        self._runs[run_id] = handle

        self._insert_run(run_id, ticket_id, body)

        with run_context(RunContext(run_id=run_id, tracer=handle.tracer)):
            handle.tracer.log(
                "run_start",
                observation={"ticket_id": ticket_id, "body": body[:_BODY_PREVIEW_CHARS]},
            )
            state = initial_state(
                run_id,
                TicketIn(ticket_id=ticket_id, body=body, reporter=reporter),
                settings.max_hops,
                history=history,
            )
            result = self._graph.invoke(state, self._config(run_id))
        return self._settle(handle, result)

    def resume(self, run_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        """Resume a suspended run with a human decision."""
        handle = self._handle(run_id)
        if handle.pending is None:
            return self.snapshot(run_id)

        with run_context(RunContext(run_id=run_id, tracer=handle.tracer)):
            result = self._graph.invoke(Command(resume=decision), self._config(run_id))
        return self._settle(handle, result)

    def _settle(self, handle: RunHandle, result: dict[str, Any]) -> dict[str, Any]:
        """Record whether the run suspended again or finished, then snapshot it."""
        if "__interrupt__" in result:
            handle.pending = result["__interrupt__"][0].value
            self._update_run(handle, status="awaiting_approval")
        else:
            handle.pending = None
            self._update_run(handle, status="done")
        return self.snapshot(handle.run_id)

    def snapshot(self, run_id: str) -> dict[str, Any]:
        """Return the current view of a run, suspended or finished."""
        handle = self._handle(run_id)
        values = self._graph.get_state(self._config(run_id)).values
        final = values.get("final")
        totals = handle.tracer.totals

        view: dict[str, Any] = {
            "run_id": run_id,
            "status": "awaiting_approval" if handle.pending else "done",
            "path": values.get("path", "undecided"),
            "outcome": final.outcome if final else None,
            "message": final.message if final else None,
            "citations": final.citations if final else [],
            "question": (final.message if final and final.outcome == "NEEDS_INFO" else None),
            "pending": self._pending_payload(handle),
            "changes": self.changes(run_id),
            "hops": values.get("hops", 0),
            "stop_reason": values.get("stop_reason"),
            "route_history": [
                step.model_dump() if hasattr(step, "model_dump") else dict(step)
                for step in values.get("route_history", [])
            ],
            "total_tokens": totals["total_tokens"],
            "cost_usd": totals["cost_usd"],
            "latency_ms": int(totals["elapsed_ms"]),
        }
        view["user_visible_message"] = visible_message(view)
        return view

    @staticmethod
    def _pending_payload(handle: RunHandle) -> dict[str, Any] | None:
        if handle.pending is None:
            return None
        pending = handle.pending
        return {
            "action": pending.get("action", ""),
            "args": pending.get("args", {}),
            "reason": pending.get("reason", ""),
            "citations": pending.get("citations", []),
            "evidence": pending.get("evidence", {}),
            "rejected": pending.get("rejected", []),
        }

    # ------------------------------------------------------------ database --
    @staticmethod
    def _insert_run(run_id: str, ticket_id: str, body: str) -> None:
        with db_session() as session:
            session.execute(
                text(
                    "INSERT INTO runs (run_id, ticket_id, ticket_body, status) "
                    "VALUES (:run_id, :ticket_id, :body, 'running')"
                ),
                {"run_id": run_id, "ticket_id": ticket_id, "body": body},
            )

    def _update_run(self, handle: RunHandle, *, status: str) -> None:
        values = self._graph.get_state(self._config(handle.run_id)).values
        totals = handle.tracer.totals
        with db_session() as session:
            session.execute(
                text(
                    "UPDATE runs SET path = :path, status = :status, stop_reason = :stop, "
                    "total_tokens = :tokens, cost_usd = :cost, latency_ms = :latency, "
                    "ended_at = CASE WHEN :status = 'done' THEN NOW() ELSE NULL END "
                    "WHERE run_id = :run_id"
                ),
                {
                    "run_id": handle.run_id,
                    "path": values.get("path", "undecided"),
                    "status": status,
                    "stop": (values.get("stop_reason") or "")[:_STOP_REASON_CHARS] or None,
                    "tokens": totals["total_tokens"],
                    "cost": totals["cost_usd"],
                    "latency": int(totals["elapsed_ms"]),
                },
            )

    @staticmethod
    def ticket_body(run_id: str) -> str:
        """Return the original ticket text for a run, or '' if it is unknown."""
        with db_session() as session:
            row = session.execute(
                text("SELECT ticket_body FROM runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).scalar()
        return str(row) if row else ""

    @staticmethod
    def changes(run_id: str) -> list[dict[str, Any]]:
        """Return the state changes actually executed in a run."""
        with db_session() as session:
            rows = (
                session.execute(
                    text(
                        "SELECT action, args_json, approver, result, executed_at "
                        "FROM action_log WHERE run_id = :run_id ORDER BY id"
                    ),
                    {"run_id": run_id},
                )
                .mappings()
                .all()
            )
        return [{**row, "executed_at": str(row["executed_at"])} for row in rows]

    @staticmethod
    def trace(run_id: str) -> list[dict[str, Any]]:
        """Return the full audit trail for a run, JSON fields already decoded."""
        with db_session() as session:
            rows = (
                session.execute(
                    text(
                        "SELECT seq, ts, agent, event, tool, args_json, observation_json, "
                        "total_tokens, cost_usd, latency_ms, error "
                        "FROM audit_events WHERE run_id = :run_id ORDER BY seq"
                    ),
                    {"run_id": run_id},
                )
                .mappings()
                .all()
            )

        events: list[dict[str, Any]] = []
        for row in rows:
            events.append(
                {
                    "seq": row["seq"],
                    "ts": str(row["ts"]),
                    "agent": row["agent"],
                    "event": row["event"],
                    "tool": row["tool"],
                    "args": json.loads(row["args_json"]) if row["args_json"] else None,
                    "observation": (
                        json.loads(row["observation_json"]) if row["observation_json"] else None
                    ),
                    "total_tokens": row["total_tokens"],
                    "cost_usd": float(row["cost_usd"]),
                    "latency_ms": row["latency_ms"],
                    "error": row["error"],
                }
            )
        return events

    @staticmethod
    def list_runs(limit: int = 50) -> list[dict[str, Any]]:
        """Return recent runs, newest first."""
        with db_session() as session:
            rows = (
                session.execute(
                    text(
                        "SELECT run_id, ticket_id, path, status, stop_reason, total_tokens, "
                        "cost_usd, latency_ms, created_at FROM runs "
                        "ORDER BY created_at DESC LIMIT :limit"
                    ),
                    {"limit": limit},
                )
                .mappings()
                .all()
            )
        return [
            {**row, "cost_usd": float(row["cost_usd"]), "created_at": str(row["created_at"])}
            for row in rows
        ]
