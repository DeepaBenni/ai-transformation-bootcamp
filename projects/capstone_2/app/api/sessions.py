"""Persistent chat-session history.

The conversation side panel needs sessions to survive a service restart, so every
turn is written to MySQL (`chat_sessions` / `chat_messages`) and the list of
session ids is mirrored to a local JSON file (`chat_sessions.json`). The graph
checkpointer persists run *state*, not the conversation transcript - that is this
module's job.

Boundary: reads and writes the two chat tables and the local index file. No graph,
no model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.config import ROOT
from app.db.session import db_session
from app.logging_conf import get_logger

LOGGER = get_logger(__name__)

# A file-based record of every session id, alongside the database.
INDEX_FILE: Path = ROOT / "chat_sessions.json"

_TITLE_MAX = 200


def _row_to_session(row: Any) -> dict[str, Any]:
    facts = row["facts"]
    if isinstance(facts, str):
        facts = json.loads(facts)
    return {
        "session_id": row["session_id"],
        "seq": row["seq"],
        "title": row["title"],
        "facts": facts or {},
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return one session's metadata, or None if it does not exist."""
    with db_session() as session:
        row = (
            session.execute(
                text(
                    "SELECT session_id, seq, title, facts, created_at, updated_at "
                    "FROM chat_sessions WHERE session_id = :sid"
                ),
                {"sid": session_id},
            )
            .mappings()
            .first()
        )
    return _row_to_session(row) if row else None


def list_sessions() -> list[dict[str, Any]]:
    """Return every session, most recently used first, with its message count."""
    with db_session() as session:
        rows = (
            session.execute(
                text(
                    "SELECT s.session_id, s.seq, s.title, s.facts, s.created_at, s.updated_at, "
                    "COUNT(m.id) AS messages "
                    "FROM chat_sessions s "
                    "LEFT JOIN chat_messages m ON m.session_id = s.session_id "
                    "GROUP BY s.session_id "
                    "ORDER BY s.updated_at DESC"
                )
            )
            .mappings()
            .all()
        )
    return [{**_row_to_session(row), "message_count": int(row["messages"])} for row in rows]


def ensure_session(session_id: str) -> dict[str, Any]:
    """Return the session, creating it with a default 'Chat <n>' title if new."""
    existing = get_session(session_id)
    if existing is not None:
        return existing

    with db_session() as session:
        next_seq = int(
            session.execute(text("SELECT COALESCE(MAX(seq), 0) + 1 FROM chat_sessions")).scalar()
        )
        session.execute(
            text(
                "INSERT INTO chat_sessions (session_id, seq, title, facts) "
                "VALUES (:sid, :seq, :title, JSON_OBJECT())"
            ),
            {"sid": session_id, "seq": next_seq, "title": f"Chat {next_seq}"},
        )
    LOGGER.info("New chat session %s (Chat %d)", session_id, next_seq)
    _write_index()
    return get_session(session_id) or {"session_id": session_id, "seq": next_seq}


def rename_session(session_id: str, title: str) -> dict[str, Any] | None:
    """Set a session's title. Returns the updated session, or None if unknown."""
    clean = title.strip()[:_TITLE_MAX] or "Untitled"
    with db_session() as session:
        session.execute(
            text("UPDATE chat_sessions SET title = :title WHERE session_id = :sid"),
            {"title": clean, "sid": session_id},
        )
    _write_index()
    return get_session(session_id)


def save_facts(session_id: str, facts: dict[str, str]) -> None:
    """Persist the remembered facts (e.g. the user's name) for a session."""
    with db_session() as session:
        session.execute(
            text("UPDATE chat_sessions SET facts = :facts WHERE session_id = :sid"),
            {"facts": json.dumps(facts), "sid": session_id},
        )


def append_message(
    session_id: str,
    role: str,
    content: str,
    *,
    kind: str = "text",
    run_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Append one turn to a session and bump its updated_at."""
    with db_session() as session:
        next_seq = int(
            session.execute(
                text("SELECT COALESCE(MAX(seq), 0) + 1 FROM chat_messages WHERE session_id = :sid"),
                {"sid": session_id},
            ).scalar()
        )
        session.execute(
            text(
                "INSERT INTO chat_messages (session_id, seq, role, kind, content, run_id, payload) "
                "VALUES (:sid, :seq, :role, :kind, :content, :run_id, :payload)"
            ),
            {
                "sid": session_id,
                "seq": next_seq,
                "role": role,
                "kind": kind,
                "content": content,
                "run_id": run_id,
                "payload": json.dumps(payload) if payload is not None else None,
            },
        )
        session.execute(
            text(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP(3) WHERE session_id = :sid"
            ),
            {"sid": session_id},
        )


def get_messages(session_id: str) -> list[dict[str, Any]]:
    """Return a session's turns in order, with the stored payload decoded."""
    with db_session() as session:
        rows = (
            session.execute(
                text(
                    "SELECT seq, role, kind, content, run_id, payload, created_at "
                    "FROM chat_messages WHERE session_id = :sid ORDER BY seq"
                ),
                {"sid": session_id},
            )
            .mappings()
            .all()
        )
    messages: list[dict[str, Any]] = []
    for row in rows:
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        messages.append(
            {
                "seq": row["seq"],
                "role": row["role"],
                "kind": row["kind"],
                "content": row["content"],
                "run_id": row["run_id"],
                "payload": payload,
                "created_at": str(row["created_at"]),
            }
        )
    return messages


def _write_index() -> None:
    """Mirror the session list to the local JSON file (best effort)."""
    try:
        rows = [
            {"session_id": s["session_id"], "seq": s["seq"], "title": s["title"]}
            for s in list_sessions()
        ]
        INDEX_FILE.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Could not write %s: %s", INDEX_FILE, exc)
