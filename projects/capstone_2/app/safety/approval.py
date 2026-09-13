"""The approval ledger: the only module that can authorise a state change.

An approval is bound to a run, an action and its exact arguments. Approving
unlock_account for U-1042 therefore does not authorise it for U-2087, and an
approval from one run cannot be replayed in another.

Boundary: grant() is called from exactly one place - the handler that processes a
real person's answer. Nothing an LLM emits ever reaches this module.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from app.db.models import Approval
from app.db.session import db_session
from app.logging_conf import get_logger

LOGGER = get_logger(__name__)


def approval_token(run_id: str, action: str, args: dict[str, Any]) -> str:
    """Derive the token that authorises exactly this call in exactly this run.

    The token is a SHA-256 over the sorted (run_id, action, args) triple, so any
    change to the user id, the group, the host - anything - produces a different
    token and the earlier approval no longer matches.
    """
    payload = json.dumps(
        {"run_id": run_id, "action": action, "args": args},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def grant(
    run_id: str,
    action: str,
    args: dict[str, Any],
    *,
    approver: str,
    reason: str = "",
    approved: bool = True,
) -> str:
    """Record a human decision and return its token.

    This function is called from exactly one place: the handler that processes a
    real person's answer. Nothing an LLM emits reaches it.
    """
    token = approval_token(run_id, action, args)
    with db_session() as session:
        existing = session.scalar(select(Approval).where(Approval.token_hash == token))
        if existing is None:
            session.add(
                Approval(
                    run_id=run_id,
                    token_hash=token,
                    action=action,
                    args_json=args,
                    decision="approved" if approved else "denied",
                    approver=approver,
                    reason=reason,
                    decided_at=datetime.now(UTC).replace(tzinfo=None),
                )
            )
    LOGGER.info(
        "Approval %s: %s %s by %s",
        "granted" if approved else "denied",
        action,
        args,
        approver,
    )
    return token


def is_granted(run_id: str, action: str, args: dict[str, Any]) -> bool:
    """Return True only if a human approved this exact call in this exact run."""
    token = approval_token(run_id, action, args)
    with db_session() as session:
        row = session.scalar(select(Approval).where(Approval.token_hash == token))
    return row is not None and row.decision == "approved"


def approver_of(run_id: str, action: str, args: dict[str, Any]) -> str:
    """Return the approver's name, or 'unknown' if there is no approval."""
    token = approval_token(run_id, action, args)
    with db_session() as session:
        row = session.scalar(select(Approval).where(Approval.token_hash == token))
    return row.approver if row is not None else "unknown"
