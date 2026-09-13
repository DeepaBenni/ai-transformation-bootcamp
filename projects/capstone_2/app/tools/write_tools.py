"""State-changing tools.

Each one re-checks the approval ledger before touching anything. This is the
second of the three locks: even if an agent is fully captured by an injected
instruction and calls a writer directly, the writer refuses.

Boundary: every public tool here starts with an is_granted() check and returns a
"REFUSED: ..." string when it fails. No code path writes before that check.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from app.audit.context import current_run
from app.db.models import Account, ActionLogEntry, GroupMembership, Service
from app.db.session import db_session
from app.logging_conf import get_logger
from app.safety.approval import approver_of, is_granted
from app.tools.schemas import EscalateArgs, GroupArgs, RestartArgs, UserIdArgs

LOGGER = get_logger(__name__)

# Length of the random component of a temporary password, in URL-safe bytes.
_TEMP_PASSWORD_BYTES: int = 9


def _refuse(action: str, args: dict[str, Any]) -> str:
    """Record a blocked write and return the refusal the agent will see."""
    run = current_run()
    run.tracer.log(
        "guard_refusal",
        tool=action,
        args=args,
        observation={"refused": True, "reason": "no approval on record"},
    )
    LOGGER.warning("BLOCKED %s %s in run %s - no approval", action, args, run.run_id)
    return (
        f"REFUSED: no human approval is on record for {action} with arguments {args}. "
        "Request approval first. Nothing has been changed."
    )


def _record(action: str, args: dict[str, Any], result: str) -> None:
    """Write the executed change to action_log and the trace.

    action_log is the narrow record of things that actually changed the estate;
    joining it to approvals is how a grader proves nothing changed unapproved.
    """
    run = current_run()
    with db_session() as session:
        session.add(
            ActionLogEntry(
                run_id=run.run_id,
                action=action,
                args_json=args,
                approver=approver_of(run.run_id, action, args),
                result=result,
                executed_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )
    run.tracer.log("action_executed", tool=action, args=args, observation={"result": result})


@tool(args_schema=UserIdArgs)
def unlock_account(user_id: str) -> str:
    """Unlock a locked account. Refuses unless a human has approved this exact call."""
    args = {"user_id": user_id}
    run = current_run()
    if not is_granted(run.run_id, "unlock_account", args):
        return _refuse("unlock_account", args)

    with db_session() as session:
        account = session.get(Account, user_id)
        if account is None:
            return f"ERROR: no account for {user_id}."
        if account.state == "disabled":
            return f"ERROR: {user_id} is disabled, not locked. Refusing to re-enable a leaver."
        account.state = "active"
        account.failed_logins_24h = 0

    result = f"{user_id} is now active; failed login counter reset."
    _record("unlock_account", args, result)
    return f"DONE: {result}"


@tool(args_schema=UserIdArgs)
def reset_password(user_id: str) -> str:
    """Issue a temporary password with must-change-at-next-logon set.

    Refuses unless a human has approved this exact call.
    """
    args = {"user_id": user_id}
    run = current_run()
    if not is_granted(run.run_id, "reset_password", args):
        return _refuse("reset_password", args)

    temporary = f"Tmp-{secrets.token_urlsafe(_TEMP_PASSWORD_BYTES)}"
    with db_session() as session:
        account = session.get(Account, user_id)
        if account is None:
            return f"ERROR: no account for {user_id}."
        account.password_set_at = datetime.now(UTC).replace(tzinfo=None)

    result = f"temporary password issued for {user_id}; must change at next logon."
    _record("reset_password", args, result)
    return f"DONE: {result} Value: {temporary}"


@tool(args_schema=RestartArgs)
def restart_service(host: str, service: str) -> str:
    """Restart a service on a host. Refuses unless a human has approved this exact call."""
    args = {"host": host, "service": service}
    run = current_run()
    if not is_granted(run.run_id, "restart_service", args):
        return _refuse("restart_service", args)

    with db_session() as session:
        row = session.get(Service, service)
        if row is None or row.host != host:
            return f"ERROR: {service} does not run on {host}."
        row.status = "restarting"
        row.status_since = datetime.now(UTC).replace(tzinfo=None)

    result = f"{service} on {host} is restarting."
    _record("restart_service", args, result)
    return f"DONE: {result}"


@tool(args_schema=GroupArgs)
def grant_group_access(user_id: str, group: str) -> str:
    """Add a user to a directory group. Refuses unless a human has approved this exact call."""
    args = {"user_id": user_id, "group": group}
    run = current_run()
    if not is_granted(run.run_id, "grant_group_access", args):
        return _refuse("grant_group_access", args)

    with db_session() as session:
        session.add(
            GroupMembership(
                user_id=user_id,
                group_name=group,
                granted_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )

    result = f"{user_id} added to {group}."
    _record("grant_group_access", args, result)
    return f"DONE: {result}"


@tool(args_schema=EscalateArgs)
def escalate_to_l2(ticket_id: str, summary: str) -> str:
    """Hand the ticket to L2 with a structured summary.

    This writes to the L2 queue and pages a human team, so it is approval-gated
    exactly like any other state change.
    """
    args = {"ticket_id": ticket_id, "summary": summary}
    run = current_run()
    if not is_granted(run.run_id, "escalate_to_l2", args):
        return _refuse("escalate_to_l2", args)

    result = f"{ticket_id} placed on the L2 queue with a {len(summary)}-character handover."
    _record("escalate_to_l2", args, result)
    return f"DONE: {result}"
