"""Read-only tools.

Nothing in this module may write to any store. That is not a convention - the
Diagnostic agent is handed this module's tools and no others, so a successful
injection against it still has no write path to reach.

Pattern per tool: schema -> plain function (`_name`) -> `@tool` wrapper. The
wrapper calls `_guarded`, which adds timeout handling and tracing so no
individual tool has to remember to.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from decimal import Decimal
from typing import Any

from langchain_core.tools import tool
from sqlalchemy import select

from app.audit.context import current_run
from app.config import get_settings
from app.db.models import Account, ConfigItem, DiskUsage, Service, Ticket, User
from app.db.session import db_session
from app.kb.retriever import search
from app.logging_conf import get_logger
from app.tools.schemas import (
    CiArgs,
    HostArgs,
    QueryArgs,
    ServiceArgs,
    TextArgs,
    UserIdArgs,
)

LOGGER = get_logger(__name__)

# Small pool so a slow query does not block the event loop; sized for local stubs.
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")

_SHORT_WORD_LEN: int = 3
_MAX_SIMILAR_TICKETS: int = 3
_RECENT_TICKET_LIMIT: int = 5


def _plain(value: Any) -> Any:
    """Convert Decimal and datetime into JSON-friendly primitives."""
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ", timespec="seconds")
    return value


def _guarded(name: str, fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Run a tool body with a timeout, and record the call and its result.

    A timeout or an exception becomes a structured observation rather than a
    raised error, so the agent can reason about the gap instead of crashing.
    The worker thread is not killed - a hung query still occupies a slot - which
    is acceptable for deterministic local stubs and is noted in the governance
    document as a real-world limitation.
    """
    run = current_run()
    run.tracer.log("tool_call", tool=name, args=kwargs)
    started = time.perf_counter()
    try:
        result = _POOL.submit(fn, **kwargs).result(timeout=get_settings().tool_timeout_s)
    except FutureTimeout:
        result = {"error": "timeout", "tool": name, "retryable": True}
    except Exception as exc:  # deliberate boundary: a tool fault must not crash the run
        result = {"error": type(exc).__name__, "detail": str(exc), "retryable": False}
        LOGGER.exception("Tool %s failed", name)

    latency_ms = int((time.perf_counter() - started) * 1000)
    run.tracer.log("observation", tool=name, observation=result, latency_ms=latency_ms)
    return result


# ------------------------------------------------------------ plain bodies --
def _get_user(user_id: str) -> dict[str, Any]:
    with db_session() as session:
        row = session.get(User, user_id)
        if row is None:
            return {"error": f"no such user {user_id}"}
        return {
            "user_id": row.user_id,
            "full_name": row.full_name,
            "email": row.email,
            "department": row.department,
            "manager": row.manager,
            "employment_status": row.employment_status,
        }


def _check_account_lockout(user_id: str) -> dict[str, Any]:
    with db_session() as session:
        row = session.get(Account, user_id)
        if row is None:
            return {"error": f"no account for {user_id}"}
        return {
            "user_id": row.user_id,
            "state": row.state,
            "failed_logins_24h": row.failed_logins_24h,
            "disabled_reason": row.disabled_reason,
            "last_login_at": _plain(row.last_login_at),
        }


def _get_ticket_history(user_id: str) -> dict[str, Any]:
    with db_session() as session:
        rows = session.scalars(
            select(Ticket)
            .where(Ticket.user_id == user_id)
            .order_by(Ticket.created_at.desc())
            .limit(_RECENT_TICKET_LIMIT)
        ).all()
    return {
        "user_id": user_id,
        "count": len(rows),
        "tickets": [
            {
                "ticket_id": row.ticket_id,
                "category": row.category,
                "status": row.status,
                "body": row.body,
                "resolution": row.resolution,
                "created_at": _plain(row.created_at),
            }
            for row in rows
        ],
    }


def _get_ci_health(ci_name: str) -> dict[str, Any]:
    with db_session() as session:
        row = session.get(ConfigItem, ci_name)
        if row is None:
            return {"error": f"no configuration item named {ci_name}"}
        return {
            "ci_name": row.ci_name,
            "ci_type": row.ci_type,
            "status": row.status,
            "cpu_pct": _plain(row.cpu_pct),
            "mem_pct": _plain(row.mem_pct),
            "last_checkin_at": _plain(row.last_checkin_at),
            "owner_team": row.owner_team,
        }


def _check_service_status(service: str) -> dict[str, Any]:
    with db_session() as session:
        row = session.get(Service, service)
        if row is None:
            return {"error": f"no service named {service}"}
        return {
            "service": row.service,
            "host": row.host,
            "status": row.status,
            "depends_on": row.depends_on,
            "status_since": _plain(row.status_since),
            "restart_window": row.restart_window,
        }


def _get_disk_usage(host: str) -> dict[str, Any]:
    with db_session() as session:
        rows = session.scalars(
            select(DiskUsage).where(DiskUsage.host == host).order_by(DiskUsage.used_pct.desc())
        ).all()
    if not rows:
        return {"error": f"no disk data for host {host}"}
    return {
        "host": host,
        "mounts": [
            {"mount": row.mount, "used_pct": _plain(row.used_pct), "free_gb": _plain(row.free_gb)}
            for row in rows
        ],
    }


def _find_similar_tickets(text: str) -> dict[str, Any]:
    """Match against resolved tickets with a simple keyword overlap score.

    Deliberately not a second vector index: 16 tickets do not justify one, and a
    transparent score is easier to defend in a trace than an opaque distance.
    """
    words = {word for word in text.lower().split() if len(word) > _SHORT_WORD_LEN}
    with db_session() as session:
        rows = session.scalars(select(Ticket).where(Ticket.status == "resolved")).all()

    scored: list[tuple[int, Ticket]] = []
    for row in rows:
        haystack = f"{row.body} {row.resolution or ''}".lower()
        overlap = sum(1 for word in words if word in haystack)
        if overlap:
            scored.append((overlap, row))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return {
        "matches": [
            {
                "ticket_id": row.ticket_id,
                "body": row.body,
                "resolution": row.resolution,
                "overlap": overlap,
            }
            for overlap, row in scored[:_MAX_SIMILAR_TICKETS]
        ]
    }


# -------------------------------------------------------------- the tools --
@tool(args_schema=UserIdArgs)
def get_user(user_id: str) -> dict[str, Any]:
    """Look up an employee: name, department, manager and employment status.

    Call this first for any ticket that names a user. An employment_status of
    'leaver' means no account change may be proposed for them.
    """
    return _guarded("get_user", _get_user, user_id=user_id)


@tool(args_schema=UserIdArgs)
def check_account_lockout(user_id: str) -> dict[str, Any]:
    """Check whether an account is active, locked or disabled, and why.

    'locked' follows repeated failed logins and is normally remediable.
    'disabled' means a leaver or ended contract and must never be re-enabled here.
    """
    return _guarded("check_account_lockout", _check_account_lockout, user_id=user_id)


@tool(args_schema=UserIdArgs)
def get_ticket_history(user_id: str) -> dict[str, Any]:
    """Return this user's five most recent tickets with their resolutions.

    Use it to spot a repeat problem: the same fault three times means the earlier
    fix did not hold, which changes the correct action.
    """
    return _guarded("get_ticket_history", _get_ticket_history, user_id=user_id)


@tool(args_schema=CiArgs)
def get_ci_health(ci_name: str) -> dict[str, Any]:
    """Return CPU, memory, status and last check-in for a configuration item."""
    return _guarded("get_ci_health", _get_ci_health, ci_name=ci_name)


@tool(args_schema=ServiceArgs)
def check_service_status(service: str) -> dict[str, Any]:
    """Return a service's status, its host, what it depends on, and since when.

    A degraded dependency often explains a user-reported fault that looks like an
    account problem.
    """
    return _guarded("check_service_status", _check_service_status, service=service)


@tool(args_schema=HostArgs)
def get_disk_usage(host: str) -> dict[str, Any]:
    """Return per-mount disk usage for a host, fullest mount first."""
    return _guarded("get_disk_usage", _get_disk_usage, host=host)


@tool(args_schema=QueryArgs)
def search_kb(query: str) -> dict[str, Any]:
    """Search the knowledge base and runbooks for guidance on a problem.

    Returns hits with their id, status and last_reviewed date. Contradictory
    documents are already resolved: `superseded` lists what was dropped and
    `unresolved` lists any contradiction that could not be settled. If
    `below_floor` is true, the knowledge base does not cover this question and
    you must say so rather than improvising.
    """
    return _guarded("search_kb", search, query=query)


@tool(args_schema=TextArgs)
def find_similar_tickets(text: str) -> dict[str, Any]:
    """Find previously resolved tickets that resemble this one, with how they were fixed."""
    return _guarded("find_similar_tickets", _find_similar_tickets, text=text)
