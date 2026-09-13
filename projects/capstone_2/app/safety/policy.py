"""Deterministic policy checks.

These run before any approval is even requested. They encode rules that are not
judgement calls, so a model is never asked to apply them and can never be talked
out of them.

Boundary: pure functions over facts already gathered. No I/O, no LLM.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# 08:00-18:59 local. range() stops before 19, which is what we want.
BUSINESS_HOURS = range(8, 19)


def is_business_hours(now: datetime | None = None) -> bool:
    """Return True if the current local hour falls inside business hours."""
    return (now or datetime.now()).hour in BUSINESS_HOURS


def preflight(action: str, args: dict[str, Any], facts: dict[str, Any]) -> str | None:
    """Return a refusal reason if policy forbids this action outright, else None.

    Args:
        action: The proposed tool name.
        args: Its arguments.
        facts: Observations already gathered, keyed by tool name.
    """
    account = facts.get("check_account_lockout", {}) or {}
    user = facts.get("get_user", {}) or {}

    # A leaver's disabled account is HR's to touch, never the Service Desk's.
    if account.get("state") == "disabled" or user.get("employment_status") == "leaver":
        return (
            "Account is disabled and the employee is a leaver. The Service Desk must not "
            "re-enable a leaver account; this belongs to HR offboarding. See RB-01 step 2."
        )

    if action == "restart_service" and is_business_hours():
        return (
            f"restart_service on {args.get('service')} is outside its permitted restart window "
            "during business hours. A change record is required."
        )

    if action == "grant_group_access" and not args.get("group"):
        return "grant_group_access requires a specific group name; none was identified."

    return None
