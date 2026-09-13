"""Every write tool must refuse when no human approval is on record.

These five parametrised cases are the R3 evidence: they prove the safety gate
holds even when a tool is invoked directly, with no agent and no supervisor.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.db.models import Account
from app.db.session import db_session
from app.safety.approval import grant
from app.tools import write_tools

GATED_CALLS = [
    (write_tools.unlock_account, {"user_id": "U-1042"}),
    (write_tools.reset_password, {"user_id": "U-1042"}),
    (write_tools.restart_service, {"host": "app-prd-02", "service": "print-spooler"}),
    (write_tools.grant_group_access, {"user_id": "U-3775", "group": "vpn-users"}),
    (write_tools.escalate_to_l2, {"ticket_id": "INC-88120", "summary": "test"}),
]


@pytest.mark.parametrize(("tool", "args"), GATED_CALLS, ids=lambda v: getattr(v, "name", ""))
def test_write_refuses_without_approval(tool, args, run) -> None:
    result = tool.invoke(args)
    assert result.startswith("REFUSED"), f"{tool.name} executed without approval"


def test_account_state_unchanged_after_refusal(run) -> None:
    with db_session() as session:
        before = session.scalar(select(Account.state).where(Account.user_id == "U-1042"))

    write_tools.unlock_account.invoke({"user_id": "U-1042"})

    with db_session() as session:
        after = session.scalar(select(Account.state).where(Account.user_id == "U-1042"))
    assert before == after == "locked"


def test_approval_does_not_transfer_between_users(run) -> None:
    """An approval for one user must not authorise the same action on another."""
    grant(run.run_id, "unlock_account", {"user_id": "U-1042"}, approver="tester")
    result = write_tools.unlock_account.invoke({"user_id": "U-5203"})
    assert result.startswith("REFUSED")
