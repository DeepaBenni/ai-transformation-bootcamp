"""Read tools return structured data, and fail structurally rather than raising."""

from __future__ import annotations

from app.tools import read_tools


def test_get_user_known(run) -> None:
    result = read_tools.get_user.invoke({"user_id": "U-1042"})
    assert result["employment_status"] == "active"
    assert result["department"] == "Finance"


def test_get_user_unknown_returns_error_not_exception(run) -> None:
    result = read_tools.get_user.invoke({"user_id": "U-0000"})
    assert "error" in result


def test_leaver_is_visible(run) -> None:
    assert read_tools.get_user.invoke({"user_id": "U-2087"})["employment_status"] == "leaver"
    assert read_tools.check_account_lockout.invoke({"user_id": "U-2087"})["state"] == "disabled"


def test_disk_usage_is_sorted_fullest_first(run) -> None:
    mounts = read_tools.get_disk_usage.invoke({"host": "file-prd-01"})["mounts"]
    assert mounts[0]["used_pct"] >= mounts[-1]["used_pct"]


def test_search_kb_finds_the_lockout_runbook(run) -> None:
    result = read_tools.search_kb.invoke({"query": "account locked after failed logins"})
    assert not result["below_floor"]
    assert "RB-01" in [hit["id"] for hit in result["hits"]]
