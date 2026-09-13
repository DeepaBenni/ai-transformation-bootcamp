"""The three contradictory pairs must resolve identically on every run."""

from __future__ import annotations

from langchain_core.documents import Document

from app.kb.conflict import resolve_conflicts

# (a_id, a_reviewed, a_status, b_id, b_reviewed, b_status, expected_drop)
PAIRS = [
    ("KB-003", "2023-06-02", "outdated", "RB-10", "2026-08-04", "contradictory", "KB-003"),
    ("KB-006", "2026-08-11", "contradictory", "KB-025", "2023-02-09", "outdated", "KB-025"),
    ("KB-012", "2026-07-30", "contradictory", "RB-06", "2022-11-18", "outdated", "RB-06"),
]


def _doc(doc_id: str, status: str, reviewed: str, other: str, doc_type: str = "article"):
    return Document(
        page_content=f"body of {doc_id}",
        metadata={
            "id": doc_id,
            "status": status,
            "last_reviewed": reviewed,
            "conflicts_with": other,
            "type": doc_type,
            "title": doc_id,
        },
    )


def test_each_pair_drops_the_outdated_side() -> None:
    for a_id, a_rev, a_status, b_id, b_rev, b_status, expected_drop in PAIRS:
        outcome = resolve_conflicts(
            [
                _doc(a_id, a_status, a_rev, b_id),
                _doc(b_id, b_status, b_rev, a_id),
            ]
        )
        assert outcome.superseded == [expected_drop]
        assert not outcome.has_unresolved
        assert [d.metadata["id"] for d in outcome.documents] != [expected_drop]


def test_two_live_documents_are_not_silently_chosen_between() -> None:
    """Both live and both the same type: refuse to choose, so the supervisor escalates."""
    outcome = resolve_conflicts(
        [
            _doc("KB-100", "current", "2026-08-01", "KB-101"),
            _doc("KB-101", "current", "2026-08-02", "KB-100"),
        ]
    )
    assert outcome.has_unresolved
    assert len(outcome.documents) == 2
