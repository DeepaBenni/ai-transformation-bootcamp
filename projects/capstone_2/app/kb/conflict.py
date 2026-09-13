"""Resolve contradictions between retrieved KB documents.

Three of the KB's document pairs contradict each other and declare it in their
`conflicts_with` frontmatter. Choosing between them is metadata arithmetic, not
judgement, so it is done here in pure code - deterministically, identically on
every run, and visibly in the trace.

Boundary: no LLM call, no I/O. Given the same hit list this returns the same
outcome forever, which is exactly why it is not an agent's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass(frozen=True)
class ConflictOutcome:
    """The result of resolving a hit list."""

    documents: list[Document]
    superseded: list[str] = field(default_factory=list)
    unresolved: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_unresolved(self) -> bool:
        """Return True when at least one contradiction could not be settled by rule."""
        return bool(self.unresolved)


def _doc_id(document: Document) -> str:
    return str(document.metadata.get("id", ""))


def resolve_conflicts(documents: list[Document], *, intent: str = "procedure") -> ConflictOutcome:
    """Drop superseded documents from a retrieval hit list.

    Args:
        documents: Retrieved documents, best match first.
        intent: "procedure" when the ticket asks how to do something, "diagnosis"
            when it asks what is wrong. Only used by rule 2.

    Returns:
        The surviving documents plus a record of what was dropped and why.

    Rules, applied in order to each declared conflict pair present in the list:
        1. If exactly one side is `status == "outdated"`, drop that side.
        2. If both are live, a runbook outranks an article for procedure, and an
           article outranks a runbook for diagnosis.
        3. Otherwise keep both and mark the pair unresolved - an unsettled
           contradiction is an L2 decision, not a guess.
    """
    by_id = {_doc_id(d): d for d in documents}
    superseded: list[str] = []
    unresolved: list[tuple[str, str]] = []
    notes: list[str] = []
    seen_pairs: set[frozenset[str]] = set()

    for document in documents:
        this_id = _doc_id(document)
        other_id = str(document.metadata.get("conflicts_with", "")).strip()
        if not other_id or other_id not in by_id:
            continue

        # A conflict pair is symmetric; process each pair only once.
        pair = frozenset({this_id, other_id})
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        other = by_id[other_id]

        # Rule 1 - staleness settles it.
        this_stale = document.metadata.get("status") == "outdated"
        other_stale = other.metadata.get("status") == "outdated"
        if this_stale != other_stale:
            loser = this_id if this_stale else other_id
            winner = other_id if this_stale else this_id
            superseded.append(loser)
            notes.append(f"{loser} superseded by {winner}: marked outdated")
            continue

        # Rule 2 - document type settles it when both are live.
        preferred_type = "runbook" if intent == "procedure" else "article"
        this_pref = document.metadata.get("type") == preferred_type
        other_pref = other.metadata.get("type") == preferred_type
        if this_pref != other_pref:
            loser = other_id if this_pref else this_id
            winner = this_id if this_pref else other_id
            superseded.append(loser)
            notes.append(f"{loser} deprioritised for {intent}: {winner} is a {preferred_type}")
            continue

        # Rule 3 - refuse to choose.
        unresolved.append((this_id, other_id))
        notes.append(f"{this_id} and {other_id} conflict and neither rule settles it")

    kept = [d for d in documents if _doc_id(d) not in superseded]
    return ConflictOutcome(kept, superseded, unresolved, notes)
