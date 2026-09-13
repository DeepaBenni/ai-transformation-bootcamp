"""Knowledge: grounded guidance with a citation on every claim.

Two mechanical guarantees, both enforced after the model has spoken:
  1. A claim whose source_id is not in the retrieved context is dropped.
  2. If retrieval scored below the floor, the model is never called at all -
     the refusal is cheaper and more reliable as code.
"""

from __future__ import annotations

from app.agents.llm import ask_structured
from app.agents.prompts import KNOWLEDGE
from app.agents.state import Citation, KnowledgeResult, OpsMateState
from app.audit.context import current_run
from app.kb.retriever import search
from app.logging_conf import get_logger

LOGGER = get_logger(__name__)

# Ticket text is truncated before it becomes a retrieval query - the tail of a
# long ticket is rarely what makes it findable, and a shorter query embeds faster.
_QUERY_CHARS: int = 300
# Below this, the ticket text is a bare follow-up ("it is U-9310") that needs the
# previous user turn to be retrievable. Above it, the ticket stands on its own and
# earlier conversation (which may be small talk) would only dilute the query.
_FRAGMENT_CHARS: int = 30
_CONTEXT_CHARS_PER_HIT: int = 900


def _retrieval_query(state: OpsMateState) -> str:
    """Build the KB query.

    Prefer Intake's `problem_summary` - a clean sentence with any manipulation
    stripped. Fall back to the ticket text, plus the prior user turn when the
    ticket is a bare follow-up ("it is U-9310").
    """
    intake = state["intake"]
    if intake and intake.problem_summary.strip():
        return intake.problem_summary.strip()[:_QUERY_CHARS]

    body = state["ticket"].body.strip()
    if len(body) >= _FRAGMENT_CHARS:
        return body[:_QUERY_CHARS]

    prior = [
        t.get("content", "").strip()
        for t in state["history"]
        if t.get("role") == "user" and t.get("content", "").strip() != body
    ]
    parts = [prior[-1]] if prior else []
    parts.append(body)
    return " ".join(parts)[:_QUERY_CHARS]


def knowledge_node(state: OpsMateState) -> dict[str, object]:
    """Retrieve guidance, ground it, and cite it - or refuse."""
    run = current_run()
    intake = state["intake"]
    intent = "procedure" if intake and intake.intent in {"access", "storage"} else "diagnosis"

    query = _retrieval_query(state)
    hits = search(query, intent=intent)
    run.tracer.log(
        "retrieval",
        agent="knowledge",
        observation={
            "ids": [h["id"] for h in hits["hits"]],
            "top_score": hits["top_score"],
            "superseded": hits["superseded"],
            "unresolved": hits["unresolved"],
        },
    )

    if hits["below_floor"] or not hits["hits"]:
        LOGGER.info(
            "Below relevance floor (%.3f); refusing without a model call", hits["top_score"]
        )
        return {
            "knowledge": KnowledgeResult(not_covered=True, superseded=hits["superseded"]),
            "citations": [],
        }

    context = "\n\n".join(
        f"[{h['id']}] ({h['type']}, status={h['status']}, reviewed={h['last_reviewed']})\n"
        f"{h['text'][:_CONTEXT_CHARS_PER_HIT]}"
        for h in hits["hits"]
    )
    result = ask_structured(
        KnowledgeResult,
        [
            {"role": "system", "content": KNOWLEDGE},
            {
                "role": "user",
                "content": (
                    f"PROBLEM:\n{query}\n\nCONTEXT:\n{context}\n\n"
                    "Answer with claims, each carrying a source_id from CONTEXT."
                ),
            },
        ],
        agent="knowledge",
    )

    # Mechanical grounding check - this is what makes R2 a guarantee, not a hope.
    valid_ids = {h["id"] for h in hits["hits"]}
    kept = [c for c in result.claims if c.source_id in valid_ids]
    dropped = [c.source_id for c in result.claims if c.source_id not in valid_ids]
    if dropped:
        run.tracer.log(
            "ungrounded_claim_dropped", agent="knowledge", observation={"source_ids": dropped}
        )
        LOGGER.warning("Dropped %d ungrounded claims: %s", len(dropped), dropped)

    result.claims = kept
    result.not_covered = result.not_covered or not kept
    result.superseded = hits["superseded"]
    result.conflict_unresolved = bool(hits["unresolved"])

    cited = {c.source_id for c in kept}
    citations = [
        Citation(
            source_id=h["id"],
            title=h["title"],
            status=h["status"],
            last_reviewed=h["last_reviewed"],
        )
        for h in hits["hits"]
        if h["id"] in cited
    ]

    run.tracer.log(
        "knowledge_done",
        agent="knowledge",
        observation={
            "claims": len(kept),
            "not_covered": result.not_covered,
            "cited": sorted(cited),
        },
    )
    return {"knowledge": result, "citations": citations}
