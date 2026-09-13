"""Build the L2 handover.

Assembled from state by code. The model writes only two of the five R5 fields,
and only the prose ones - it never supplies evidence or actions, because those
are facts the graph already holds.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from app.agents.llm import ask_structured
from app.agents.prompts import HANDOVER
from app.agents.state import (
    ActionProposal,
    EvidenceItem,
    Handover,
    OpsMateState,
    RejectedAction,
)
from app.audit.context import current_run

_OBSERVATION_CHARS: int = 400
_RECORD_CHARS: int = 5000


class _Prose(BaseModel):
    """The two fields the model is allowed to write."""

    symptom: str = Field(description="What the user experiences, concretely, 1-2 sentences.")
    next_question: str = Field(description="The single specific thing L2 must determine.")


def build_handover(state: OpsMateState) -> Handover:
    """Assemble the five R5 fields, deterministically where possible."""
    evidence = [
        EvidenceItem(
            tool=tool, observation=json.dumps(observation, default=str)[:_OBSERVATION_CHARS]
        )
        for tool, observation in state["facts"].items()
    ]

    approval = state["approval"]
    actions_taken = (
        [
            f"{state['proposal'].action}({state['proposal'].args}) approved by "
            f"{approval.approver}: {approval.result}"
        ]
        if approval and approval.approved and state["proposal"]
        else ["None. No state-changing action was executed."]
    )

    proposal: ActionProposal | None = state["proposal"]
    rejected = list(proposal.alternatives_rejected) if proposal else []
    if state["policy_refusal"]:
        rejected.append(
            RejectedAction(
                action=proposal.action if proposal else "unknown",
                reason=state["policy_refusal"],
            )
        )
    if approval and not approval.approved and proposal and proposal.action != "none":
        rejected.append(
            RejectedAction(
                action=proposal.action,
                reason=(
                    f"declined by {approval.approver}: " f"{approval.reason or 'no reason given'}"
                ),
            )
        )

    record = {
        "ticket_id": state["ticket"].ticket_id,
        "intent": state["intake"].intent if state["intake"] else "unknown",
        "evidence": [e.model_dump() for e in evidence],
        "citations": [c.source_id for c in state["citations"]],
        "knowledge_gap": state["knowledge"].not_covered if state["knowledge"] else None,
        "conflict_unresolved": (
            state["knowledge"].conflict_unresolved if state["knowledge"] else False
        ),
        "stop_reason": state["stop_reason"],
        "actions_taken": actions_taken,
        "actions_rejected": [r.model_dump() for r in rejected],
    }

    prose = ask_structured(
        _Prose,
        [
            {"role": "system", "content": HANDOVER},
            {"role": "user", "content": json.dumps(record, default=str)[:_RECORD_CHARS]},
        ],
        agent="handover",
    )

    handover = Handover(
        symptom=prose.symptom,
        evidence=evidence,
        actions_taken=actions_taken,
        actions_rejected=rejected,
        next_question=prose.next_question,
    )
    current_run().tracer.log(
        "handover_built",
        agent="handover",
        observation={"evidence_items": len(evidence), "rejected": len(rejected)},
    )
    return handover


def render_handover(handover: Handover, ticket_id: str) -> str:
    """Render the handover as the text an engineer will actually read."""
    lines = [
        f"L2 HANDOVER - {ticket_id}",
        "",
        f"SYMPTOM: {handover.symptom}",
        "",
        "EVIDENCE GATHERED:",
    ]
    lines += [f"  - {e.tool}: {e.observation}" for e in handover.evidence] or ["  - none"]
    lines += ["", "ACTIONS TAKEN:"]
    lines += [f"  - {a}" for a in handover.actions_taken]
    lines += ["", "ACTIONS REJECTED:"]
    lines += [f"  - {r.action}: {r.reason}" for r in handover.actions_rejected] or ["  - none"]
    lines += ["", f"NEXT QUESTION FOR L2: {handover.next_question}"]
    return "\n".join(lines)
