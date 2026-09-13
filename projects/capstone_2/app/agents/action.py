"""Action: propose a remediation, then execute only what a human approved.

Split into two nodes on purpose. interrupt() re-runs its node from the top on
resume, so anything expensive or non-deterministic must happen before the node
that interrupts: the model call lives in `propose_node`, and `interrupt()` is the
first statement of `approval_node`.
"""

from __future__ import annotations

import json

from langgraph.types import interrupt

from app.agents.llm import ask_structured
from app.agents.prompts import PROPOSE
from app.agents.state import ActionProposal, ApprovalRecord, OpsMateState
from app.audit.context import current_run
from app.logging_conf import get_logger
from app.safety.approval import grant
from app.safety.policy import preflight
from app.safety.sanitize import as_untrusted
from app.tools.registry import BY_NAME

LOGGER = get_logger(__name__)

# The record handed to the model is capped so a huge fact blob cannot crowd out
# the system prompt.
_RECORD_CHARS: int = 4000

# Which entity fills which argument, per action. Used to complete an `args` dict
# the model left partly blank - the values come from identifiers Intake already
# extracted, so this is assembly, not a second guess at what the ticket said.
_ACTION_ARG_SOURCES: dict[str, dict[str, str]] = {
    "unlock_account": {"user_id": "user_id"},
    "reset_password": {"user_id": "user_id"},
    "restart_service": {"host": "host", "service": "service"},
    "grant_group_access": {"user_id": "user_id", "group": "group"},
}


def _complete_args(action: str, args: dict[str, object], entities: dict[str, object]) -> dict:
    """Fill any argument the model omitted from the entities Intake extracted."""
    filled = dict(args)
    for arg_key, entity_key in _ACTION_ARG_SOURCES.get(action, {}).items():
        if not filled.get(arg_key) and entities.get(entity_key):
            filled[arg_key] = entities[entity_key]
    return filled


def propose_node(state: OpsMateState) -> dict[str, object]:
    """Ask the model for one remediation, then run the deterministic policy check."""
    run = current_run()

    claims = state["knowledge"].claims if state["knowledge"] else []
    entities = state["intake"].entities.model_dump() if state["intake"] else {}
    payload = {
        "evidence": state["facts"],
        "guidance": [{"text": c.text, "source_id": c.source_id} for c in claims],
        "entities": entities,
    }

    proposal = ask_structured(
        ActionProposal,
        [
            {"role": "system", "content": PROPOSE},
            {
                "role": "user",
                "content": (
                    f"{as_untrusted(state['ticket'].body)}\n\n"
                    f"RECORD:\n{json.dumps(payload, default=str)[:_RECORD_CHARS]}\n\n"
                    "Propose one action, or 'none'. List every alternative you rejected."
                ),
            },
        ],
        agent="action",
    )

    proposal.args = _complete_args(proposal.action, proposal.args, entities)
    refusal = preflight(proposal.action, proposal.args, state["facts"])
    run.tracer.log(
        "proposal",
        agent="action",
        observation={
            "action": proposal.action,
            "args": proposal.args,
            "policy_refusal": refusal,
        },
    )

    if refusal:
        LOGGER.warning("Policy refused %s: %s", proposal.action, refusal)
    return {"proposal": proposal, "policy_refusal": refusal}


def approval_node(state: OpsMateState) -> dict[str, object]:
    """Halt for a human decision, then execute only if approved.

    interrupt() is the first statement. Nothing above it may be added, or a resume
    would re-run it and could change the proposal the human saw.
    """
    proposal = state["proposal"]
    decision = interrupt(
        {
            "kind": "approval_request",
            "run_id": state["run_id"],
            "ticket_id": state["ticket"].ticket_id,
            "action": proposal.action if proposal else "none",
            "args": proposal.args if proposal else {},
            "reason": proposal.reason if proposal else "",
            "evidence": state["facts"],
            "citations": [c.source_id for c in state["citations"]],
            "rejected": (
                [r.model_dump() for r in proposal.alternatives_rejected] if proposal else []
            ),
        }
    )

    run = current_run()
    approved = bool(decision.get("approved"))
    approver = str(decision.get("approver", "operator"))
    reason = str(decision.get("reason", ""))

    run.tracer.log(
        "approval_decision",
        agent="action",
        observation={"approved": approved, "approver": approver, "reason": reason},
    )

    if proposal is None or proposal.action == "none":
        return {
            "approval": ApprovalRecord(
                approved=False, approver=approver, reason="no action was proposed"
            )
        }

    if not approved:
        LOGGER.info("Denied: %s %s", proposal.action, proposal.args)
        return {"approval": ApprovalRecord(approved=False, approver=approver, reason=reason)}

    # Only here, and only from a real decision, does an approval enter the ledger.
    grant(state["run_id"], proposal.action, proposal.args, approver=approver, reason=reason)

    tool = BY_NAME.get(proposal.action)
    if tool is None:
        return {
            "approval": ApprovalRecord(
                approved=False, approver=approver, reason=f"unknown tool {proposal.action}"
            )
        }

    result = str(tool.invoke(proposal.args))
    executed = result.startswith("DONE")
    return {
        "approval": ApprovalRecord(
            approved=executed, approver=approver, reason=reason, result=result
        )
    }
