"""Assemble and compile the OpsMate graph.

Boundary: this module wires nodes to edges and nothing else. Node behaviour lives
in the per-agent modules; the endpoints (`clarify`, `escalate`, `respond`) are
small enough to keep here.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.action import approval_node, propose_node
from app.agents.diagnostic import diagnostic_node
from app.agents.handover import build_handover, render_handover
from app.agents.intake import intake_node
from app.agents.knowledge import knowledge_node
from app.agents.llm import ask_text
from app.agents.prompts import RESPOND
from app.agents.state import ActionProposal, FinalResponse, OpsMateState
from app.agents.supervisor import supervisor_node
from app.audit.context import current_run
from app.logging_conf import get_logger

LOGGER = get_logger(__name__)

_DEFAULT_DIAGRAM_PATH: str = "docs/graph.png"


# ------------------------------------------------------------- endpoints --
def clarify_node(state: OpsMateState) -> dict[str, object]:
    """Ask the one question Intake identified. No model call is needed."""
    intake = state["intake"]
    question = (
        intake.clarifying_question
        if intake and intake.clarifying_question
        else "Could you tell us which user account or device is affected?"
    )
    current_run().tracer.log("clarify", agent="supervisor", observation={"question": question})
    return {
        "path": "clarify",
        "final": FinalResponse(outcome="NEEDS_INFO", message=question, action_executed=False),
    }


def escalate_node(state: OpsMateState) -> dict[str, object]:
    """Build the L2 handover and stage the gated escalate_to_l2 call."""
    handover = build_handover(state)
    rendered = render_handover(handover, state["ticket"].ticket_id)
    proposal = ActionProposal(
        action="escalate_to_l2",
        args={"ticket_id": state["ticket"].ticket_id, "summary": rendered},
        reason=state["stop_reason"] or "requires an L2 engineer",
        alternatives_rejected=handover.actions_rejected,
    )
    return {
        "path": "escalate",
        "handover": handover,
        "proposal": proposal,
        "policy_refusal": None,
    }


def respond_node(state: OpsMateState) -> dict[str, object]:
    """Compose the final message from the state record only. This is Lock 3.

    The ticket text is deliberately NOT supplied. Text injected into a ticket
    therefore cannot reach the sentence a user finally reads.
    """
    approval = state["approval"]
    executed = bool(approval and approval.approved)
    path = (
        state["path"]
        if state["path"] != "undecided"
        else ("escalate" if state["handover"] else "resolve")
    )

    if path == "escalate" and state["handover"]:
        message = render_handover(state["handover"], state["ticket"].ticket_id)
        if not executed:
            message += "\n\nNOTE: the escalation was not confirmed, so nothing was queued to L2."
        outcome = "ESCALATED"
    else:
        record = {
            "account_state": (state["facts"].get("check_account_lockout") or {}).get("state"),
            "employment_status": (state["facts"].get("get_user") or {}).get("employment_status"),
            "citations": [c.source_id for c in state["citations"]],
            "claims": [c.text for c in state["knowledge"].claims] if state["knowledge"] else [],
            "action_requested": state["proposal"].action if state["proposal"] else "none",
            "action_executed": executed,
            "execution_result": approval.result if approval else "",
            "approver": approval.approver if approval else "none",
        }
        message = ask_text(
            [
                {"role": "system", "content": RESPOND},
                {"role": "user", "content": f"RECORD:\n{record}"},
            ],
            agent="respond",
        )
        outcome = "RESOLVED" if executed else ("ESCALATED" if path == "escalate" else "RESOLVED")

    final = FinalResponse(
        outcome=outcome,  # type: ignore[arg-type]
        message=message,
        citations=[c.source_id for c in state["citations"]],
        action_executed=executed,
    )
    current_run().tracer.log("run_end", observation={"outcome": outcome, "executed": executed})
    return {"path": path, "final": final}


# ---------------------------------------------------------------- routing --
def route_from_supervisor(state: OpsMateState) -> str:
    """Read the decision the supervisor node already recorded."""
    return state["next_node"] or "respond"


def route_after_propose(state: OpsMateState) -> str:
    """A policy refusal returns to the supervisor, which will escalate."""
    return "supervisor" if state["policy_refusal"] else "approval"


# ----------------------------------------------------------------- build --
def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Assemble the graph: nodes first, then edges, so the shape stays readable."""
    graph = StateGraph(OpsMateState)

    graph.add_node("intake", intake_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("diagnostic", diagnostic_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("propose", propose_node)
    graph.add_node("approval", approval_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "diagnostic": "diagnostic",
            "knowledge": "knowledge",
            "propose": "propose",
            "clarify": "clarify",
            "escalate": "escalate",
            "respond": "respond",
        },
    )
    graph.add_edge("diagnostic", "supervisor")
    graph.add_edge("knowledge", "supervisor")
    graph.add_conditional_edges(
        "propose", route_after_propose, {"approval": "approval", "supervisor": "supervisor"}
    )
    graph.add_edge("escalate", "approval")
    graph.add_edge("approval", "respond")
    graph.add_edge("clarify", END)
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=checkpointer)


def export_diagram(path: str = _DEFAULT_DIAGRAM_PATH) -> None:
    """Write the rendered graph to disk. This image is R1 evidence."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(build_graph().get_graph().draw_mermaid_png())
    LOGGER.info("Graph diagram written to %s", path)
