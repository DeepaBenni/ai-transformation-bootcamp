"""The supervisor.

Two layers. Layer 1 is a deterministic precondition ladder that settles most
tickets with no model call at all. Layer 2 is an LLM router, consulted only when
no rule applies, and fenced on the way out as well as on the way in.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from app.agents.llm import ask_structured
from app.agents.prompts import SUPERVISOR
from app.agents.state import NextNode, OpsMateState, RouteStep
from app.audit.context import current_run
from app.logging_conf import get_logger

LOGGER = get_logger(__name__)

_FACT_CHARS: int = 220
_DIGEST_CHARS: int = 2500
_HISTORY_TURNS: int = 4
_HISTORY_CHARS: int = 140


class Route(BaseModel):
    """The supervisor's choice of next specialist."""

    next: NextNode
    reason: str = Field(description="One sentence. It will be quoted as the stop reason.")


# A route is legal only when its preconditions hold, whoever chose it.
PRECONDITIONS: dict[str, Callable[[OpsMateState], bool]] = {
    "propose": lambda s: bool(s["facts"]) and bool(s["citations"]),
    "respond": lambda s: bool(s["facts"]),
}


def _forced_route(state: OpsMateState) -> tuple[str, str] | None:
    """Return (next_node, reason) when the route is not a judgement call.

    Ordered by precedence: termination and safety first, then data dependencies.
    """
    if state["hops"] > state["max_hops"]:
        return "escalate", (
            f"hop cap of {state['max_hops']} reached without reaching a resolvable path"
        )

    if state["policy_refusal"]:
        return "escalate", state["policy_refusal"]

    if state["approval"] is not None:
        return "respond", "a human has decided; report the outcome"

    # "Refuse and escalate immediately" (R7): a leaver's disabled account is HR's,
    # never the Service Desk's. This is a rule, not a judgement, so it is settled
    # here rather than left to the router or the policy pre-check.
    account = state["facts"].get("check_account_lockout") or {}
    user = state["facts"].get("get_user") or {}
    if account.get("state") == "disabled" or user.get("employment_status") == "leaver":
        return "escalate", (
            "the account is disabled and the employee is a leaver; re-enabling belongs to "
            "HR offboarding, not the Service Desk (RB-01 step 2, RB-03)"
        )

    intake = state["intake"]
    if intake and not intake.sufficient:
        return "clarify", f"intake reports a missing field: {intake.missing_field}"

    knowledge = state["knowledge"]
    if knowledge and knowledge.not_covered:
        return "escalate", "the knowledge base does not cover this issue; refusing to improvise"
    if knowledge and knowledge.conflict_unresolved:
        return (
            "escalate",
            "two live knowledge base documents conflict; resolution is an L2 decision",
        )

    if not state["facts"]:
        return "diagnostic", "no facts have been established yet"
    if not state["citations"] and not knowledge:
        return "knowledge", "facts are established but no guidance has been retrieved"

    return None


def _digest(state: OpsMateState) -> str:
    """Summarise state for the router. Deliberately excludes the raw ticket text.

    A few recent conversation turns are included so the router can weigh a
    follow-up in context ("he asked twice already, escalate").
    """
    knowledge = state["knowledge"]
    payload: dict[str, Any] = {
        "intent": state["intake"].intent if state["intake"] else None,
        "entities": state["intake"].entities.model_dump() if state["intake"] else {},
        "conversation": [
            f"{t.get('role', '?')}: {t.get('content', '')[:_HISTORY_CHARS]}"
            for t in state["history"][-_HISTORY_TURNS:]
        ],
        "facts_gathered": list(state["facts"]),
        "facts": {k: str(v)[:_FACT_CHARS] for k, v in state["facts"].items()},
        "citations": [c.source_id for c in state["citations"]],
        "claims": [c.text for c in knowledge.claims] if knowledge else [],
        "hops": state["hops"],
        "max_hops": state["max_hops"],
    }
    return json.dumps(payload, default=str)[:_DIGEST_CHARS]


def supervisor_node(state: OpsMateState) -> dict[str, object]:
    """Choose the next specialist. Deterministic where possible, model only for the residue."""
    run = current_run()
    hops = state["hops"] + 1
    working = {**state, "hops": hops}

    forced = _forced_route(working)  # type: ignore[arg-type]
    if forced is not None:
        node, why = forced
        step = RouteStep(hop=hops, to=node, why=why, by="rule")
        run.tracer.log("route", agent="supervisor", observation=step.model_dump())
        return {
            "hops": hops,
            "next_node": node,
            "stop_reason": why,
            "route_history": [*state["route_history"], step],
        }

    choice = ask_structured(
        Route,
        [
            {"role": "system", "content": SUPERVISOR},
            {"role": "user", "content": f"STATE:\n{_digest(working)}\n\nChoose the next step."},
        ],
        agent="supervisor",
    )

    node, why = choice.next, choice.reason
    guard = PRECONDITIONS.get(node)
    if guard is not None and not guard(working):  # type: ignore[arg-type]
        LOGGER.warning("Router chose %s without preconditions; overriding", node)
        node = "diagnostic" if not working["facts"] else "knowledge"
        why = f"router chose {choice.next} without its preconditions; gathering evidence instead"

    step = RouteStep(hop=hops, to=node, why=why, by="model")
    run.tracer.log("route", agent="supervisor", observation=step.model_dump())
    return {
        "hops": hops,
        "next_node": node,
        "stop_reason": why,
        "route_history": [*state["route_history"], step],
    }
