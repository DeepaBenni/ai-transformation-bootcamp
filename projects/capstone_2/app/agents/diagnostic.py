"""Diagnostic: establish facts with read-only tools.

Built on create_react_agent, scoped to DIAGNOSTIC_TOOLS. The scope is the
security boundary - there is no writer in the list, so no injection against this
agent can produce a state change.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from app.agents.llm import get_llm
from app.agents.prompts import DIAGNOSTIC
from app.agents.state import OpsMateState
from app.audit.context import current_run
from app.config import get_settings
from app.logging_conf import get_logger
from app.safety.sanitize import as_untrusted
from app.tools.registry import DIAGNOSTIC_TOOLS

LOGGER = get_logger(__name__)

# Recursion limit for the sub-agent: roughly four tool calls plus their reasoning turns.
MAX_TOOL_HOPS = 8


def _build_agent():  # noqa: ANN202 - LangGraph's compiled graph type is not public
    """Create the diagnostic ReAct agent, scoped to read-only tools."""
    return create_react_agent(model=get_llm(), tools=DIAGNOSTIC_TOOLS, prompt=DIAGNOSTIC)


def diagnostic_node(state: OpsMateState) -> dict[str, object]:
    """Gather facts. Merges observations into state['facts'] keyed by tool name."""
    run = current_run()
    entities = state["intake"].entities.model_dump() if state["intake"] else {}

    prompt = (
        f"{as_untrusted(state['ticket'].body)}\n\n"
        f"Identifiers: {json.dumps(entities)}\n"
        f"Facts already established: {json.dumps(list(state['facts']))}\n"
        f"Establish the remaining facts with the fewest tool calls."
    )

    result = _build_agent().invoke(
        {"messages": [HumanMessage(content=prompt)]},
        {"recursion_limit": MAX_TOOL_HOPS},
    )

    facts = dict(state["facts"])
    calls: list[str] = []
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            if message.usage_metadata:
                usage = message.usage_metadata
                run.tracer.log(
                    "agent_turn",
                    agent="diagnostic",
                    model=get_settings().chat_model,
                    prompt_tokens=int(usage.get("input_tokens", 0)),
                    completion_tokens=int(usage.get("output_tokens", 0)),
                )
            calls.extend(call["name"] for call in (message.tool_calls or []))

    # The tools already traced their own observations; re-read them from the trace
    # so facts and the audit record cannot diverge.
    for event in run.tracer.events:
        if event["event"] == "observation" and event["tool"]:
            facts[event["tool"]] = event["observation"]

    summary = str(result["messages"][-1].content)
    run.tracer.log(
        "diagnostic_done",
        agent="diagnostic",
        observation={"tools_called": calls, "summary": summary[:400]},
    )
    return {"facts": facts}
