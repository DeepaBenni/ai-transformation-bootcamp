"""The conversation front door.

Every message the user types reaches this module first. It decides whether the
message is a ticket (run the agent graph), small talk (answer directly), or out of
scope (refuse, and say what the assistant is for). It is given the conversation
history so a follow-up like "who am I" can be answered from what was said earlier.

Boundary: classification and a short reply only - no tools, no graph, no writes.
The out-of-scope refusal text is a fixed constant, not model output, so the stated
scope of the assistant cannot drift from one refusal to the next.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.agents.llm import ask_structured
from app.agents.prompts import SCOPE_REFUSAL, TRIAGE
from app.safety.sanitize import as_untrusted

# How much of the conversation the classifier sees. Enough to resolve a reference
# a few turns back without spending the whole context window on chit-chat.
_HISTORY_TURNS: int = 12


class TriageResult(BaseModel):
    """What the front door decided about one user message."""

    kind: Literal["ticket", "smalltalk", "out_of_scope"] = Field(
        description="ticket = run the service-desk graph; smalltalk = answer here; "
        "out_of_scope = refuse."
    )
    reply: str = Field(
        default="",
        description="The reply to show for smalltalk. Empty for ticket and out_of_scope.",
    )
    remember: dict[str, str] = Field(
        default_factory=dict,
        description='Facts to carry forward, e.g. {"user_name": "Ankit"}. Empty if none.',
    )


def _format_history(history: list[dict[str, str]]) -> str:
    """Render recent turns as a short transcript for the classifier."""
    recent = history[-_HISTORY_TURNS:]
    if not recent:
        return "(no earlier turns)"
    return "\n".join(f"{turn.get('role', '?')}: {turn.get('content', '')}" for turn in recent)


def triage(
    message: str,
    history: list[dict[str, str]],
    known_facts: dict[str, str],
) -> TriageResult:
    """Classify one message and, for small talk, write the reply.

    Args:
        message: The newest thing the user said.
        history: Prior turns, oldest first, as [{role, content}].
        known_facts: What is already remembered about the user, e.g. their name.

    Returns:
        A TriageResult. For `out_of_scope` the reply is replaced with the fixed
        `SCOPE_REFUSAL` text so it is identical every time.
    """
    result = ask_structured(
        TriageResult,
        [
            {"role": "system", "content": TRIAGE},
            {
                "role": "user",
                "content": (
                    f"KNOWN FACTS: {known_facts or '(none)'}\n\n"
                    f"CONVERSATION SO FAR:\n{_format_history(history)}\n\n"
                    f"NEWEST MESSAGE:\n{as_untrusted(message)}\n\n"
                    "Classify the newest message."
                ),
            },
        ],
        agent="triage",
    )
    if result.kind == "out_of_scope":
        result.reply = SCOPE_REFUSAL
    return result
