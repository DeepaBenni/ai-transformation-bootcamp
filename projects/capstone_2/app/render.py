"""Human-facing rendering of a run's state.

Shared by the CLI and the HTTP API so a given situation always reads the same
way. `visible_message` turns the structured run view into one natural-language
line: what happened, and what the user should do next. Everything else in the
view - the action, the evidence, the numbers, the routing - is metadata.

Boundary: pure formatting. No I/O, no model calls, imports nothing from the rest
of the app.
"""

from __future__ import annotations

from typing import Any


def fmt_args(args: dict[str, Any]) -> str:
    """Render tool arguments for a sentence, dropping the bulky handover summary."""
    inner = ", ".join(f"{key}={value}" for key, value in args.items() if key != "summary")
    return f" ({inner})" if inner else ""


def visible_message(view: dict[str, Any]) -> str:
    """One natural-language line: what happened, and what the user should do next.

    Deterministic - the same situation always produces the same wording.
    """
    pending = view.get("pending")
    changes = view.get("changes") or []
    outcome = view.get("outcome")

    if view.get("status") == "awaiting_approval" and pending:
        action = pending.get("action", "")
        cites = ", ".join(pending.get("citations") or [])
        source = f" I'm going by {cites}." if cites else ""
        if action == "escalate_to_l2":
            return (
                "I've investigated as far as I safely can, and this needs an L2 engineer. "
                "I've written up a handover with the symptom, the evidence and what I ruled "
                f"out.{source}\n\n"
                "**Nothing has been changed.** Open the details to read the handover, then "
                "choose **Approve** to send it to the L2 queue or **Decline** to hold it here."
            )
        reason = pending.get("reason", "")
        return (
            f"I think the fix is **{action}**{fmt_args(pending.get('args', {}))}"
            f"{(' - ' + reason) if reason else ''}.{source}\n\n"
            "**Nothing has been changed yet.** Open the details to check my evidence, then "
            "**Approve** to let me do it or **Decline** to stop."
        )

    if outcome == "NEEDS_INFO":
        question = view.get("question") or view.get("message") or "I need one more detail."
        return f"{question}\n\nType the answer in your next message and I'll carry straight on."

    if outcome == "ESCALATED":
        if changes:
            return (
                "Sent to L2 - an engineer will take it from the handover I prepared. "
                "Nothing on the estate was changed. Anything else I can look at?"
            )
        return (
            "I've marked this as an L2 escalation but it was NOT sent, and nothing was "
            "changed. Message me again if you'd like me to take another look."
        )

    if outcome == "RESOLVED":
        if changes:
            done = " ".join(change.get("result", "") for change in changes)
            return f"Done. {done}\n\nYou're all set - tell me if there's anything else."
        base = view.get("message") or "I looked into it and there was nothing to change."
        return (
            f"{base}\n\n**Nothing was changed.** Let me know if you'd like me to try another way."
        )

    return view.get("message") or "I'm still working on this - give me a moment."
