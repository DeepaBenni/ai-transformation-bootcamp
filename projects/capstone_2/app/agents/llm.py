"""Model access.

Two rules enforced here rather than remembered elsewhere:
  1. Every model call emits an agent_turn trace row with tokens and latency.
  2. Structured output uses include_raw=True, because the parsed object alone
     discards usage metadata and R6 requires a token count per turn.

Boundary: this module wraps the chat model. It contains no domain logic and no
prompts - callers pass their own messages.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.audit.context import current_run
from app.config import get_settings
from app.logging_conf import get_logger

LOGGER = get_logger(__name__)

# Seconds to wait for a single model response, and retry budget on transient errors.
_REQUEST_TIMEOUT_S: int = 60
_MAX_RETRIES: int = 2
_MS_PER_SECOND: int = 1000

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """Return the shared chat model, created on first use."""
    global _llm
    if _llm is None:
        settings = get_settings()
        _llm = ChatOpenAI(
            model=settings.chat_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
            timeout=_REQUEST_TIMEOUT_S,
            max_retries=_MAX_RETRIES,
        )
    return _llm


def _record(agent: str, message: AIMessage, started: float, extra: dict[str, Any]) -> None:
    """Log one agent_turn trace row with token usage and latency."""
    usage = message.usage_metadata or {}
    current_run().tracer.log(
        "agent_turn",
        agent=agent,
        model=get_settings().chat_model,
        prompt_tokens=int(usage.get("input_tokens", 0)),
        completion_tokens=int(usage.get("output_tokens", 0)),
        latency_ms=int((time.perf_counter() - started) * _MS_PER_SECOND),
        **extra,
    )


def ask_structured[TModel: BaseModel](
    schema: type[TModel],
    messages: list[BaseMessage] | list[dict[str, str]],
    *,
    agent: str,
) -> TModel:
    """Call the model and return a validated instance of `schema`.

    Args:
        schema: The Pydantic model the response must satisfy.
        messages: The message list to send (LangChain messages or role/content dicts).
        agent: Name of the calling agent, used for the trace row.

    Returns:
        A validated instance of `schema`.

    Raises:
        ValueError: if the model's output could not be parsed into the schema.
            Failing loudly is deliberate - a silently empty object would produce
            an ungrounded answer downstream.
    """
    started = time.perf_counter()
    # method="function_calling" rather than the newer strict "json_schema" default:
    # our schemas carry an open dict[str, Any] field (ActionProposal.args), which the
    # strict mode rejects for lacking additionalProperties=false.
    result = (
        get_llm()
        .with_structured_output(schema, method="function_calling", include_raw=True)
        .invoke(messages)
    )

    raw: AIMessage = result["raw"]
    _record(agent, raw, started, {"observation": {"schema": schema.__name__}})

    if result.get("parsing_error") or result.get("parsed") is None:
        current_run().tracer.log(
            "schema_violation", agent=agent, error=str(result.get("parsing_error"))
        )
        raise ValueError(f"{agent}: model output did not match {schema.__name__}")
    return result["parsed"]


def ask_text(messages: list[BaseMessage] | list[dict[str, str]], *, agent: str) -> str:
    """Call the model for free-form prose and return its text."""
    started = time.perf_counter()
    message = get_llm().invoke(messages)
    _record(agent, message, started, {})
    return str(message.content)
