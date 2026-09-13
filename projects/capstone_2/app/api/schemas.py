"""Request and response models for the HTTP API.

These are the API's contract. They are deliberately separate from the agent
state models: the graph may be refactored without breaking clients.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RunStatus = Literal["running", "awaiting_approval", "awaiting_reply", "done", "error"]


class TicketRequest(BaseModel):
    """A ticket submitted for triage."""

    body: str = Field(min_length=3, description="The ticket text as written by the user.")
    ticket_id: str = Field(
        default="INC-API",
        max_length=24,
        description="Ticket reference (max 24 chars - matches the runs.ticket_id column).",
    )
    reporter: str | None = None


class PendingApproval(BaseModel):
    """Everything an approver needs to decide, shown before they decide."""

    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    citations: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    rejected: list[dict[str, Any]] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    """A human decision on a pending change."""

    approved: bool
    approver: str = Field(default="operator", description="Who is accountable for this decision.")
    reason: str = ""


class ReplyRequest(BaseModel):
    """A user's answer to a clarifying question."""

    answer: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """One message in a conversation with the assistant."""

    session_id: str = Field(
        min_length=1,
        max_length=64,
        description="The chat session this message belongs to. Every turn is persisted "
        "against it, so the session history panel can reload the conversation.",
    )
    message: str = Field(min_length=1, description="What the user typed.")


class SessionRename(BaseModel):
    """A new title for a chat session."""

    title: str = Field(min_length=1, max_length=200)


class ChatMessage(BaseModel):
    """One stored turn of a chat session."""

    seq: int
    role: Literal["user", "assistant"]
    kind: str = "text"
    content: str
    run_id: str | None = None
    payload: dict[str, Any] | None = None
    created_at: str


class SessionSummary(BaseModel):
    """One row in the session history panel."""

    session_id: str
    seq: int
    title: str
    message_count: int = 0
    created_at: str = ""
    updated_at: str = ""


class SessionDetail(BaseModel):
    """A session plus its full transcript, for reloading it in the UI."""

    session_id: str
    seq: int
    title: str
    facts: dict[str, str] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)


class RunResponse(BaseModel):
    """The state of one run, whether suspended or finished."""

    run_id: str
    status: RunStatus
    user_visible_message: str = Field(
        default="",
        description="A natural-language summary of where things stand and what the user "
        "should do next. This is the line to show in a chat UI; everything else is metadata.",
    )
    path: str = "undecided"
    outcome: str | None = None
    message: str | None = None
    citations: list[str] = Field(default_factory=list)
    question: str | None = None
    pending: PendingApproval | None = None
    changes: list[dict[str, Any]] = Field(default_factory=list)
    hops: int = 0
    stop_reason: str | None = None
    route_history: list[dict[str, Any]] = Field(default_factory=list)
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


class ChatResponse(BaseModel):
    """The assistant's answer to one chat message.

    `kind` says how to read it: "smalltalk" and "out_of_scope" carry a plain
    `reply`; "ticket" means the message started a service-desk run, whose full
    state is in `run` (which may be suspended for approval).
    """

    session_id: str
    kind: Literal["ticket", "smalltalk", "out_of_scope"]
    reply: str = Field(
        default="",
        description="The natural-language line to show the user. For a ticket it mirrors "
        "run.user_visible_message; for small talk / out of scope it is the whole reply.",
    )
    run: RunResponse | None = None


class RunSummary(BaseModel):
    """One row in the runs list."""

    run_id: str
    ticket_id: str | None
    path: str
    status: str
    stop_reason: str | None
    total_tokens: int
    cost_usd: float
    latency_ms: int
    created_at: str


class TraceEvent(BaseModel):
    """One audit row."""

    seq: int
    ts: str
    agent: str | None
    event: str
    tool: str | None
    args: Any = None
    observation: Any = None
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str | None = None


class HealthResponse(BaseModel):
    """Dependency health."""

    status: Literal["ok", "degraded"]
    mysql: bool
    chroma_chunks: int
    chat_model: str
    embed_model: str
    active_runs: int
