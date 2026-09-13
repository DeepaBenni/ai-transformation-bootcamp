"""The OpsMate HTTP API.

Seven endpoints. The interesting one is POST /runs/{run_id}/approve: it is the
only path by which an approval can enter the ledger, and it takes the approver's
identity from the request, never from the graph.

Boundary: this module is transport only - request parsing, status codes, and
handing off to OpsMateService. All orchestration lives in app/api/service.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    ApprovalRequest,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ReplyRequest,
    RunResponse,
    RunSummary,
    SessionDetail,
    SessionRename,
    SessionSummary,
    TicketRequest,
    TraceEvent,
)
from app.api.service import OpsMateService
from app.config import get_settings
from app.db.session import ping
from app.kb.retriever import chunk_count
from app.logging_conf import configure_logging, get_logger

LOGGER = get_logger(__name__)
service: OpsMateService | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Compile the graph once at startup rather than per request."""
    global service
    configure_logging()
    service = OpsMateService()
    LOGGER.info("OpsMate API ready")
    yield
    LOGGER.info("OpsMate API shutting down")


app = FastAPI(
    title="OpsMate",
    version="1.0.0",
    description=(
        "Agentic L1 service desk assistant. Investigates freely; changes nothing "
        "without human approval."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local development only
    allow_methods=["*"],
    allow_headers=["*"],
)


def _service() -> OpsMateService:
    """Return the started service, or 503 if the graph is still compiling."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service is still starting")
    return service


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    """Report whether every dependency is reachable."""
    settings = get_settings()
    chunks = chunk_count()
    mysql_ok = ping()
    return {
        "status": "ok" if (mysql_ok and chunks > 0) else "degraded",
        "mysql": mysql_ok,
        "chroma_chunks": chunks,
        "chat_model": settings.chat_model,
        "embed_model": settings.embed_model,
        "active_runs": _service().active_runs,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict[str, Any]:
    """The conversational front door.

    Every message goes through here first. Small talk and questions about the
    assistant are answered directly; an out-of-scope request is refused with a
    fixed statement of what the assistant is for; anything that looks like a
    service-desk issue starts a ticket run (whose state is returned in `run`).
    The conversation's memory - the user's name, earlier turns - is threaded into
    whichever path is taken.
    """
    try:
        return _service().chat(request.session_id, request.message)
    except Exception as exc:  # API boundary: any failure becomes a 500 with its cause
        LOGGER.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/sessions", response_model=list[SessionSummary])
def list_sessions() -> list[dict[str, Any]]:
    """List every chat session, most recently used first."""
    return _service().list_sessions()


@app.post("/sessions", response_model=SessionSummary)
def create_session() -> dict[str, Any]:
    """Start a fresh, empty session (the 'New chat' button)."""
    return _service().new_session()


@app.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str) -> dict[str, Any]:
    """Return a session's metadata and its full transcript, for reloading it."""
    try:
        return _service().session_detail(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"No such session: {session_id}") from exc


@app.patch("/sessions/{session_id}", response_model=SessionSummary)
def rename_session(session_id: str, request: SessionRename) -> dict[str, Any]:
    """Rename a session."""
    try:
        return _service().rename_session(session_id, request.title)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"No such session: {session_id}") from exc


@app.post("/tickets", response_model=RunResponse)
def submit_ticket(request: TicketRequest) -> dict[str, Any]:
    """Submit a ticket for triage, bypassing the conversational front door.

    Returns when the run finishes or suspends for approval. A suspended run
    carries a `pending` block with the proposed action, the evidence behind it,
    and the alternatives that were rejected.
    """
    try:
        return _service().start(request.body, request.ticket_id, request.reporter)
    except Exception as exc:  # API boundary: any failure becomes a 500 with its cause
        LOGGER.exception("Ticket run failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> dict[str, Any]:
    """Return the current state of a run."""
    try:
        return _service().snapshot(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"No such run: {run_id}") from exc


@app.post("/runs/{run_id}/approve", response_model=RunResponse)
def decide(
    run_id: str,
    request: ApprovalRequest,
    x_approver: str | None = Header(default=None, alias="X-Approver"),
) -> dict[str, Any]:
    """Approve or decline a pending state change, and resume the run.

    The approver's identity comes from the request or the X-Approver header -
    never from anything the graph or the ticket produced. In production this
    would come from an authenticated session with a role check; that gap is
    named in the governance note.
    """
    decision = {
        "approved": request.approved,
        "approver": x_approver or request.approver,
        "reason": request.reason,
    }
    try:
        return _service().resume(run_id, decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"No such run: {run_id}") from exc


@app.post("/runs/{run_id}/reply", response_model=RunResponse)
def reply(run_id: str, request: ReplyRequest) -> dict[str, Any]:
    """Answer a clarifying question by submitting a new, enriched ticket.

    A clarification is a fresh run rather than a resumed one: the answer changes
    the facts, so re-running intake on the complete text is more honest than
    patching state mid-flight.
    """
    try:
        previous = _service().snapshot(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"No such run: {run_id}") from exc

    # Rebuild a complete ticket: the original text, then the answer to the one
    # question Intake asked. Re-running on the whole thing is more honest than
    # patching state mid-flight.
    original = _service().ticket_body(run_id)
    question = previous.get("question") or "the missing detail"
    enriched = (
        f"{original}\n\n"
        f"[Clarification] We asked: {question}\n"
        f"The user answered: {request.answer}"
    )
    # runs.ticket_id is VARCHAR(24); keep a short link to the run being answered.
    return _service().start(enriched, ticket_id=f"reply-{run_id[-6:]}")


@app.get("/runs/{run_id}/trace", response_model=list[TraceEvent])
def get_trace(run_id: str) -> list[dict[str, Any]]:
    """Return the complete audit trail for a run."""
    events = _service().trace(run_id)
    if not events:
        raise HTTPException(status_code=404, detail=f"No trace for run: {run_id}")
    return events


@app.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    """List recent runs, newest first."""
    return _service().list_runs(limit)
