"""Retrieval over the KB collection, with the conflict rule applied.

Boundary: this module reads from Chroma and runs the deterministic conflict
resolver. It does not embed new documents (that is app/kb/ingest.py) and it does
not call an LLM.
"""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.kb.conflict import ConflictOutcome, resolve_conflicts

# Opened once on first search and reused; Chroma handles are safe to share.
_store: Chroma | None = None


def get_store() -> Chroma:
    """Return the shared Chroma handle, opening it on first use.

    Raises:
        SystemExit: if the collection is empty, with the command that fixes it.
    """
    global _store
    if _store is None:
        settings = get_settings()
        _store = Chroma(
            collection_name=settings.kb_collection,
            embedding_function=OpenAIEmbeddings(
                model=settings.embed_model, api_key=settings.openai_api_key
            ),
            persist_directory=str(settings.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        if _store._collection.count() == 0:
            raise SystemExit("KB collection is empty. Run: python -m app.kb.ingest")
    return _store


def chunk_count() -> int:
    """Return the number of embedded chunks. Used by the health endpoint."""
    try:
        return int(get_store()._collection.count())
    except Exception:  # health probe: treat any failure as "no chunks available"
        return 0


def search(query: str, *, k: int | None = None, intent: str = "procedure") -> dict[str, Any]:
    """Search the KB and resolve any contradictions among the hits.

    Args:
        query: Natural-language description of the problem.
        k: Number of chunks to retrieve; defaults to settings.retrieval_k.
        intent: Passed through to the conflict resolver ("procedure" or "diagnosis").

    Returns:
        A mapping with `hits` (id, title, text, score, status, last_reviewed),
        `top_score`, `below_floor`, `superseded`, `unresolved` and `notes`.
        `below_floor` is the signal the Knowledge agent uses to refuse rather
        than improvise.
    """
    settings = get_settings()
    scored = get_store().similarity_search_with_relevance_scores(query, k=k or settings.retrieval_k)
    if not scored:
        return {
            "hits": [],
            "top_score": 0.0,
            "below_floor": True,
            "superseded": [],
            "unresolved": [],
            "notes": [],
        }

    documents = [doc for doc, _ in scored]
    # Key scores by id(doc) so we can look the score back up after the resolver
    # has filtered and reordered the document list.
    scores = {id(doc): score for doc, score in scored}
    outcome: ConflictOutcome = resolve_conflicts(documents, intent=intent)

    hits = [
        {
            "id": doc.metadata.get("id", ""),
            "title": doc.metadata.get("title", ""),
            "type": doc.metadata.get("type", ""),
            "status": doc.metadata.get("status", ""),
            "last_reviewed": doc.metadata.get("last_reviewed", ""),
            "score": round(scores.get(id(doc), 0.0), 4),
            "text": doc.page_content[:900],
        }
        for doc in outcome.documents
    ]
    top_score = max((hit["score"] for hit in hits), default=0.0)

    return {
        "hits": hits,
        "top_score": top_score,
        "below_floor": top_score < settings.relevance_floor,
        "superseded": outcome.superseded,
        "unresolved": [list(pair) for pair in outcome.unresolved],
        "notes": outcome.notes,
    }
