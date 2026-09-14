"""Embed the knowledge base into ChromaDB.

Each markdown file carries YAML frontmatter, which becomes chunk metadata. The
metadata is what makes the conflict rule possible, so it is attached to every
chunk, not just the first.

    python -m app.kb.ingest              # build if empty
    python -m app.kb.ingest --rebuild    # delete and rebuild
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Final

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.logging_conf import configure_logging, get_logger

LOGGER = get_logger(__name__)

# Matches a leading "--- ... ---" YAML frontmatter block at the top of a file.
_FRONTMATTER: Final[re.Pattern[str]] = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Split a document into its frontmatter mapping and its body text."""
    match = _FRONTMATTER.match(raw)
    if match is None:
        return {}, raw

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, raw[match.end() :]


def load_documents(source_dir: Path) -> list[Document]:
    """Read every markdown file into a Document with frontmatter as metadata.

    Chroma metadata values must be str, int, float or bool - never None. Absent
    fields become the empty string, which is why the conflict rule tests for
    truthiness rather than for None.
    """
    documents: list[Document] = []
    for path in sorted(source_dir.glob("*.md")):
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        documents.append(
            Document(
                page_content=f"{meta.get('title', path.stem)}. {body.strip()}",
                metadata={
                    "id": meta.get("id", path.stem),
                    "title": meta.get("title", path.stem),
                    "type": meta.get("type", "article"),
                    "status": meta.get("status", "current"),
                    "last_reviewed": meta.get("last_reviewed", ""),
                    "conflicts_with": meta.get("conflicts_with", ""),
                    "source_file": path.name,
                },
            )
        )
    return documents


def _open_collection(embeddings: OpenAIEmbeddings) -> Chroma:
    """Open (or create) the persistent KB collection with cosine similarity."""
    settings = get_settings()
    return Chroma(
        collection_name=settings.kb_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_store(*, rebuild: bool = False) -> Chroma:
    """Return the KB collection, building it if it is empty or a rebuild is asked for."""
    settings = get_settings()
    embeddings = OpenAIEmbeddings(model=settings.embed_model, api_key=settings.openai_api_key)
    store = _open_collection(embeddings)

    # store._collection.count() is the only way to size the collection -
    # langchain-chroma exposes no public API for it.
    if rebuild and store._collection.count() > 0:
        LOGGER.warning("Rebuilding: deleting %d existing chunks", store._collection.count())
        store.delete_collection()
        store = _open_collection(embeddings)

    if store._collection.count() > 0:
        LOGGER.info("Collection already holds %d chunks; nothing to do", store._collection.count())
        return store

    documents = load_documents(settings.kb_source_dir)
    if not documents:
        raise SystemExit(
            f"No markdown files found in {settings.kb_source_dir}\n"
            "Copy the Day 12 knowledge base into data/capstone_2_knowledge_base/ first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    store.add_documents(chunks)
    LOGGER.info("Embedded %d documents into %d chunks", len(documents), len(chunks))
    return store


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Embed the OpsMate knowledge base.")
    parser.add_argument("--rebuild", action="store_true", help="delete the collection first")
    args = parser.parse_args()

    configure_logging()
    store = build_store(rebuild=args.rebuild)
    LOGGER.info(
        "Collection '%s' holds %d chunks",
        get_settings().kb_collection,
        store._collection.count(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
