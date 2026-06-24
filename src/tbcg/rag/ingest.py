"""Ingest the local knowledge base into ChromaDB.

Loads every text/markdown file under ``knowledge_base/``, splits into chunks,
embeds locally, and upserts with stable ids (so re-running is idempotent).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..rag.embeddings import get_embeddings
from ..tools.rag import get_collection, reset_collection

SUPPORTED_SUFFIXES = {".md", ".txt"}


def _chunk_id(source: str, idx: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{idx}:{text}".encode()).hexdigest()[:16]
    return f"{source}-{idx}-{digest}"


def ingest_directory(kb_dir: str | Path, reset: bool = True) -> dict[str, int]:
    """Ingest all supported files under ``kb_dir``. Returns counts.

    ``reset=True`` (default) drops the collection first so docs removed from the
    KB don't linger as stale vectors — the KB folder is the source of truth.
    """
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_path}")

    files = [p for p in kb_path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        return {"files": 0, "chunks": 0}

    if reset:
        reset_collection()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", ". ", " ", ""]
    )
    embeddings = get_embeddings()
    collection = get_collection()

    total_chunks = 0
    for path in files:
        source = path.relative_to(kb_path).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = splitter.split_text(text)
        if not chunks:
            continue

        ids = [_chunk_id(source, i, c) for i, c in enumerate(chunks)]
        vectors = embeddings.embed_documents(chunks)
        metadatas = [{"source": source, "chunk": i} for i in range(len(chunks))]

        collection.upsert(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)
        total_chunks += len(chunks)

    return {"files": len(files), "chunks": total_chunks}
