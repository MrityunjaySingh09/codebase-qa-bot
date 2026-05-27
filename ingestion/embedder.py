"""
ingestion/embedder.py — Embed code chunks and persist them in ChromaDB.

Design decisions:
  - One ChromaDB collection per repo (keyed by owner__name).
    This enables multi-repo support and clean deletion.
  - We check content_hash before re-embedding so repeated indexing
    of the same repo is near-instant (only changed files are re-embedded).
  - Embeddings are generated via Ollama's nomic-embed-text model
    (local, free, 768-dim, good at code).
  - Batch size of 32 balances throughput vs Ollama memory pressure.
  - We store the raw content in ChromaDB documents so we can retrieve
    it without a separate lookup.
"""

import hashlib
from pathlib import Path
from typing import Callable, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from tqdm import tqdm

from config import get_settings
from ingestion.parser import CodeChunk

# ---------------------------------------------------------------------------
# ChromaDB client (singleton, lazy init)
# ---------------------------------------------------------------------------

_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        cfg = get_settings()
        cfg.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(cfg.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB initialised at {cfg.chroma_persist_dir}")
    return _chroma_client


def _collection_name(repo_url: str) -> str:
    """
    Derive a stable ChromaDB collection name from the repo URL.
    ChromaDB collection names must be 3-63 chars, alphanumeric + underscores.
    """
    cfg = get_settings()
    # e.g. "https://github.com/owner/repo" → "owner__repo"
    slug = repo_url.rstrip("/").split("github.com/")[-1].replace("/", "__").lower()
    # Sanitise to ChromaDB-safe characters
    slug = "".join(c if c.isalnum() or c == "_" else "_" for c in slug)
    return f"{cfg.chroma_collection_prefix}{slug}"[:63]


# ---------------------------------------------------------------------------
# Embedding model (lazy init)
# ---------------------------------------------------------------------------

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        cfg = get_settings()
        if cfg.embedding_provider.lower() == "huggingface":
            _embed_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Embedding model: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)")
        else:
            _embed_model = OllamaEmbeddings(
                model=cfg.ollama_embed_model,
                base_url=cfg.ollama_base_url,
            )
            logger.info(f"Embedding model: {cfg.ollama_embed_model} via {cfg.ollama_base_url}")
    return _embed_model


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _already_indexed(collection: chromadb.Collection, chunk: CodeChunk) -> bool:
    """
    Return True if this exact chunk content is already in the collection.
    We check by chunk_id first (fast), then verify content_hash matches.
    """
    try:
        result = collection.get(ids=[chunk.chunk_id], include=["metadatas"])
        if result["ids"]:
            stored_hash = result["metadatas"][0].get("content_hash", "")
            return stored_hash == chunk.content_hash
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_and_store(
    chunks: list[CodeChunk],
    repo_url: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    batch_size: int = 32,
) -> dict:
    """
    Embed a list of CodeChunks and store them in ChromaDB.

    Args:
        chunks:            Chunks produced by ingestion/parser.py.
        repo_url:          Canonical GitHub URL (used as collection key).
        progress_callback: fn(message, current, total) for UI updates.
        batch_size:        How many chunks to embed per Ollama call.

    Returns:
        Stats dict: {total, new, skipped, failed, collection_name}
    """
    client = _get_chroma_client()
    embed_model = _get_embed_model()
    col_name = _collection_name(repo_url)

    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for embeddings
    )

    stats = {"total": len(chunks), "new": 0, "skipped": 0, "failed": 0,
             "collection_name": col_name}

    # Split into new vs cached
    new_chunks: list[CodeChunk] = []
    for chunk in chunks:
        if _already_indexed(collection, chunk):
            stats["skipped"] += 1
        else:
            new_chunks.append(chunk)

    logger.info(
        f"Embedding {len(new_chunks)} new chunks "
        f"({stats['skipped']} cached) into '{col_name}'"
    )

    if not new_chunks:
        if progress_callback:
            progress_callback("All chunks already indexed — cache hit!", 1, 1)
        return stats

    # Process in batches
    total_batches = (len(new_chunks) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        batch = new_chunks[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_num = batch_idx + 1

        if progress_callback:
            progress_callback(
                f"Embedding batch {batch_num}/{total_batches} "
                f"({len(batch)} chunks) ...",
                batch_num,
                total_batches,
            )

        # Build texts with a context prefix so the model understands it's code
        texts = [
            f"# File: {c.file_path} | Symbol: {c.symbol_name} | Lang: {c.language}\n\n{c.content}"
            for c in batch
        ]

        try:
            embeddings = embed_model.embed_documents(texts)
        except Exception as exc:
            logger.error(f"Embedding batch {batch_num} failed: {exc}")
            stats["failed"] += len(batch)
            continue

        # Upsert into ChromaDB
        try:
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=[c.content for c in batch],
                metadatas=[c.to_metadata() for c in batch],
            )
            stats["new"] += len(batch)
        except Exception as exc:
            logger.error(f"ChromaDB upsert failed for batch {batch_num}: {exc}")
            stats["failed"] += len(batch)

    logger.success(
        f"Indexing complete: {stats['new']} new, "
        f"{stats['skipped']} cached, {stats['failed']} failed"
    )
    return stats


def get_collection(repo_url: str) -> Optional[chromadb.Collection]:
    """Return the ChromaDB collection for a repo, or None if not indexed."""
    client = _get_chroma_client()
    col_name = _collection_name(repo_url)
    try:
        return client.get_collection(col_name)
    except Exception:
        return None


def delete_collection(repo_url: str) -> bool:
    """Delete all embeddings for a repo. Returns True if deleted."""
    client = _get_chroma_client()
    col_name = _collection_name(repo_url)
    try:
        client.delete_collection(col_name)
        logger.info(f"Deleted collection '{col_name}'")
        return True
    except Exception:
        return False


def list_indexed_repos() -> list[dict]:
    """Return a list of all indexed repos with their collection stats."""
    client = _get_chroma_client()
    cfg = get_settings()
    result = []
    for col in client.list_collections():
        if col.name.startswith(cfg.chroma_collection_prefix):
            count = col.count()
            result.append({"collection": col.name, "chunk_count": count})
    return result
