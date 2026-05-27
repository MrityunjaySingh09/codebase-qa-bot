"""
retrieval/vector_search.py — Dense vector search via ChromaDB.

Given a natural-language query, embed it with the same Ollama model
used during indexing, then retrieve the top-K most similar chunks
using cosine similarity.

Returns a list of SearchResult objects so downstream code (hybrid.py,
reranker.py) never has to touch ChromaDB directly.
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from config import get_settings
from ingestion.embedder import _get_chroma_client, _get_embed_model, _collection_name


# ---------------------------------------------------------------------------
# Shared result type (used by all search modules + reranker)
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    chunk_id: str
    content: str
    file_path: str
    language: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    repo_url: str
    score: float          # higher = more relevant (normalised 0-1)
    source: str           # "vector" | "bm25" | "reranked"

    def display_location(self) -> str:
        """Human-readable source location for UI display."""
        loc = self.file_path
        if self.symbol_name:
            loc += f" → {self.symbol_name}"
        loc += f" (lines {self.start_line}–{self.end_line})"
        return loc


def vector_search(
    query: str,
    repo_url: str,
    top_k: Optional[int] = None,
) -> list[SearchResult]:
    """
    Embed the query and retrieve the top-K closest chunks from ChromaDB.

    Args:
        query:   Natural language question.
        repo_url: Which repo's collection to search.
        top_k:   Number of results. Defaults to settings.vector_top_k.

    Returns:
        List of SearchResult sorted by descending similarity score.
    """
    cfg = get_settings()
    k = top_k or cfg.vector_top_k

    col_name = _collection_name(repo_url)
    client = _get_chroma_client()

    try:
        collection = client.get_collection(col_name)
    except Exception:
        logger.warning(f"No collection found for {repo_url} — index it first.")
        return []

    # Embed query with the same model used at index time
    embed_model = _get_embed_model()
    try:
        query_embedding = embed_model.embed_query(query)
    except Exception as exc:
        logger.error(f"Failed to embed query: {exc}")
        return []

    # Query ChromaDB
    try:
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error(f"ChromaDB query failed: {exc}")
        return []

    results: list[SearchResult] = []
    ids       = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]   # cosine distance: 0=identical, 2=opposite

    for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
        # Convert cosine distance → similarity score (0-1)
        score = 1.0 - (dist / 2.0)
        results.append(SearchResult(
            chunk_id    = cid,
            content     = doc,
            file_path   = meta.get("file_path", ""),
            language    = meta.get("language", ""),
            symbol_name = meta.get("symbol_name", ""),
            symbol_type = meta.get("symbol_type", ""),
            start_line  = int(meta.get("start_line", 0)),
            end_line    = int(meta.get("end_line", 0)),
            repo_url    = meta.get("repo_url", repo_url),
            score       = score,
            source      = "vector",
        ))

    logger.debug(f"Vector search: {len(results)} results for query '{query[:60]}'")
    return results
