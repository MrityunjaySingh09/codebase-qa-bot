"""
retrieval/bm25_search.py — Sparse BM25 keyword search over indexed chunks.

Why BM25 alongside vector search?
  Vector search excels at semantic similarity ("how does auth work?") but
  can miss exact-match queries ("where is JWT_SECRET defined?"). BM25 is
  the opposite — great for exact tokens, weak on paraphrase. Combining
  both (see hybrid.py) gives much better recall than either alone.

Implementation:
  - We load all chunk texts from ChromaDB at search time and build a
    BM25 index in memory. This is fast enough for repos up to ~50k chunks.
  - The index is cached per (repo_url, collection_count) so repeated
    queries on the same repo don't rebuild it.
  - Tokenisation: lowercase split on non-alphanumeric characters, with
    camelCase splitting for code identifiers.
"""

import re
from functools import lru_cache
from typing import Optional

from loguru import logger
from rank_bm25 import BM25Okapi

from config import get_settings
from ingestion.embedder import _get_chroma_client, _collection_name
from retrieval.vector_search import SearchResult


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _tokenise(text: str) -> list[str]:
    """
    Code-aware tokeniser:
      1. Split camelCase → individual words
      2. Lowercase everything
      3. Split on non-alphanumeric chars
      4. Drop tokens shorter than 2 chars
    """
    # Split camelCase
    text = _CAMEL_RE.sub(" ", text)
    # Lowercase and split on non-alnum
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# BM25 index cache
# ---------------------------------------------------------------------------

# Cache key: (collection_name, doc_count)
# We invalidate when doc_count changes (new chunks indexed)
_bm25_cache: dict[tuple[str, int], tuple[BM25Okapi, list[dict]]] = {}


def _get_bm25_index(
    collection_name: str,
) -> tuple[BM25Okapi, list[dict]] | tuple[None, None]:
    """
    Build (or return cached) BM25Okapi index for a ChromaDB collection.

    Returns (bm25_index, chunk_records) or (None, None) on error.
    Each chunk_record is a dict with content + metadata fields.
    """
    client = _get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
    except Exception:
        return None, None

    cache_key = (collection_name, count)
    if cache_key in _bm25_cache:
        logger.debug(f"BM25 cache hit for '{collection_name}' ({count} docs)")
        return _bm25_cache[cache_key]

    logger.info(f"Building BM25 index for '{collection_name}' ({count} docs) ...")

    # Fetch ALL documents from ChromaDB (paginated for large collections)
    PAGE = 5000
    all_docs: list[dict] = []
    offset = 0
    while offset < count:
        batch = collection.get(
            limit=PAGE,
            offset=offset,
            include=["documents", "metadatas"],
        )
        for doc, meta, cid in zip(
            batch["documents"], batch["metadatas"], batch["ids"]
        ):
            all_docs.append({"id": cid, "content": doc, **meta})
        offset += PAGE

    # Build tokenised corpus
    corpus = [_tokenise(d["content"]) for d in all_docs]
    index = BM25Okapi(corpus)

    _bm25_cache[cache_key] = (index, all_docs)
    logger.info(f"BM25 index built: {len(all_docs)} documents")
    return index, all_docs


def invalidate_bm25_cache(repo_url: str) -> None:
    """Call this after re-indexing a repo to force BM25 rebuild."""
    col_name = _collection_name(repo_url)
    keys_to_remove = [k for k in _bm25_cache if k[0] == col_name]
    for k in keys_to_remove:
        del _bm25_cache[k]
    logger.debug(f"BM25 cache invalidated for '{col_name}'")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bm25_search(
    query: str,
    repo_url: str,
    top_k: Optional[int] = None,
) -> list[SearchResult]:
    """
    BM25 keyword search over all indexed chunks for a repo.

    Args:
        query:    Natural language or code-snippet query.
        repo_url: Which repo to search.
        top_k:    Number of results. Defaults to settings.bm25_top_k.

    Returns:
        List of SearchResult sorted by descending BM25 score (normalised 0-1).
    """
    cfg = get_settings()
    k = top_k or cfg.bm25_top_k

    col_name = _collection_name(repo_url)
    index, all_docs = _get_bm25_index(col_name)

    if index is None or not all_docs:
        logger.warning(f"BM25: no index available for {repo_url}")
        return []

    query_tokens = _tokenise(query)
    if not query_tokens:
        return []

    scores = index.get_scores(query_tokens)       # numpy array, one score per doc

    # Get top-k indices
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:k]

    # Normalise scores to 0-1 range
    max_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 1.0
    if max_score == 0:
        return []

    results: list[SearchResult] = []
    for idx in top_indices:
        raw_score = float(scores[idx])
        if raw_score <= 0:
            break   # BM25 scores are 0 for non-matching docs
        doc = all_docs[int(idx)]
        results.append(SearchResult(
            chunk_id    = doc.get("id", ""),
            content     = doc.get("content", ""),
            file_path   = doc.get("file_path", ""),
            language    = doc.get("language", ""),
            symbol_name = doc.get("symbol_name", ""),
            symbol_type = doc.get("symbol_type", ""),
            start_line  = int(doc.get("start_line", 0)),
            end_line    = int(doc.get("end_line", 0)),
            repo_url    = doc.get("repo_url", repo_url),
            score       = raw_score / max_score,   # normalise
            source      = "bm25",
        ))

    logger.debug(f"BM25 search: {len(results)} results for '{query[:60]}'")
    return results
