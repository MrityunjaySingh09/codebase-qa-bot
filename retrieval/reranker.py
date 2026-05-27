"""
retrieval/reranker.py — Cross-encoder reranking of hybrid search results.

Why rerank?
  Vector and BM25 retrieval both use "bi-encoder" style scoring — the
  query and document are encoded independently. This is fast but imprecise.
  A cross-encoder processes (query, document) together, giving much more
  accurate relevance scores, at the cost of being slower (O(k) forward
  passes vs O(1) for vector search).

  We only rerank the top-K candidates from hybrid search (default 20),
  making the cross-encoder fast enough for interactive use.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M params, ~50ms per (query, doc) pair on CPU
  - Trained on MS MARCO passage ranking — generalises well to code Q&A
  - Completely free, runs locally via HuggingFace

Fallback:
  If sentence-transformers is not installed (e.g. CI environment),
  the reranker transparently falls back to returning results sorted
  by their hybrid RRF score. This keeps the pipeline functional.
"""

from typing import Optional

from loguru import logger

from config import get_settings
from retrieval.vector_search import SearchResult

_RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_rerank_model = None   # lazy-loaded on first use


def _load_rerank_model():
    """Lazy-load the cross-encoder. Returns None if unavailable."""
    global _rerank_model
    if _rerank_model is not None:
        return _rerank_model
    try:
        from sentence_transformers import CrossEncoder
        _rerank_model = CrossEncoder(_RERANK_MODEL_NAME)
        logger.info(f"Cross-encoder loaded: {_RERANK_MODEL_NAME}")
        return _rerank_model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — reranker will use RRF scores. "
            "Install with: pip install sentence-transformers"
        )
        return None
    except Exception as exc:
        logger.error(f"Failed to load cross-encoder: {exc}")
        return None


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: Optional[int] = None,
) -> list[SearchResult]:
    """
    Rerank a list of SearchResults using a cross-encoder.

    Args:
        query:   The original user query.
        results: Candidates from hybrid_search (or any retriever).
        top_k:   How many to return after reranking.
                 Defaults to settings.rerank_top_k.

    Returns:
        Top-K SearchResult list, sorted by descending cross-encoder score,
        with source="reranked".
    """
    cfg = get_settings()
    k = top_k or cfg.rerank_top_k

    if not results:
        return []

    model = _load_rerank_model()

    if model is None:
        # Graceful fallback: return top-k by existing RRF score
        logger.debug("Reranker unavailable — falling back to RRF scores")
        fallback = sorted(results, key=lambda r: r.score, reverse=True)[:k]
        for r in fallback:
            r.source = "reranked"
        return fallback

    # Build (query, passage) pairs for the cross-encoder
    # We prepend file path + symbol name so the model has full context
    pairs = [
        (
            query,
            f"# {r.file_path} | {r.symbol_name}\n\n{r.content}"
        )
        for r in results
    ]

    logger.debug(f"Cross-encoder scoring {len(pairs)} pairs ...")
    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as exc:
        logger.error(f"Cross-encoder prediction failed: {exc}")
        # Fall back to RRF scores
        fallback = sorted(results, key=lambda r: r.score, reverse=True)[:k]
        for r in fallback:
            r.source = "reranked"
        return fallback

    # Attach cross-encoder scores and sort
    import numpy as np
    scored = list(zip(results, scores.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]

    # Normalise scores to 0-1 using sigmoid
    def _sigmoid(x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-x))

    reranked: list[SearchResult] = []
    for result, raw_score in top:
        reranked.append(SearchResult(
            chunk_id    = result.chunk_id,
            content     = result.content,
            file_path   = result.file_path,
            language    = result.language,
            symbol_name = result.symbol_name,
            symbol_type = result.symbol_type,
            start_line  = result.start_line,
            end_line    = result.end_line,
            repo_url    = result.repo_url,
            score       = _sigmoid(raw_score),
            source      = "reranked",
        ))

    logger.info(
        f"Reranking complete: {len(results)} → {len(reranked)} results | "
        f"top score: {reranked[0].score:.3f}"
    )
    return reranked
