"""
retrieval/hybrid.py — Combine vector + BM25 results via Reciprocal Rank Fusion.

Why RRF?
  Both vector and BM25 searches return ranked lists with incompatible
  score scales. RRF is a score-free fusion algorithm that only uses
  rank positions — it's robust, parameter-light, and consistently
  outperforms simple score averaging in information retrieval benchmarks.

RRF formula:
  RRF(d) = Σ  1 / (k + rank_i(d))
           i
  where k=60 (standard default that dampens the impact of top ranks).

  A document appearing at rank 1 in both lists gets:
    1/(60+1) + 1/(60+1) ≈ 0.0328
  A document at rank 1 in one and rank 20 in another:
    1/(60+1) + 1/(60+20) ≈ 0.0289

  Documents not present in a list are simply not counted.
"""

from collections import defaultdict
from typing import Optional

from loguru import logger

from config import get_settings
from retrieval.bm25_search import bm25_search
from retrieval.vector_search import SearchResult, vector_search

# Standard RRF damping constant — do not change without benchmarking
_RRF_K = 60


def _reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = _RRF_K,
) -> list[SearchResult]:
    """
    Merge multiple ranked result lists into a single fused ranking.

    Args:
        result_lists: Each inner list is a ranked result list from one source.
        k:            RRF damping constant (default 60).

    Returns:
        Merged list sorted by descending RRF score, with source="hybrid".
    """
    # Map chunk_id → (rrf_score, best SearchResult object)
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_store: dict[str, SearchResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            cid = result.chunk_id
            rrf_scores[cid] += 1.0 / (k + rank)
            # Keep the result object (prefer vector result if both exist)
            if cid not in chunk_store or result.source == "vector":
                chunk_store[cid] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    # Normalise RRF scores to 0-1
    max_rrf = rrf_scores[sorted_ids[0]] if sorted_ids else 1.0

    fused: list[SearchResult] = []
    for cid in sorted_ids:
        result = chunk_store[cid]
        fused.append(SearchResult(
            chunk_id    = result.chunk_id,
            content     = result.content,
            file_path   = result.file_path,
            language    = result.language,
            symbol_name = result.symbol_name,
            symbol_type = result.symbol_type,
            start_line  = result.start_line,
            end_line    = result.end_line,
            repo_url    = result.repo_url,
            score       = rrf_scores[cid] / max_rrf,
            source      = "hybrid",
        ))
    return fused


def hybrid_search(
    query: str,
    repo_url: str,
    top_k: Optional[int] = None,
    vector_weight: float = 0.5,   # reserved for future weighted RRF variant
    bm25_weight: float = 0.5,
) -> list[SearchResult]:
    """
    Run vector + BM25 search in parallel, fuse with RRF, return top_k.

    Args:
        query:         Natural language question.
        repo_url:      Which repo to search.
        top_k:         Final number of results to return (before reranking).
                       Defaults to max(vector_top_k, bm25_top_k) from settings.
        vector_weight: Currently unused — reserved for weighted RRF extension.
        bm25_weight:   Currently unused — reserved for weighted RRF extension.

    Returns:
        Top-K fused SearchResult list, sorted by descending RRF score.
    """
    cfg = get_settings()
    # Retrieve more candidates than needed — reranker will prune to rerank_top_k
    candidate_k = max(cfg.vector_top_k, cfg.bm25_top_k)

    logger.info(f"Hybrid search | query='{query[:60]}' | repo={repo_url}")

    # Run both searches
    vec_results = vector_search(query, repo_url, top_k=cfg.vector_top_k)
    bm25_results = bm25_search(query, repo_url, top_k=cfg.bm25_top_k)

    logger.debug(
        f"  Vector: {len(vec_results)} results | BM25: {len(bm25_results)} results"
    )

    if not vec_results and not bm25_results:
        logger.warning("Both vector and BM25 returned no results.")
        return []

    # Fuse
    fused = _reciprocal_rank_fusion([vec_results, bm25_results])

    # Trim to requested top_k
    final_k = top_k or candidate_k
    results = fused[:final_k]

    logger.info(f"Hybrid RRF: {len(fused)} unique chunks → returning top {len(results)}")
    return results
