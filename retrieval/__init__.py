"""
retrieval/ — Hybrid search + reranking pipeline.

Public interface:
    from retrieval import retrieve

    results = retrieve("How does auth work?", repo_url="https://github.com/x/y")
"""

from retrieval.bm25_search import bm25_search, invalidate_bm25_cache
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from retrieval.vector_search import SearchResult, vector_search


def retrieve(
    query: str,
    repo_url: str,
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Full retrieval pipeline: hybrid search → rerank → top_k.
    """
    from config import get_settings
    cfg = get_settings()

    candidates = hybrid_search(query, repo_url)
    results = rerank(query, candidates, top_k=top_k or cfg.rerank_top_k)
    return results


__all__ = [
    "retrieve",
    "hybrid_search",
    "vector_search",
    "bm25_search",
    "rerank",
    "invalidate_bm25_cache",
    "SearchResult",
]
