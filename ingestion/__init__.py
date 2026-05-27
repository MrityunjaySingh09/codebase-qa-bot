"""
ingestion/ — GitHub repo ingestion pipeline.

Public interface:
    from ingestion import ingest_repo
"""

from ingestion.cloner import RepoInfo, clone_repo, delete_repo_cache
from ingestion.embedder import (
    delete_collection,
    embed_and_store,
    get_collection,
    list_indexed_repos,
)
from ingestion.parser import CodeChunk, get_repo_summary, parse_repo


def ingest_repo(
    url: str,
    force_reclone: bool = False,
    progress_callback=None,
) -> dict:
    """
    Full ingestion pipeline: clone → parse → embed → store.

    Args:
        url:               GitHub repo URL.
        force_reclone:     Re-clone even if cached locally.
        progress_callback: fn(stage: str, message: str, pct: float)
                           where pct is 0.0–1.0.

    Returns:
        {
            "repo_info":   RepoInfo,
            "summary":     {total_files, languages},
            "embed_stats": {total, new, skipped, failed, collection_name},
        }
    """

    def _cb(stage: str, msg: str, pct: float):
        if progress_callback:
            progress_callback(stage, msg, pct)

    # ── 1. Clone ──────────────────────────────────────────────
    if force_reclone:
        try:
            delete_collection(url)
        except Exception:
            pass

    _cb("clone", f"Cloning {url} ...", 0.0)
    repo_info = clone_repo(
        url,
        progress_callback=lambda m: _cb("clone", m, 0.1),
        force_reclone=force_reclone,
    )

    # ── 2. Summarise ─────────────────────────────────────────
    summary = get_repo_summary(repo_info.local_path)
    _cb("parse", f"Found {summary['total_files']} files", 0.15)

    # ── 3. Parse ──────────────────────────────────────────────
    all_chunks: list[CodeChunk] = []
    total_files = summary["total_files"]

    def parse_progress(msg: str, current: int, total: int):
        pct = 0.15 + (current / max(total, 1)) * 0.35
        _cb("parse", msg, pct)

    for chunk in parse_repo(
        repo_info.local_path,
        repo_info.url,
        progress_callback=parse_progress,
    ):
        all_chunks.append(chunk)

    _cb("embed", f"Parsed {len(all_chunks)} chunks — starting embedding ...", 0.50)

    # ── 4. Embed & Store ──────────────────────────────────────
    def embed_progress(msg: str, current: int, total: int):
        pct = 0.50 + (current / max(total, 1)) * 0.48
        _cb("embed", msg, pct)

    embed_stats = embed_and_store(
        all_chunks,
        repo_info.url,
        progress_callback=embed_progress,
        batch_size=32,
    )

    _cb("done", "✅ Indexing complete!", 1.0)

    return {
        "repo_info": repo_info,
        "summary": summary,
        "embed_stats": embed_stats,
    }


__all__ = [
    "ingest_repo",
    "clone_repo",
    "parse_repo",
    "embed_and_store",
    "get_collection",
    "delete_collection",
    "list_indexed_repos",
    "RepoInfo",
    "CodeChunk",
    "get_repo_summary",
    "delete_repo_cache",
]
