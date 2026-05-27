"""
rag/ — RAG pipeline.

Public interface:
    from rag import stream_answer, answer, generate_repo_summary
"""

from rag.pipeline import (
    answer,
    build_file_tree,
    generate_repo_summary,
    stream_answer,
)

__all__ = ["stream_answer", "answer", "generate_repo_summary", "build_file_tree"]
