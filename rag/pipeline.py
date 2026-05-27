"""
rag/pipeline.py — Full RAG pipeline: query → retrieve → generate.

Pipeline flow:
  1. (Optional) Condense follow-up questions using chat history
  2. Hybrid retrieval → reranking → top-K chunks
  3. Format retrieved chunks into a context block with citations
  4. Stream response from Ollama LLM
  5. Return answer + source citations

Features:
  - Streaming via LangChain's stream() API — tokens arrive in real time
  - LangSmith tracing — every step logged if LANGCHAIN_TRACING_V2=true
  - Chat history support — multi-turn Q&A that remembers context
  - Source attribution — every answer includes file + line citations
  - Repo summary generation — called once after indexing
"""

import os
from typing import Generator, Iterator, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from loguru import logger

from config import get_settings
from rag.prompts import CONDENSE_PROMPT, QA_PROMPT, SUMMARY_PROMPT
from retrieval import retrieve
from retrieval.vector_search import SearchResult


# ---------------------------------------------------------------------------
# LangSmith tracing setup
# ---------------------------------------------------------------------------

def _setup_tracing() -> None:
    """Enable LangSmith tracing if API key is configured."""
    cfg = get_settings()
    if cfg.langchain_tracing_v2 and cfg.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = cfg.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = cfg.langchain_project
        logger.info(f"LangSmith tracing enabled → project: {cfg.langchain_project}")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


_setup_tracing()


# ---------------------------------------------------------------------------
# LLM (lazy singleton)
# ---------------------------------------------------------------------------

_llm = None


def _get_llm(streaming: bool = True):
    global _llm
    if _llm is None:
        cfg = get_settings()
        if cfg.groq_api_key:
            _llm = ChatGroq(
                model=cfg.groq_llm_model,
                groq_api_key=cfg.groq_api_key,
                temperature=0.1,
            )
            logger.info(f"LLM: Groq ({cfg.groq_llm_model})")
        else:
            _llm = ChatOllama(
                model=cfg.ollama_llm_model,
                base_url=cfg.ollama_base_url,
                temperature=0.1,       # low temp for factual code Q&A
                num_ctx=8192,          # context window
            )
            logger.info(f"LLM: {cfg.ollama_llm_model} @ {cfg.ollama_base_url}")
    return _llm


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def _format_context(results: list[SearchResult]) -> str:
    """
    Format retrieved chunks into a structured context block.

    Each chunk is presented with:
      - A numbered citation label [1], [2], ...
      - File path + symbol name + line range
      - The actual code content

    The citation numbers are referenced in the LLM's answer and
    matched back to SearchResult objects in the caller.
    """
    if not results:
        return "No relevant code found."

    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        header = (
            f"[{i}] File: {r.file_path}"
            + (f" | Symbol: {r.symbol_name}" if r.symbol_name else "")
            + f" | Lines: {r.start_line}-{r.end_line}"
            + f" | Language: {r.language}"
        )
        parts.append(f"{header}\n```{r.language}\n{r.content}\n```")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Question condensation (multi-turn)
# ---------------------------------------------------------------------------

def _condense_question(
    question: str,
    chat_history: list[tuple[str, str]],
) -> str:
    """
    If there's chat history, rephrase the follow-up question as standalone.
    Returns the original question unchanged if no history.
    """
    if not chat_history:
        return question

    llm = _get_llm(streaming=False)
    chain = CONDENSE_PROMPT | llm | StrOutputParser()

    history_str = "\n".join(
        f"Human: {h}\nAssistant: {a}" for h, a in chat_history
    )
    try:
        condensed = chain.invoke({
            "chat_history": history_str,
            "question": question,
        })
        logger.debug(f"Condensed question: {condensed!r}")
        return condensed.strip()
    except Exception as exc:
        logger.warning(f"Question condensation failed: {exc} — using original")
        return question


# ---------------------------------------------------------------------------
# Chat history formatting
# ---------------------------------------------------------------------------

def _build_chat_history_messages(
    chat_history: list[tuple[str, str]],
) -> list:
    """Convert (human, ai) tuples to LangChain message objects."""
    messages = []
    for human, ai in chat_history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    return messages


# ---------------------------------------------------------------------------
# Main RAG answer function — streaming
# ---------------------------------------------------------------------------

def stream_answer(
    question: str,
    repo_url: str,
    chat_history: Optional[list[tuple[str, str]]] = None,
    top_k: Optional[int] = None,
) -> Generator[dict, None, None]:
    """
    Run the full RAG pipeline and stream the response token by token.

    Yields dicts of shape:
        {"type": "retrieval", "results": list[SearchResult]}  # once, before generation
        {"type": "token",     "content": str}                 # one per token
        {"type": "done",      "answer": str, "sources": list[SearchResult]}

    This design lets the Streamlit UI:
      1. Show retrieved sources immediately (before generation starts)
      2. Stream tokens to a chat bubble in real time
      3. Render the final source citations after the answer

    Args:
        question:     User's natural language question.
        repo_url:     Which indexed repo to query.
        chat_history: List of (human, assistant) string tuples.
        top_k:        Override number of chunks to retrieve.
    """
    cfg = get_settings()
    history = chat_history or []

    # ── 1. Condense question if follow-up ─────────────────────────────────
    standalone_q = _condense_question(question, history)

    # ── 2. Retrieve ────────────────────────────────────────────────────────
    logger.info(f"Retrieving for: {standalone_q!r}")
    try:
        results = retrieve(standalone_q, repo_url, top_k=top_k or cfg.rerank_top_k)
    except Exception as exc:
        logger.error(f"Retrieval failed: {exc}")
        results = []

    # Yield retrieval results so UI can show them immediately
    yield {"type": "retrieval", "results": results}

    # ── 3. Format context ─────────────────────────────────────────────────
    context = _format_context(results)

    # ── 4. Build prompt ───────────────────────────────────────────────────
    prompt_input = {
        "context": context,
        "question": standalone_q,
        "chat_history": _build_chat_history_messages(history),
    }

    # ── 5. Stream LLM response ────────────────────────────────────────────
    llm = _get_llm(streaming=True)
    chain = QA_PROMPT | llm | StrOutputParser()

    full_answer = ""
    try:
        for token in chain.stream(prompt_input):
            full_answer += token
            yield {"type": "token", "content": token}
    except Exception as exc:
        error_msg = f"\n\n⚠️ Generation error: {exc}"
        full_answer += error_msg
        yield {"type": "token", "content": error_msg}
        logger.error(f"LLM streaming failed: {exc}")

    # ── 6. Done ───────────────────────────────────────────────────────────
    yield {"type": "done", "answer": full_answer, "sources": results}


# ---------------------------------------------------------------------------
# Non-streaming variant (for eval / testing)
# ---------------------------------------------------------------------------

def answer(
    question: str,
    repo_url: str,
    chat_history: Optional[list[tuple[str, str]]] = None,
    top_k: Optional[int] = None,
) -> dict:
    """
    Non-streaming RAG answer. Returns full result dict.

    Returns:
        {
            "question":   str,
            "answer":     str,
            "sources":    list[SearchResult],
            "context":    str,   # formatted context passed to LLM
        }
    """
    cfg = get_settings()
    history = chat_history or []

    standalone_q = _condense_question(question, history)
    results = retrieve(standalone_q, repo_url, top_k=top_k or cfg.rerank_top_k)
    context = _format_context(results)

    llm = _get_llm(streaming=False)
    chain = QA_PROMPT | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": standalone_q,
        "chat_history": _build_chat_history_messages(history),
    })

    return {
        "question": question,
        "answer":   response,
        "sources":  results,
        "context":  context,
    }


# ---------------------------------------------------------------------------
# Repo summary generation
# ---------------------------------------------------------------------------

def generate_repo_summary(
    repo_url: str,
    file_tree: str,
    sample_chunks: Optional[list[SearchResult]] = None,
) -> str:
    """
    Generate a high-level technical summary of the repo.
    Called once after indexing completes.

    Args:
        repo_url:      Canonical GitHub URL (for logging).
        file_tree:     String representation of the repo directory tree.
        sample_chunks: A handful of representative code chunks to give
                       the LLM a taste of the codebase style.

    Returns:
        Markdown-formatted repo summary string.
    """
    code_samples = ""
    if sample_chunks:
        samples = sample_chunks[:5]   # cap at 5 to stay within context
        code_samples = _format_context(samples)

    llm = _get_llm(streaming=False)
    chain = SUMMARY_PROMPT | llm | StrOutputParser()

    try:
        summary = chain.invoke({
            "file_tree": file_tree,
            "code_samples": code_samples or "No samples available.",
        })
        logger.info(f"Repo summary generated for {repo_url}")
        return summary
    except Exception as exc:
        logger.error(f"Summary generation failed: {exc}")
        return f"Could not generate summary: {exc}"


# ---------------------------------------------------------------------------
# File tree helper (used by generate_repo_summary)
# ---------------------------------------------------------------------------

def build_file_tree(repo_root, max_depth: int = 3) -> str:
    """
    Generate a compact text file tree for the repo summary prompt.

    Example output:
        src/
          auth/
            service.py
            models.py
          api/
            routes.py
        tests/
          test_auth.py
    """
    from pathlib import Path
    from ingestion.parser import SKIP_DIRS

    root = Path(repo_root)
    lines: list[str] = []

    def _walk(path: Path, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return
        for entry in entries:
            if entry.name.startswith(".") or entry.name in SKIP_DIRS:
                continue
            indent = "  " * depth
            if entry.is_dir():
                lines.append(f"{indent}{entry.name}/")
                _walk(entry, depth + 1, indent)
            else:
                lines.append(f"{indent}{entry.name}")

    _walk(root, 0, "")
    return "\n".join(lines[:200])   # cap output length
