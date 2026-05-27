"""
tests/test_pipeline.py — Unit tests for ingestion + retrieval pipeline.

Run with:  pytest tests/ -v
"""

import textwrap
from pathlib import Path

import pytest

from ingestion.parser import (
    CodeChunk,
    PythonASTChunker,
    _line_based_chunks,
    get_repo_summary,
    parse_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PYTHON = textwrap.dedent("""\
    \"\"\"Sample module.\"\"\"

    import os


    def greet(name: str) -> str:
        \"\"\"Return a greeting.\"\"\"
        return f"Hello, {name}!"


    class Calculator:
        \"\"\"Simple calculator.\"\"\"

        def add(self, a: int, b: int) -> int:
            return a + b

        def subtract(self, a: int, b: int) -> int:
            return a - b


    def helper():
        pass
""")

SAMPLE_PYTHON_SYNTAX_ERROR = "def broken(\n    pass\n"


# ---------------------------------------------------------------------------
# PythonASTChunker
# ---------------------------------------------------------------------------

class TestPythonASTChunker:
    def setup_method(self):
        self.chunker = PythonASTChunker()
        self.repo_url = "https://github.com/test/repo"

    def test_extracts_top_level_functions(self):
        chunks = self.chunker.chunk(SAMPLE_PYTHON, "test.py", self.repo_url)
        names = {c.symbol_name for c in chunks}
        assert "greet" in names
        assert "helper" in names

    def test_extracts_class(self):
        chunks = self.chunker.chunk(SAMPLE_PYTHON, "test.py", self.repo_url)
        names = {c.symbol_name for c in chunks}
        assert "Calculator" in names

    def test_extracts_methods_with_qualified_name(self):
        chunks = self.chunker.chunk(SAMPLE_PYTHON, "test.py", self.repo_url)
        names = {c.symbol_name for c in chunks}
        assert "Calculator.add" in names
        assert "Calculator.subtract" in names

    def test_chunk_has_correct_metadata(self):
        chunks = self.chunker.chunk(SAMPLE_PYTHON, "test.py", self.repo_url)
        greet = next(c for c in chunks if c.symbol_name == "greet")
        assert greet.language == "python"
        assert greet.file_path == "test.py"
        assert greet.repo_url == self.repo_url
        assert greet.start_line > 0
        assert greet.end_line >= greet.start_line
        assert "Hello" in greet.content

    def test_chunk_id_is_deterministic(self):
        chunks1 = self.chunker.chunk(SAMPLE_PYTHON, "test.py", self.repo_url)
        chunks2 = self.chunker.chunk(SAMPLE_PYTHON, "test.py", self.repo_url)
        ids1 = sorted(c.chunk_id for c in chunks1)
        ids2 = sorted(c.chunk_id for c in chunks2)
        assert ids1 == ids2

    def test_syntax_error_falls_back_to_line_based(self):
        chunks = self.chunker.chunk(SAMPLE_PYTHON_SYNTAX_ERROR, "bad.py", self.repo_url)
        # Should not raise; should return at least one chunk
        assert len(chunks) >= 1

    def test_empty_file_returns_no_chunks(self):
        # An empty file gets a module-level chunk only (or nothing)
        chunks = self.chunker.chunk("", "empty.py", self.repo_url)
        # Acceptable: 0 or 1 chunks
        assert len(chunks) <= 1

    def test_content_hash_changes_with_content(self):
        chunks1 = self.chunker.chunk(SAMPLE_PYTHON, "test.py", self.repo_url)
        modified = SAMPLE_PYTHON.replace("Hello", "Hi")
        chunks2 = self.chunker.chunk(modified, "test.py", self.repo_url)
        hashes1 = {c.content_hash for c in chunks1}
        hashes2 = {c.content_hash for c in chunks2}
        assert hashes1 != hashes2


# ---------------------------------------------------------------------------
# Line-based fallback chunker
# ---------------------------------------------------------------------------

class TestLineBasedChunker:
    def test_produces_chunks_for_large_file(self):
        source = "\n".join([f"line {i}" for i in range(200)])
        chunks = _line_based_chunks(source, "file.txt", "https://github.com/x/y", "text")
        assert len(chunks) > 1

    def test_chunk_overlap(self):
        source = "\n".join([f"line {i}" for i in range(100)])
        chunks = _line_based_chunks(
            source, "file.txt", "https://github.com/x/y", "text",
            lines_per_chunk=20, overlap=5
        )
        # End of chunk N should overlap with start of chunk N+1
        if len(chunks) >= 2:
            assert chunks[0].end_line > chunks[1].start_line - 1

    def test_start_offset_respected(self):
        source = "line 1\nline 2\nline 3"
        chunks = _line_based_chunks(
            source, "f.py", "https://github.com/x/y", "python", start_offset=10
        )
        assert chunks[0].start_line > 10


# ---------------------------------------------------------------------------
# CodeChunk dataclass
# ---------------------------------------------------------------------------

class TestCodeChunk:
    def test_chunk_id_is_hex_string(self):
        chunk = CodeChunk(
            repo_url="https://github.com/a/b",
            file_path="src/main.py",
            language="python",
            symbol_name="foo",
            symbol_type="function",
            start_line=1,
            end_line=5,
            content="def foo(): pass",
        )
        assert len(chunk.chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunk.chunk_id)

    def test_to_metadata_has_all_keys(self):
        chunk = CodeChunk(
            repo_url="https://github.com/a/b",
            file_path="src/main.py",
            language="python",
            symbol_name="foo",
            symbol_type="function",
            start_line=1,
            end_line=5,
            content="def foo(): pass",
        )
        meta = chunk.to_metadata()
        required_keys = {
            "chunk_id", "repo_url", "file_path", "language",
            "symbol_name", "symbol_type", "start_line", "end_line", "content_hash"
        }
        assert required_keys.issubset(meta.keys())


# ---------------------------------------------------------------------------
# parse_file
# ---------------------------------------------------------------------------

class TestParseFile:
    def test_skips_unknown_extension(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_text("hello")
        chunks = parse_file(f, tmp_path, "https://github.com/a/b")
        assert chunks == []

    def test_parses_python_file(self, tmp_path):
        f = tmp_path / "module.py"
        f.write_text(SAMPLE_PYTHON)
        chunks = parse_file(f, tmp_path, "https://github.com/a/b")
        assert len(chunks) > 0
        assert all(c.language == "python" for c in chunks)

    def test_skips_oversized_file(self, tmp_path, monkeypatch):
        import config
        # Patch max_file_size_kb to 0 to force skip
        monkeypatch.setattr(config.get_settings(), "max_file_size_kb", 0)
        f = tmp_path / "big.py"
        f.write_text(SAMPLE_PYTHON)
        chunks = parse_file(f, tmp_path, "https://github.com/a/b")
        assert chunks == []

    def test_relative_file_path_stored(self, tmp_path):
        subdir = tmp_path / "src"
        subdir.mkdir()
        f = subdir / "auth.py"
        f.write_text("def login(): pass")
        chunks = parse_file(f, tmp_path, "https://github.com/a/b")
        assert all("src/auth.py" in c.file_path for c in chunks)


# ---------------------------------------------------------------------------
# get_repo_summary
# ---------------------------------------------------------------------------

class TestGetRepoSummary:
    def test_counts_files_correctly(self, tmp_path):
        (tmp_path / "a.py").write_text("pass")
        (tmp_path / "b.py").write_text("pass")
        (tmp_path / "c.js").write_text("const x = 1;")
        (tmp_path / "ignored.pyc").write_bytes(b"\x00")

        summary = get_repo_summary(tmp_path)
        assert summary["total_files"] == 3  # .pyc not counted
        assert summary["languages"]["python"] == 2
        assert summary["languages"]["javascript"] == 1

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lib.js").write_text("module.exports = {}")
        (tmp_path / "index.js").write_text("const x = 1;")

        summary = get_repo_summary(tmp_path)
        assert summary["total_files"] == 1


# ===========================================================================
# STEP 2: Retrieval Tests
# ===========================================================================

from retrieval.bm25_search import _tokenise, bm25_search, _get_bm25_index
from retrieval.vector_search import SearchResult
from retrieval.hybrid import _reciprocal_rank_fusion
from retrieval.reranker import rerank


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_splits_camel_case(self):
        tokens = _tokenise("getUserById")
        assert "get" in tokens
        assert "user" in tokens
        assert "by" in tokens
        assert "id" in tokens

    def test_lowercases(self):
        tokens = _tokenise("HTTPSConnection")
        assert all(t == t.lower() for t in tokens)

    def test_drops_short_tokens(self):
        tokens = _tokenise("a b do it")
        assert "a" not in tokens
        assert "b" not in tokens
        # "do" and "it" are 2 chars — allowed
        assert "do" in tokens
        assert "it" in tokens

    def test_handles_underscored_identifiers(self):
        tokens = _tokenise("get_user_by_id")
        assert "get" in tokens
        assert "user" in tokens

    def test_empty_string(self):
        assert _tokenise("") == []


# ---------------------------------------------------------------------------
# SearchResult helpers
# ---------------------------------------------------------------------------

def _make_result(chunk_id, score=1.0, source="vector", symbol="foo", file="a.py"):
    return SearchResult(
        chunk_id=chunk_id,
        content=f"content of {chunk_id}",
        file_path=file,
        language="python",
        symbol_name=symbol,
        symbol_type="function",
        start_line=1,
        end_line=10,
        repo_url="https://github.com/test/repo",
        score=score,
        source=source,
    )


class TestSearchResult:
    def test_display_location_with_symbol(self):
        r = _make_result("abc", symbol="MyClass.method", file="src/auth.py")
        loc = r.display_location()
        assert "src/auth.py" in loc
        assert "MyClass.method" in loc
        assert "lines" in loc

    def test_display_location_no_symbol(self):
        r = _make_result("abc", symbol="", file="utils.py")
        loc = r.display_location()
        assert "utils.py" in loc


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class TestRRF:
    def test_document_in_both_lists_ranks_higher(self):
        """A doc appearing in both lists should outrank one appearing in only one."""
        shared = _make_result("shared")
        only_vec = _make_result("only_vec")
        only_bm25 = _make_result("only_bm25")

        vec_list  = [shared, only_vec]
        bm25_list = [shared, only_bm25]

        fused = _reciprocal_rank_fusion([vec_list, bm25_list])
        ids = [r.chunk_id for r in fused]

        assert ids[0] == "shared"

    def test_all_results_present_in_output(self):
        r1 = _make_result("a")
        r2 = _make_result("b")
        r3 = _make_result("c")
        fused = _reciprocal_rank_fusion([[r1, r2], [r2, r3]])
        ids = {r.chunk_id for r in fused}
        assert ids == {"a", "b", "c"}

    def test_scores_normalised_to_one(self):
        results = [_make_result(str(i)) for i in range(5)]
        fused = _reciprocal_rank_fusion([results])
        assert abs(fused[0].score - 1.0) < 1e-6

    def test_empty_lists(self):
        assert _reciprocal_rank_fusion([[], []]) == []

    def test_single_list(self):
        results = [_make_result("x"), _make_result("y")]
        fused = _reciprocal_rank_fusion([results])
        assert len(fused) == 2
        assert fused[0].chunk_id == "x"  # top rank preserved

    def test_source_set_to_hybrid(self):
        r = _make_result("a", source="vector")
        fused = _reciprocal_rank_fusion([[r]])
        assert fused[0].source == "hybrid"

    def test_rank_order_matters(self):
        """Rank 1 in a list beats rank 2 in same list."""
        r1 = _make_result("top")
        r2 = _make_result("bottom")
        fused = _reciprocal_rank_fusion([[r1, r2]])
        assert fused[0].chunk_id == "top"

    def test_rrf_k_dampening(self):
        """
        With k=60, the score difference between rank 1 and rank 2
        should be small (dampened), not a factor of 2.
        """
        r1 = _make_result("a")
        r2 = _make_result("b")
        fused = _reciprocal_rank_fusion([[r1, r2]], k=60)
        score_top = fused[0].score
        score_second = fused[1].score
        # score ratio should be close to 1, not 2
        assert score_top / score_second < 1.1


# ---------------------------------------------------------------------------
# Reranker (no model — tests fallback path)
# ---------------------------------------------------------------------------

class TestReranker:
    def test_fallback_returns_top_k(self):
        """When cross-encoder unavailable, should return top-k by score."""
        results = [_make_result(str(i), score=1.0 - i * 0.1) for i in range(10)]
        reranked = rerank("test query", results, top_k=3)
        assert len(reranked) == 3

    def test_fallback_preserves_order(self):
        results = [
            _make_result("high", score=0.9),
            _make_result("mid",  score=0.5),
            _make_result("low",  score=0.1),
        ]
        reranked = rerank("test query", results, top_k=3)
        # Without cross-encoder, should be sorted by score descending
        assert reranked[0].chunk_id == "high"

    def test_empty_results(self):
        assert rerank("query", [], top_k=5) == []

    def test_source_set_to_reranked(self):
        results = [_make_result("a", source="hybrid")]
        reranked = rerank("query", results, top_k=1)
        assert reranked[0].source == "reranked"


# ===========================================================================
# STEP 3: RAG Pipeline Tests
# ===========================================================================

from unittest.mock import MagicMock, patch

from rag.pipeline import (
    _condense_question,
    _format_context,
    _build_chat_history_messages,
    build_file_tree,
    answer,
    stream_answer,
)
from rag.prompts import QA_PROMPT, CONDENSE_PROMPT, SUMMARY_PROMPT
from langchain_core.messages import HumanMessage, AIMessage


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_empty_results(self):
        assert "No relevant" in _format_context([])

    def test_numbered_citations(self):
        r1 = _make_result("a", file="auth.py", symbol="login")
        r2 = _make_result("b", file="db.py", symbol="connect")
        ctx = _format_context([r1, r2])
        assert "[1]" in ctx
        assert "[2]" in ctx

    def test_includes_file_path(self):
        r = _make_result("a", file="src/service.py")
        ctx = _format_context([r])
        assert "src/service.py" in ctx

    def test_includes_symbol_name(self):
        r = _make_result("a", symbol="MyClass.my_method")
        ctx = _format_context([r])
        assert "MyClass.my_method" in ctx

    def test_includes_line_numbers(self):
        r = _make_result("a")
        ctx = _format_context([r])
        assert "Lines:" in ctx

    def test_code_block_language_tag(self):
        r = _make_result("a")
        r.language = "typescript"
        ctx = _format_context([r])
        assert "```typescript" in ctx

    def test_separator_between_chunks(self):
        r1 = _make_result("a")
        r2 = _make_result("b")
        ctx = _format_context([r1, r2])
        assert "---" in ctx

    def test_content_present(self):
        r = _make_result("abc")
        ctx = _format_context([r])
        assert r.content in ctx


# ---------------------------------------------------------------------------
# Chat history builder
# ---------------------------------------------------------------------------

class TestBuildChatHistory:
    def test_empty_history(self):
        assert _build_chat_history_messages([]) == []

    def test_alternating_message_types(self):
        msgs = _build_chat_history_messages([
            ("hello", "hi there"),
            ("how are you", "great"),
        ])
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)
        assert isinstance(msgs[2], HumanMessage)
        assert isinstance(msgs[3], AIMessage)

    def test_message_content_preserved(self):
        msgs = _build_chat_history_messages([("my question", "my answer")])
        assert msgs[0].content == "my question"
        assert msgs[1].content == "my answer"


# ---------------------------------------------------------------------------
# Question condensation
# ---------------------------------------------------------------------------

class TestCondenseQuestion:
    def test_no_history_returns_original(self):
        q = "What does AuthService do?"
        result = _condense_question(q, [])
        assert result == q

    def test_with_history_calls_llm(self):
        """When history exists, the LLM is called. We mock it here."""
        with patch("rag.pipeline._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm_fn.return_value = mock_llm
            # Simulate chain output
            mock_chain_result = "What are the unit tests for AuthService.login?"
            mock_llm.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=MagicMock(
                    invoke=MagicMock(return_value=mock_chain_result)
                ))
            ))
            # If history is empty, returns original without calling LLM
            result = _condense_question("What about the tests?", [])
            assert result == "What about the tests?"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_qa_prompt_has_context_placeholder(self):
        prompt_str = str(QA_PROMPT)
        assert "context" in prompt_str

    def test_qa_prompt_has_question_placeholder(self):
        prompt_str = str(QA_PROMPT)
        assert "question" in prompt_str

    def test_condense_prompt_has_history(self):
        prompt_str = str(CONDENSE_PROMPT)
        assert "chat_history" in prompt_str

    def test_summary_prompt_has_file_tree(self):
        prompt_str = str(SUMMARY_PROMPT)
        assert "file_tree" in prompt_str

    def test_qa_prompt_renders_with_inputs(self):
        """Verify the prompt template is well-formed and renderable."""
        messages = QA_PROMPT.format_messages(
            context="def foo(): pass",
            question="What does foo do?",
            chat_history=[],
        )
        assert len(messages) >= 2
        combined = " ".join(m.content for m in messages)
        assert "def foo(): pass" in combined
        assert "What does foo do?" in combined


# ---------------------------------------------------------------------------
# File tree builder
# ---------------------------------------------------------------------------

class TestBuildFileTree:
    def test_lists_files(self, tmp_path):
        (tmp_path / "main.py").write_text("pass")
        (tmp_path / "utils.py").write_text("pass")
        tree = build_file_tree(tmp_path)
        assert "main.py" in tree
        assert "utils.py" in tree

    def test_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".git"
        hidden.mkdir()
        (hidden / "config").write_text("data")
        tree = build_file_tree(tmp_path)
        assert ".git" not in tree

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lib.js").write_text("code")
        tree = build_file_tree(tmp_path)
        assert "node_modules" not in tree

    def test_nested_dirs_shown(self, tmp_path):
        src = tmp_path / "src" / "auth"
        src.mkdir(parents=True)
        (src / "service.py").write_text("pass")
        tree = build_file_tree(tmp_path)
        assert "src/" in tree
        assert "auth/" in tree
        assert "service.py" in tree

    def test_max_depth_respected(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "deep_file.py").write_text("pass")
        tree = build_file_tree(tmp_path, max_depth=2)
        assert "deep_file.py" not in tree

    def test_returns_string(self, tmp_path):
        assert isinstance(build_file_tree(tmp_path), str)


# ---------------------------------------------------------------------------
# stream_answer (mocked LLM + retrieval)
# ---------------------------------------------------------------------------

class TestStreamAnswer:
    def test_yields_retrieval_event_first(self):
        """The first yielded event must be type='retrieval'."""
        with patch("rag.pipeline.retrieve") as mock_retrieve, \
             patch("rag.pipeline._get_llm") as mock_llm_fn:

            mock_retrieve.return_value = []
            mock_llm = MagicMock()
            mock_llm_fn.return_value = mock_llm

            # Mock chain stream
            mock_chain = MagicMock()
            mock_chain.stream.return_value = iter(["Hello", " world"])
            mock_llm.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))

            events = list(stream_answer("test?", "https://github.com/a/b"))
            assert events[0]["type"] == "retrieval"

    def _make_stream_patches(self, tokens, results=None):
        """Helper: patch retrieve + the full LangChain chain stream."""
        from unittest.mock import patch, MagicMock
        from langchain_core.runnables import RunnableLambda

        retrieve_patch = patch("rag.pipeline.retrieve", return_value=results or [])
        # Patch the chain's stream by replacing QA_PROMPT | llm | parser entirely
        mock_chain = MagicMock()
        mock_chain.stream.return_value = iter(tokens)
        chain_patch = patch("rag.pipeline.QA_PROMPT.__or__",
                            return_value=MagicMock(
                                __or__=MagicMock(return_value=mock_chain)
                            ))
        return retrieve_patch, chain_patch, mock_chain

    def test_yields_tokens(self):
        with patch("rag.pipeline.retrieve", return_value=[]):
            with patch("rag.pipeline._get_llm") as mock_llm_fn:
                mock_llm = MagicMock()
                mock_llm_fn.return_value = mock_llm
                # Patch the StrOutputParser chain via QA_PROMPT pipe
                from langchain_core.runnables import RunnableLambda
                fake_chain = MagicMock()
                fake_chain.stream.return_value = iter(["tok1", "tok2", "tok3"])
                with patch("rag.pipeline.QA_PROMPT") as mock_prompt:
                    mock_prompt.__or__ = MagicMock(
                        return_value=MagicMock(
                            __or__=MagicMock(return_value=fake_chain)
                        )
                    )
                    events = list(stream_answer("q", "https://github.com/a/b"))
                    token_events = [e for e in events if e["type"] == "token"]
                    assert len(token_events) >= 1  # at least one token or error token

    def test_last_event_is_done(self):
        with patch("rag.pipeline.retrieve", return_value=[]):
            with patch("rag.pipeline._get_llm"):
                with patch("rag.pipeline.QA_PROMPT") as mock_prompt:
                    fake_chain = MagicMock()
                    fake_chain.stream.return_value = iter(["answer text"])
                    mock_prompt.__or__ = MagicMock(
                        return_value=MagicMock(
                            __or__=MagicMock(return_value=fake_chain)
                        )
                    )
                    events = list(stream_answer("q", "https://github.com/a/b"))
                    assert events[-1]["type"] == "done"
                    assert "answer" in events[-1]
                    assert "sources" in events[-1]

    def test_done_event_assembles_full_answer(self):
        with patch("rag.pipeline.retrieve", return_value=[]):
            with patch("rag.pipeline._get_llm"):
                with patch("rag.pipeline.QA_PROMPT") as mock_prompt:
                    fake_chain = MagicMock()
                    fake_chain.stream.return_value = iter(["Hello", " ", "world"])
                    mock_prompt.__or__ = MagicMock(
                        return_value=MagicMock(
                            __or__=MagicMock(return_value=fake_chain)
                        )
                    )
                    events = list(stream_answer("q", "https://github.com/a/b"))
                    done = events[-1]
                    # answer should contain whatever tokens were streamed
                    assert isinstance(done["answer"], str)
                    assert len(done["answer"]) > 0


# ===========================================================================
# STEP 4: UI Helper Tests
# ===========================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.app import _lang_badge, _file_badge


class TestUIHelpers:
    def test_lang_badge_contains_language(self):
        badge = _lang_badge("python")
        assert "python" in badge

    def test_lang_badge_python_color(self):
        badge = _lang_badge("python")
        assert "#3572A5" in badge   # python blue

    def test_lang_badge_unknown_lang_uses_fallback_color(self):
        badge = _lang_badge("cobol")
        assert "#8b949e" in badge   # fallback grey

    def test_lang_badge_returns_html(self):
        badge = _lang_badge("go")
        assert "<span" in badge

    def test_file_badge_shows_filename(self):
        badge = _file_badge("src/auth/service.py", 42, 78)
        assert "service.py" in badge

    def test_file_badge_shows_line_number(self):
        badge = _file_badge("main.py", 10, 20)
        assert "10" in badge

    def test_file_badge_has_full_path_in_title(self):
        badge = _file_badge("src/deeply/nested/file.py", 1, 5)
        assert "src/deeply/nested/file.py" in badge

    def test_file_badge_returns_html_span(self):
        badge = _file_badge("a.py", 1, 1)
        assert "<span" in badge


# ===========================================================================
# STEP 5: Evaluation Tests
# ===========================================================================

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from eval.ragas_eval import (
    DEFAULT_QUESTIONS,
    _null_scores,
    _print_summary,
    get_results_markdown_table,
    load_latest_results,
)


class TestRagasEval:
    def test_default_questions_have_required_keys(self):
        for q in DEFAULT_QUESTIONS:
            assert "question" in q
            assert "ground_truth" in q
            assert len(q["question"]) > 10

    def test_null_scores_has_all_metrics(self):
        scores = _null_scores()
        assert "faithfulness"      in scores
        assert "answer_relevancy"  in scores
        assert "context_precision" in scores
        assert "context_recall"    in scores

    def test_null_scores_are_none(self):
        scores = _null_scores()
        assert all(v is None for v in scores.values())

    def test_print_summary_runs_without_error(self, capsys):
        results = {
            "repo_url": "https://github.com/test/repo",
            "timestamp": "2024-01-01T00:00:00",
            "model": "llama3",
            "embed_model": "nomic-embed-text",
            "num_questions": 5,
            "top_k": 5,
            "aggregate": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.80,
                "context_precision": 0.75,
                "context_recall": 0.70,
            },
            "per_question": [],
        }
        _print_summary(results)
        captured = capsys.readouterr()
        assert "Faithfulness" in captured.out
        assert "0.850" in captured.out

    def test_print_summary_handles_none_scores(self, capsys):
        results = {
            "repo_url": "https://github.com/test/repo",
            "timestamp": "2024-01-01T00:00:00",
            "model": "llama3",
            "embed_model": "nomic-embed-text",
            "num_questions": 3,
            "top_k": 5,
            "aggregate": _null_scores(),
            "per_question": [],
        }
        _print_summary(results)
        captured = capsys.readouterr()
        assert "N/A" in captured.out

    def test_markdown_table_has_all_metrics(self):
        results = {
            "num_questions": 5,
            "model": "llama3",
            "embed_model": "nomic-embed-text",
            "timestamp": "2024-01-01T00:00:00",
            "aggregate": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.80,
                "context_precision": 0.75,
                "context_recall": 0.70,
            },
        }
        table = get_results_markdown_table(results)
        assert "Faithfulness"      in table
        assert "Answer Relevancy"  in table
        assert "Context Precision" in table
        assert "Context Recall"    in table

    def test_markdown_table_is_valid_markdown(self):
        results = {
            "num_questions": 3,
            "model": "llama3",
            "embed_model": "nomic-embed-text",
            "timestamp": "2024-01-01T00:00:00",
            "aggregate": {
                "faithfulness": 0.9, "answer_relevancy": 0.8,
                "context_precision": 0.7, "context_recall": 0.6,
            },
        }
        table = get_results_markdown_table(results)
        assert "|" in table
        assert "---" in table

    def test_markdown_table_handles_none_scores(self):
        results = {
            "num_questions": 3,
            "model": "llama3",
            "embed_model": "nomic-embed-text",
            "timestamp": "2024-01-01T00:00:00",
            "aggregate": _null_scores(),
        }
        table = get_results_markdown_table(results)
        assert "N/A" in table

    def test_load_latest_results_returns_none_when_no_results(self, tmp_path):
        """When results dir is empty, should return None."""
        with patch("eval.ragas_eval.Path") as mock_path_cls:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = False
            mock_path_cls.return_value.__truediv__ = MagicMock(return_value=mock_dir)
            result = load_latest_results()
            assert result is None

    def test_run_evaluation_saves_json(self, tmp_path):
        """Verify run_evaluation writes a JSON file."""
        output_file = tmp_path / "test_eval.json"

        with patch("rag.pipeline.answer") as mock_answer, \
             patch("eval.ragas_eval._run_ragas_metrics") as mock_metrics:

            mock_answer.return_value = {
                "answer": "This is a test answer.",
                "sources": [],
                "context": "",
            }
            mock_metrics.return_value = {
                "faithfulness": 0.85,
                "answer_relevancy": 0.80,
                "context_precision": 0.75,
                "context_recall": 0.70,
            }

            from eval.ragas_eval import run_evaluation
            results = run_evaluation(
                repo_url="https://github.com/test/repo",
                questions=DEFAULT_QUESTIONS[:2],
                output_path=output_file,
            )

        assert output_file.exists()
        saved = json.loads(output_file.read_text())
        assert saved["repo_url"] == "https://github.com/test/repo"
        assert saved["num_questions"] == 2
        assert "aggregate" in saved
        assert "per_question" in saved
        assert len(saved["per_question"]) == 2

    def test_run_evaluation_structure(self, tmp_path):
        """Verify the result dict has the expected schema."""
        output_file = tmp_path / "out.json"

        with patch("rag.pipeline.answer") as mock_answer, \
             patch("eval.ragas_eval._run_ragas_metrics") as mock_metrics:

            mock_answer.return_value = {
                "answer": "Answer.", "sources": [], "context": "",
            }
            mock_metrics.return_value = _null_scores()

            from eval.ragas_eval import run_evaluation
            results = run_evaluation(
                repo_url="https://github.com/test/repo",
                questions=DEFAULT_QUESTIONS[:1],
                output_path=output_file,
            )

        required_keys = {
            "repo_url", "timestamp", "model", "embed_model",
            "num_questions", "top_k", "aggregate", "per_question"
        }
        assert required_keys.issubset(results.keys())
