"""
ingestion/parser.py — Parse source files and produce semantic code chunks.

This is the most critical module in the ingestion pipeline.

Design philosophy:
  - "Semantic" chunking means we try to keep logical units whole:
    functions, classes, methods — not arbitrary N-character windows.
  - We use Python's built-in `ast` module for .py files (most reliable).
  - For other languages we use tree-sitter (fast, accurate, Wasm-free).
  - If parsing fails for ANY reason, we fall back gracefully to
    line-based sliding-window chunking so no file is silently skipped.
  - Every chunk carries rich metadata so the UI can show
    "Found in src/auth.py → AuthService.login() (lines 42-78)".

Chunk metadata schema:
    {
        "chunk_id":      "<sha256 of content>",
        "repo_url":      "https://github.com/owner/repo",
        "file_path":     "src/auth/service.py",          # relative
        "language":      "python",
        "symbol_name":   "AuthService.login",            # fn/class name
        "symbol_type":   "method",                       # function|class|module
        "start_line":    42,
        "end_line":      78,
        "content":       "<raw source code>",
        "content_hash":  "<sha256>",                     # for cache check
    }
"""

import ast
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Optional

from loguru import logger

from config import get_settings


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

# Map file extension → (language_name, parser_strategy)
# strategy: "python_ast" | "tree_sitter" | "line_based"
EXTENSION_MAP: dict[str, tuple[str, str]] = {
    ".py":    ("python",     "python_ast"),
    ".js":    ("javascript", "tree_sitter"),
    ".jsx":   ("javascript", "tree_sitter"),
    ".ts":    ("typescript", "tree_sitter"),
    ".tsx":   ("typescript", "tree_sitter"),
    ".java":  ("java",       "tree_sitter"),
    ".go":    ("go",         "tree_sitter"),
    ".rs":    ("rust",       "tree_sitter"),
    ".cpp":   ("cpp",        "line_based"),
    ".c":     ("c",          "line_based"),
    ".h":     ("c",          "line_based"),
    ".cs":    ("csharp",     "line_based"),
    ".rb":    ("ruby",       "line_based"),
    ".php":   ("php",        "line_based"),
    ".swift": ("swift",      "line_based"),
    ".kt":    ("kotlin",     "line_based"),
    ".scala": ("scala",      "line_based"),
    ".sh":    ("bash",       "line_based"),
    ".yaml":  ("yaml",       "line_based"),
    ".yml":   ("yaml",       "line_based"),
    ".json":  ("json",       "line_based"),
    ".toml":  ("toml",       "line_based"),
    ".md":    ("markdown",   "line_based"),
    ".sql":   ("sql",        "line_based"),
    ".html":  ("html",       "line_based"),
    ".css":   ("css",        "line_based"),
}

# Files/dirs to always skip
SKIP_DIRS = {
    ".git", ".github", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".env", "dist", "build", "target", ".next", ".nuxt",
    "vendor", "bower_components", ".idea", ".vscode", "coverage",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
}

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".lock", ".sum",
    ".min.js", ".min.css",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    repo_url: str
    file_path: str          # relative to repo root
    language: str
    symbol_name: str        # e.g. "MyClass.my_method" or "" for module-level
    symbol_type: str        # "function" | "class" | "method" | "module"
    start_line: int
    end_line: int
    content: str

    # computed on __post_init__
    chunk_id: str = field(init=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        digest = hashlib.sha256(self.content.encode()).hexdigest()
        self.content_hash = digest
        self.chunk_id = hashlib.sha256(
            f"{self.repo_url}::{self.file_path}::{self.start_line}".encode()
        ).hexdigest()[:16]

    def to_metadata(self) -> dict:
        """Serialise to a flat dict suitable for ChromaDB metadata."""
        return {
            "chunk_id":     self.chunk_id,
            "repo_url":     self.repo_url,
            "file_path":    self.file_path,
            "language":     self.language,
            "symbol_name":  self.symbol_name,
            "symbol_type":  self.symbol_type,
            "start_line":   self.start_line,
            "end_line":     self.end_line,
            "content_hash": self.content_hash,
        }


# ---------------------------------------------------------------------------
# Python AST chunker
# ---------------------------------------------------------------------------

class PythonASTChunker:
    """
    Uses Python's built-in `ast` module to extract top-level and nested
    function/class definitions as individual chunks.

    Falls back to line-based chunking if the file has syntax errors.
    """

    def __init__(self, max_chunk_lines: int = 150):
        self.max_chunk_lines = max_chunk_lines

    def chunk(
        self,
        source: str,
        file_path: str,
        repo_url: str,
    ) -> list[CodeChunk]:
        lines = source.splitlines()
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            logger.debug(f"AST parse failed for {file_path}: {exc} — using line fallback")
            return _line_based_chunks(source, file_path, repo_url, "python")

        chunks: list[CodeChunk] = []
        self._visit(tree, lines, file_path, repo_url, parent_name="", chunks=chunks)

        # If the whole file produced no AST chunks (e.g. script with only
        # top-level statements), treat the file as a single module chunk.
        if not chunks:
            chunks.append(CodeChunk(
                repo_url=repo_url,
                file_path=file_path,
                language="python",
                symbol_name="<module>",
                symbol_type="module",
                start_line=1,
                end_line=len(lines),
                content=source,
            ))
        return chunks

    def _visit(
        self,
        node: ast.AST,
        lines: list[str],
        file_path: str,
        repo_url: str,
        parent_name: str,
        chunks: list[CodeChunk],
    ) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{parent_name}.{child.name}" if parent_name else child.name
                start = child.lineno
                end = child.end_lineno or start

                # If the function is very large, further split it
                if (end - start) > self.max_chunk_lines:
                    sub = _line_based_chunks(
                        "\n".join(lines[start - 1:end]),
                        file_path,
                        repo_url,
                        "python",
                        start_offset=start - 1,
                    )
                    for s in sub:
                        s.symbol_name = name
                        s.symbol_type = "function"
                    chunks.extend(sub)
                else:
                    chunks.append(CodeChunk(
                        repo_url=repo_url,
                        file_path=file_path,
                        language="python",
                        symbol_name=name,
                        symbol_type="function",
                        start_line=start,
                        end_line=end,
                        content="\n".join(lines[start - 1:end]),
                    ))
                # Recurse into nested functions
                self._visit(child, lines, file_path, repo_url, name, chunks)

            elif isinstance(child, ast.ClassDef):
                name = f"{parent_name}.{child.name}" if parent_name else child.name
                start = child.lineno
                end = child.end_lineno or start
                # Add a chunk for the class header + docstring (first ~10 lines)
                header_end = min(start + 10, end)
                chunks.append(CodeChunk(
                    repo_url=repo_url,
                    file_path=file_path,
                    language="python",
                    symbol_name=name,
                    symbol_type="class",
                    start_line=start,
                    end_line=end,
                    content="\n".join(lines[start - 1:end]),
                ))
                # Recurse into methods
                self._visit(child, lines, file_path, repo_url, name, chunks)


# ---------------------------------------------------------------------------
# Tree-sitter chunker (JS, TS, Java, Go, Rust)
# ---------------------------------------------------------------------------

class TreeSitterChunker:
    """
    Uses tree-sitter to parse non-Python source files.

    We lazily load language grammars to avoid import-time overhead.
    Falls back to line_based if tree-sitter is unavailable or parse fails.
    """

    # Mapping language name → tree-sitter grammar module
    _GRAMMAR_MODULES = {
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "java":       "tree_sitter_java",
        "go":         "tree_sitter_go",
        "rust":       "tree_sitter_rust",
    }

    # Node types we consider "semantic units" per language
    _SYMBOL_NODES = {
        "javascript": {"function_declaration", "arrow_function", "class_declaration",
                       "method_definition", "function_expression"},
        "typescript": {"function_declaration", "arrow_function", "class_declaration",
                       "method_definition", "function_expression", "interface_declaration"},
        "java":       {"method_declaration", "class_declaration", "constructor_declaration",
                       "interface_declaration"},
        "go":         {"function_declaration", "method_declaration", "type_declaration"},
        "rust":       {"function_item", "impl_item", "struct_item", "trait_item",
                       "enum_item", "mod_item"},
    }

    def chunk(
        self,
        source: str,
        file_path: str,
        repo_url: str,
        language: str,
    ) -> list[CodeChunk]:
        try:
            import tree_sitter  # noqa: F401
            grammar_mod = self._GRAMMAR_MODULES.get(language)
            if not grammar_mod:
                raise ImportError(f"No grammar for {language}")

            import importlib
            mod = importlib.import_module(grammar_mod)

            from tree_sitter import Language, Parser
            lang = Language(mod.language())
            parser = Parser(lang)
            tree = parser.parse(bytes(source, "utf-8"))
        except Exception as exc:
            logger.debug(f"tree-sitter unavailable for {language}: {exc} — falling back")
            return _line_based_chunks(source, file_path, repo_url, language)

        lines = source.splitlines()
        symbol_node_types = self._SYMBOL_NODES.get(language, set())
        chunks: list[CodeChunk] = []
        self._walk(tree.root_node, lines, file_path, repo_url, language,
                   symbol_node_types, chunks, parent_name="")

        if not chunks:
            # Nothing matched — treat whole file as one chunk
            chunks.append(CodeChunk(
                repo_url=repo_url,
                file_path=file_path,
                language=language,
                symbol_name="<module>",
                symbol_type="module",
                start_line=1,
                end_line=len(lines),
                content=source,
            ))
        return chunks

    def _walk(self, node, lines, file_path, repo_url, language,
              symbol_types, chunks, parent_name):
        if node.type in symbol_types:
            # Extract name from common child patterns
            name = self._extract_name(node) or node.type
            qualified = f"{parent_name}.{name}" if parent_name else name
            start = node.start_point[0] + 1  # 0-indexed → 1-indexed
            end = node.end_point[0] + 1
            content = "\n".join(lines[start - 1:end])
            symbol_type = "class" if "class" in node.type else "function"
            chunks.append(CodeChunk(
                repo_url=repo_url,
                file_path=file_path,
                language=language,
                symbol_name=qualified,
                symbol_type=symbol_type,
                start_line=start,
                end_line=end,
                content=content,
            ))
            # Recurse with this symbol as parent
            for child in node.children:
                self._walk(child, lines, file_path, repo_url, language,
                           symbol_types, chunks, qualified)
        else:
            for child in node.children:
                self._walk(child, lines, file_path, repo_url, language,
                           symbol_types, chunks, parent_name)

    @staticmethod
    def _extract_name(node) -> Optional[str]:
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "property_identifier"):
                return child.text.decode("utf-8", errors="replace")
        return None


# ---------------------------------------------------------------------------
# Fallback: line-based sliding-window chunker
# ---------------------------------------------------------------------------

def _line_based_chunks(
    source: str,
    file_path: str,
    repo_url: str,
    language: str,
    start_offset: int = 0,
    lines_per_chunk: int = 80,
    overlap: int = 10,
) -> list[CodeChunk]:
    """
    Simple sliding-window fallback. Produces chunks of ~80 lines
    with 10-line overlap to preserve context at boundaries.
    """
    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    i = 0
    while i < len(lines):
        end = min(i + lines_per_chunk, len(lines))
        content = "\n".join(lines[i:end])
        if content.strip():
            chunks.append(CodeChunk(
                repo_url=repo_url,
                file_path=file_path,
                language=language,
                symbol_name="",
                symbol_type="module",
                start_line=start_offset + i + 1,
                end_line=start_offset + end,
                content=content,
            ))
        i += lines_per_chunk - overlap
    return chunks


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

_python_chunker = PythonASTChunker()
_ts_chunker = TreeSitterChunker()


def parse_file(
    file_path: Path,
    repo_root: Path,
    repo_url: str,
) -> list[CodeChunk]:
    """
    Parse a single source file and return its chunks.

    Args:
        file_path:  Absolute path to the file.
        repo_root:  Absolute path to the repo root (for relative paths).
        repo_url:   Canonical GitHub URL (stored in metadata).

    Returns:
        List of CodeChunk objects. Empty list if file should be skipped.
    """
    cfg = get_settings()
    ext = "".join(file_path.suffixes).lower()

    # Skip unknown/binary extensions
    if ext not in EXTENSION_MAP:
        return []

    # Skip oversized files
    size_kb = file_path.stat().st_size / 1024
    if size_kb > cfg.max_file_size_kb:
        logger.debug(f"Skipping {file_path.name} ({size_kb:.0f} KB > limit)")
        return []

    language, strategy = EXTENSION_MAP[ext]
    rel_path = str(file_path.relative_to(repo_root))

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning(f"Cannot read {rel_path}: {exc}")
        return []

    if not source.strip():
        return []

    if strategy == "python_ast":
        return _python_chunker.chunk(source, rel_path, repo_url)
    elif strategy == "tree_sitter":
        return _ts_chunker.chunk(source, rel_path, repo_url, language)
    else:
        return _line_based_chunks(source, rel_path, repo_url, language)


def parse_repo(
    repo_root: Path,
    repo_url: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Generator[CodeChunk, None, None]:
    """
    Walk an entire repo and yield CodeChunks.

    Args:
        repo_root:         Path to the cloned repo.
        repo_url:          Canonical GitHub URL.
        progress_callback: fn(message, current_file_index, total_files).

    Yields:
        CodeChunk objects, one per parsed code unit.
    """
    # Collect eligible files first (for progress reporting)
    all_files: list[Path] = []
    for root, dirs, files in os.walk(repo_root):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            fpath = Path(root) / fname
            ext = "".join(fpath.suffixes).lower()
            if ext in EXTENSION_MAP and ext not in SKIP_EXTENSIONS:
                all_files.append(fpath)

    total = len(all_files)
    logger.info(f"Found {total} source files to parse in {repo_root}")

    for idx, fpath in enumerate(all_files):
        if progress_callback:
            progress_callback(
                f"Parsing {fpath.relative_to(repo_root)} ({idx + 1}/{total})",
                idx + 1,
                total,
            )
        chunks = parse_file(fpath, repo_root, repo_url)
        for chunk in chunks:
            yield chunk


def get_repo_summary(repo_root: Path) -> dict:
    """
    Return a summary dict: total files, language breakdown, total chunks estimate.
    Used by the Streamlit UI on first load.
    """
    lang_counts: dict[str, int] = {}
    total_files = 0
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            ext = "".join(Path(fname).suffixes).lower()
            if ext in EXTENSION_MAP:
                lang, _ = EXTENSION_MAP[ext]
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                total_files += 1
    return {
        "total_files": total_files,
        "languages": dict(sorted(lang_counts.items(), key=lambda x: -x[1])),
    }
