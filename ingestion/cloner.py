"""
ingestion/cloner.py — Clone public GitHub repositories locally.

Responsibilities:
  - Parse and validate a GitHub URL
  - Clone (shallow, depth=1) into a deterministic local path
  - Skip re-cloning if the repo is already present (cache hit)
  - Check repo size before cloning to avoid disk exhaustion
  - Return metadata: repo name, owner, local path, commit SHA

Design decisions:
  - Shallow clone (depth=1) keeps disk usage minimal; we only need
    the latest source, not git history.
  - We derive a stable directory name from the URL so multiple calls
    with the same URL never duplicate work.
  - GitPython is used for cross-platform compat and programmatic
    access to commit info without subprocess juggling.
"""

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import git
import httpx
from loguru import logger

from config import get_settings

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RepoInfo:
    owner: str
    name: str
    url: str                  # normalised https URL
    local_path: Path
    commit_sha: str           # HEAD after clone
    is_cached: bool           # True if we skipped cloning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GITHUB_RE = re.compile(
    r"(?:https?://github\.com/|git@github\.com:)"
    r"(?P<owner>[^/]+)/(?P<name>[^/\.]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)


def _parse_github_url(url: str) -> tuple[str, str]:
    """Return (owner, repo_name) or raise ValueError."""
    m = _GITHUB_RE.match(url.strip())
    if not m:
        raise ValueError(
            f"Invalid GitHub URL: {url!r}\n"
            "Expected format: https://github.com/owner/repo"
        )
    return m.group("owner"), m.group("name")


def _repo_dir(owner: str, name: str, base: Path) -> Path:
    """Deterministic local directory for a repo."""
    slug = f"{owner}__{name}".lower()
    return base / slug


def _check_repo_size_mb(owner: str, name: str, token: str = "") -> float:
    """
    Query the GitHub API for repo size (in KB) and convert to MB.
    Returns 0.0 if the request fails (we'll attempt clone anyway).
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{owner}/{name}",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        size_kb: int = resp.json().get("size", 0)
        return size_kb / 1024
    except Exception as exc:
        logger.warning(f"Could not fetch repo metadata: {exc}")
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clone_repo(
    url: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    force_reclone: bool = False,
) -> RepoInfo:
    """
    Clone a public GitHub repository and return RepoInfo.

    Args:
        url:               GitHub repo URL (https or ssh format).
        progress_callback: Optional fn(message: str) for UI updates.
        force_reclone:     Delete existing clone and start fresh.

    Returns:
        RepoInfo with local_path pointing to the cloned directory.

    Raises:
        ValueError:  Invalid URL or repo too large.
        RuntimeError: Clone failure.
    """
    cfg = get_settings()

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    # 1. Parse URL
    owner, name = _parse_github_url(url)
    https_url = f"https://github.com/{owner}/{name}.git"
    if cfg.github_token:
        https_url = f"https://{cfg.github_token}@github.com/{owner}/{name}.git"

    _progress(f"📦 Preparing to clone {owner}/{name} ...")

    # 2. Size guard
    size_mb = _check_repo_size_mb(owner, name, cfg.github_token)
    if size_mb > cfg.max_repo_size_mb:
        raise ValueError(
            f"Repository {owner}/{name} is {size_mb:.1f} MB, "
            f"which exceeds the {cfg.max_repo_size_mb} MB limit."
        )
    if size_mb:
        _progress(f"   Repo size: {size_mb:.1f} MB — within limits ✓")

    # 3. Resolve local path
    cfg.repos_dir.mkdir(parents=True, exist_ok=True)
    local_path = _repo_dir(owner, name, cfg.repos_dir)

    # 4. Cache check
    if local_path.exists() and not force_reclone:
        _progress(f"✅ Cache hit — using existing clone at {local_path}")
        try:
            repo = git.Repo(local_path)
            sha = repo.head.commit.hexsha
            return RepoInfo(
                owner=owner,
                name=name,
                url=f"https://github.com/{owner}/{name}",
                local_path=local_path,
                commit_sha=sha,
                is_cached=True,
            )
        except git.InvalidGitRepositoryError:
            logger.warning("Existing directory is not a valid git repo — re-cloning.")
            shutil.rmtree(local_path)

    # 5. Force reclone
    if local_path.exists() and force_reclone:
        _progress("🗑️  Removing existing clone for fresh re-clone ...")
        shutil.rmtree(local_path)

    # 6. Clone
    _progress(f"⬇️  Cloning {owner}/{name} (shallow, depth=1) ...")
    try:
        repo = git.Repo.clone_from(
            https_url,
            local_path,
            depth=1,                 # shallow — we don't need history
            multi_options=["--no-tags"],
        )
    except git.GitCommandError as exc:
        raise RuntimeError(
            f"Failed to clone {https_url}\n"
            f"Git error: {exc}"
        ) from exc

    sha = repo.head.commit.hexsha
    _progress(f"✅ Cloned {owner}/{name} @ {sha[:8]}")

    return RepoInfo(
        owner=owner,
        name=name,
        url=f"https://github.com/{owner}/{name}",
        local_path=local_path,
        commit_sha=sha,
        is_cached=False,
    )


def delete_repo_cache(url: str) -> bool:
    """Remove a locally cloned repo. Returns True if deleted."""
    cfg = get_settings()
    try:
        owner, name = _parse_github_url(url)
    except ValueError:
        return False
    local_path = _repo_dir(owner, name, cfg.repos_dir)
    if local_path.exists():
        shutil.rmtree(local_path)
        logger.info(f"Deleted cache for {owner}/{name}")
        return True
    return False
