"""Reads HF's local cache via `huggingface_hub.scan_cache_dir` - a pure
filesystem scan, never `models/puller.py`'s network-touching
`pull_model`.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import CachedRepoInfo, CachedRevisionInfo, scan_cache_dir
from huggingface_hub.errors import CacheNotFound


def list_cached_models() -> list[CachedRepoInfo]:
    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        # Nothing pulled yet - normal on a fresh install, not an error.
        return []
    return sorted(
        (repo for repo in cache_info.repos if repo.repo_type == "model"),
        key=lambda repo: repo.repo_id,
    )


def _find_cached_repo(cache_info, repo_id: str) -> CachedRepoInfo | None:
    return next(
        (r for r in cache_info.repos if r.repo_type == "model" and r.repo_id == repo_id), None
    )


def _most_recent_revision(repo: CachedRepoInfo) -> CachedRevisionInfo | None:
    return max(repo.revisions, key=lambda r: r.last_modified) if repo.revisions else None


def is_model_cached(repo_id: str) -> bool:
    """`False` both for a repo never pulled and one only partially
    cached."""
    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        return False
    repo = _find_cached_repo(cache_info, repo_id)
    return repo is not None and bool(repo.revisions)


def local_estimate_vram_gb(repo_id: str) -> float | None:
    """Weights-only, from the real on-disk size of cached `.safetensors`
    files. `None` if uncached, or the most recent revision has no
    `.safetensors` (e.g. GGUF-only)."""
    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        return None
    repo = _find_cached_repo(cache_info, repo_id)
    revision = _most_recent_revision(repo) if repo is not None else None
    if revision is None:
        return None

    total_bytes = sum(
        f.size_on_disk for f in revision.files if f.file_name.endswith(".safetensors")
    )
    if total_bytes == 0:
        return None
    return total_bytes / (1024**3)


def cached_revision_hash(repo_id: str) -> str | None:
    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        return None
    repo = _find_cached_repo(cache_info, repo_id)
    revision = _most_recent_revision(repo) if repo is not None else None
    return revision.commit_hash if revision is not None else None


def cached_file(repo_id: str, filename: str) -> Path | None:
    """`filename` from the most recent cached revision's snapshot."""
    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        return None
    repo = _find_cached_repo(cache_info, repo_id)
    revision = _most_recent_revision(repo) if repo is not None else None
    if revision is None:
        return None
    path = Path(revision.snapshot_path) / filename
    return path if path.is_file() else None
