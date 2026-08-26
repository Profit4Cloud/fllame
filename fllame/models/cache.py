"""Lists models already present in HF's own local cache - a pure
filesystem scan (`huggingface_hub.scan_cache_dir`), no network involved,
so this always works offline.
"""

from __future__ import annotations

from huggingface_hub import CachedRepoInfo, scan_cache_dir
from huggingface_hub.errors import CacheNotFound


def list_cached_models() -> list[CachedRepoInfo]:
    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        # Nothing has ever been pulled - a normal state on a fresh
        # install, not an error.
        return []
    return sorted(
        (repo for repo in cache_info.repos if repo.repo_type == "model"),
        key=lambda repo: repo.repo_id,
    )
