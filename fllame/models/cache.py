"""Lists models already present in HF's own local cache - a pure
filesystem scan (`huggingface_hub.scan_cache_dir`), no network involved,
so this always works offline. `is_model_cached` and
`local_estimate_vram_gb` (below) are `fllame serve`/`recipe build`'s
presence and VRAM checks respectively - deliberately built on this same
filesystem scan rather than `models/puller.py`'s `pull_model`, which can
touch the network: this module belongs entirely to the "verify, never
fetch" side of the setup/running boundary (see CLAUDE.md).
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


def _find_cached_repo(cache_info, repo_id: str) -> CachedRepoInfo | None:
    return next(
        (r for r in cache_info.repos if r.repo_type == "model" and r.repo_id == repo_id), None
    )


def is_model_cached(repo_id: str) -> bool:
    """Whether repo_id has at least one revision fully present in the
    local HF cache. This is the presence check `fllame serve` and
    `recipe build`/`recipe add --build` use to decide whether to
    proceed at all - a pure filesystem scan, never a network call, so
    it can never be the thing that lets `serve` cross the internet
    boundary. `False` for a repo that was never pulled, same as one
    that's only partially there.
    """
    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        return False
    repo = _find_cached_repo(cache_info, repo_id)
    return repo is not None and bool(repo.revisions)


def local_estimate_vram_gb(repo_id: str) -> float | None:
    """A weights-only VRAM estimate for a model already present in the
    local HF cache, summed directly from the real on-disk size of its
    cached `.safetensors` files - a pure filesystem scan, same as
    `list_cached_models`/`is_model_cached` above, no network involved.
    This is `fllame serve`'s pre-flight sanity check's data source
    (`cli.py`'s `_warn_if_vram_likely_insufficient`), run only after
    `is_model_cached` has already confirmed the model is present - so
    there's no need for a Hub lookup the way `models/discovery.py`'s
    `_estimated_vram_gb` (used by `model scan`, for a model that isn't
    downloaded yet) has to make - and this on-disk figure is the
    literal physical size, not one derived from a per-dtype element
    count.

    `None` when the repo isn't cached at all, has no revisions on
    disk, or its most recently used revision has no `.safetensors`
    files (e.g. a GGUF-only download) - never a guess.
    """
    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        return None
    repo = _find_cached_repo(cache_info, repo_id)
    if repo is None or not repo.revisions:
        return None

    revision = max(repo.revisions, key=lambda r: r.last_modified)
    total_bytes = sum(
        f.size_on_disk for f in revision.files if f.file_name.endswith(".safetensors")
    )
    if total_bytes == 0:
        return None
    return total_bytes / (1024**3)
