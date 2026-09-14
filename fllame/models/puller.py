from __future__ import annotations

from huggingface_hub import snapshot_download


def pull_model(repo_id: str, *, offline: bool = False) -> str:
    """A no-op if `repo_id` is already fully cached. Returns the local
    snapshot path."""
    return snapshot_download(repo_id, local_files_only=offline)
