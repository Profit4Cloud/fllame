"""Downloads a model into HF's own cache - wherever huggingface_hub
itself resolves that to - so fllame never has an opinion of its own
about where models live on disk, and `vllm serve` never has to fall back
on its own (unreliable) auto-download inside the container: fllame
guarantees the model is already fully present first.
"""

from __future__ import annotations

from huggingface_hub import snapshot_download


def pull_model(repo_id: str) -> str:
    """Downloads repo_id into the HF cache - a no-op if it's already
    fully present - and returns the local snapshot path."""
    return snapshot_download(repo_id)
