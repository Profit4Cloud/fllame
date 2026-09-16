"""Reads the architecture facts `models/sizing.py`'s KV-cache math needs
out of a cached model's own `config.json` - a second, narrower pass over
the same local cache `models/cache.py` scans for `.safetensors` sizes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound

from fllame.models.cache import _find_cached_repo, _most_recent_revision


@dataclass(frozen=True)
class ModelArchitecture:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_context_length: int


def read_architecture(repo_id: str) -> ModelArchitecture | None:
    """`None` whenever `repo_id` isn't cached, has no `config.json`, or
    that file doesn't match the common dense-transformer shape this
    parses - a deliberate skip, not an error, so every caller treats it
    as "this feature doesn't apply here", never as a bug to surface."""
    try:
        cache_info = scan_cache_dir()
    except CacheNotFound:
        return None
    repo = _find_cached_repo(cache_info, repo_id)
    revision = _most_recent_revision(repo) if repo is not None else None
    if revision is None:
        return None

    config_path = revision.snapshot_path / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError):
        return None

    num_layers = config.get("num_hidden_layers")
    hidden_size = config.get("hidden_size")
    max_context_length = config.get("max_position_embeddings")
    num_attention_heads = config.get("num_attention_heads")
    if (
        num_layers is None
        or hidden_size is None
        or max_context_length is None
        or num_attention_heads is None
    ):
        return None

    num_kv_heads = config.get("num_key_value_heads")
    if num_kv_heads is None:
        num_kv_heads = num_attention_heads

    head_dim = config.get("head_dim")
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads

    return ModelArchitecture(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        # Used as-is, deliberately not adjusted for `rope_scaling`:
        # whether that field's `factor` should multiply this, or whether
        # `max_position_embeddings` is already the post-scaling value, is
        # inconsistent enough across configs that guessing wrong risks
        # *overstating* the model's real context ceiling - the unsafe
        # direction here. Understating (ignoring rope-extended context)
        # only makes this default more conservative than it strictly
        # needs to be.
        max_context_length=max_context_length,
    )
