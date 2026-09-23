import json
from pathlib import Path

import pytest

from fllame.models import vram
from fllame.models.vram import RUNTIME_OVERHEAD_GB, VramEstimateError, estimate_vram

_GIB = 1024**3

# 8 layers x 8 KV heads x 128 head dim x (K+V) x 2 bytes = 32 KiB per
# token, so 1 GiB of KV cache per 32768-token sequence.
_DENSE_CONFIG = {
    "num_hidden_layers": 8,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "torch_dtype": "bfloat16",
}


def _patch_cache(monkeypatch, tmp_path: Path, config: dict | None, *, weights_gb=10.0, cached=True):
    config_path = tmp_path / "config.json"
    if config is not None:
        config_path.write_text(json.dumps(config))
    monkeypatch.setattr(vram, "is_model_cached", lambda repo_id: cached)
    monkeypatch.setattr(vram, "local_estimate_vram_gb", lambda repo_id: weights_gb)
    monkeypatch.setattr(
        vram, "cached_file", lambda repo_id, name: config_path if config is not None else None
    )


def _part(estimate, label):
    return next(p for p in estimate.parts if p.label == label)


def test_total_is_weights_plus_kv_cache_plus_overhead(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path, _DENSE_CONFIG)

    estimate = estimate_vram("org/m", max_model_len=32768, max_num_seqs=4)

    assert _part(estimate, "Weights").gb == 10.0
    assert _part(estimate, "KV cache").gb == 4.0
    assert estimate.total_gb == 10.0 + 4.0 + RUNTIME_OVERHEAD_GB


def test_fp8_kv_cache_halves_it(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path, _DENSE_CONFIG)

    estimate = estimate_vram("org/m", max_model_len=32768, max_num_seqs=1, kv_cache_dtype="fp8")

    assert _part(estimate, "KV cache").gb == 0.5


def test_auto_kv_cache_follows_the_model_dtype(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path, {**_DENSE_CONFIG, "torch_dtype": "float32"})

    estimate = estimate_vram("org/m", max_model_len=32768, max_num_seqs=1)

    assert _part(estimate, "KV cache").gb == 2.0


def test_unsupported_kv_cache_dtype_is_an_error(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path, _DENSE_CONFIG)

    with pytest.raises(VramEstimateError, match="kv-cache-dtype"):
        estimate_vram("org/m", max_model_len=32768, max_num_seqs=1, kv_cache_dtype="weird")


def test_head_dim_derived_and_kv_heads_default_to_attention_heads(monkeypatch, tmp_path):
    config = {"num_hidden_layers": 8, "num_attention_heads": 8, "hidden_size": 1024}
    _patch_cache(monkeypatch, tmp_path, config)

    estimate = estimate_vram("org/m", max_model_len=32768, max_num_seqs=1)

    assert _part(estimate, "KV cache").gb == 1.0


def test_multimodal_config_reads_text_config_and_notes_the_vision_encoder(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path, {"text_config": _DENSE_CONFIG, "vision_config": {}})

    estimate = estimate_vram("org/m", max_model_len=32768, max_num_seqs=1)

    assert _part(estimate, "KV cache").gb == 1.0
    assert any("vision encoder" in note for note in estimate.notes)


def test_hybrid_model_caches_only_full_attention_layers_plus_linear_state(monkeypatch, tmp_path):
    config = {
        **_DENSE_CONFIG,
        "layer_types": ["linear_attention"] * 6 + ["full_attention"] * 2,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    _patch_cache(monkeypatch, tmp_path, config)

    estimate = estimate_vram("org/m", max_model_len=32768, max_num_seqs=2)

    assert _part(estimate, "KV cache").gb == 0.5
    # Per layer per sequence: 4 x 128 x 128 recurrent + 3 x (2 x 2 x 128 + 4 x 128) conv.
    per_seq_bytes = 6 * (4 * 128 * 128 + 3 * (2 * 2 * 128 + 4 * 128)) * 2
    assert _part(estimate, "Linear-attention state").gb == 2 * per_seq_bytes / _GIB


def test_full_attention_interval_counts_every_nth_layer(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path, {**_DENSE_CONFIG, "full_attention_interval": 4})

    estimate = estimate_vram("org/m", max_model_len=32768, max_num_seqs=1)

    assert _part(estimate, "KV cache").gb == 0.25


def test_sliding_window_layers_cache_at_most_their_window(monkeypatch, tmp_path):
    config = {
        **_DENSE_CONFIG,
        "layer_types": ["sliding_attention"] * 4 + ["full_attention"] * 4,
        "sliding_window": 4096,
    }
    _patch_cache(monkeypatch, tmp_path, config)

    estimate = estimate_vram("org/m", max_model_len=32768, max_num_seqs=1)

    assert _part(estimate, "KV cache").gb == 0.5 + 0.5 / 8


def test_mla_caches_one_latent_per_token_per_layer(monkeypatch, tmp_path):
    config = {
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "torch_dtype": "bfloat16",
    }
    _patch_cache(monkeypatch, tmp_path, config)

    estimate = estimate_vram("org/m", max_model_len=32768, max_num_seqs=1, tensor_parallel_size=8)

    assert _part(estimate, "KV cache").gb == 576 * 2 * 61 * 32768 / _GIB


def test_tensor_parallel_splits_weights_and_kv_heads_per_gpu(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path, _DENSE_CONFIG)

    estimate = estimate_vram("org/m", max_model_len=32768, max_num_seqs=1, tensor_parallel_size=2)

    assert _part(estimate, "Weights").gb == 5.0
    assert _part(estimate, "KV cache").gb == 0.5
    assert any("Per GPU" in note for note in estimate.notes)


def test_not_pulled_is_an_error(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path, _DENSE_CONFIG, cached=False)

    with pytest.raises(VramEstimateError, match="not pulled"):
        estimate_vram("org/m", max_model_len=32768, max_num_seqs=1)


def test_missing_config_is_an_error(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path, None)

    with pytest.raises(VramEstimateError, match="config.json"):
        estimate_vram("org/m", max_model_len=32768, max_num_seqs=1)


def test_unrecognized_architecture_is_an_error(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path, {"model_type": "something-new"})

    with pytest.raises(VramEstimateError, match="unrecognized architecture"):
        estimate_vram("org/m", max_model_len=32768, max_num_seqs=1)
