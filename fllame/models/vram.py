"""Estimates the VRAM a pulled model actually needs under a given vLLM
configuration, from its cached `config.json` and weight files - no
network. vLLM itself claims `--gpu-memory-utilization` of the GPU
regardless, so this is the number to size that flag against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from fllame.models.cache import cached_file, is_model_cached, local_estimate_vram_gb

# CUDA context, activation workspace and CUDA graphs - a fixed guess,
# not derived from the model; typically 1-3 GB in practice.
RUNTIME_OVERHEAD_GB = 2.0

_GIB = 1024**3

_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "half": 2, "float32": 4, "float": 4}


class VramEstimateError(Exception):
    pass


@dataclass(frozen=True)
class VramPart:
    label: str
    gb: float
    formula: str


@dataclass(frozen=True)
class VramEstimate:
    parts: list[VramPart]
    notes: list[str]

    @property
    def total_gb(self) -> float:
        return sum(part.gb for part in self.parts)


def estimate_vram(
    repo_id: str,
    *,
    max_model_len: int,
    max_num_seqs: int,
    kv_cache_dtype: str = "auto",
    tensor_parallel_size: int = 1,
) -> VramEstimate:
    """Per GPU when `tensor_parallel_size` > 1: weights and attention KV
    heads split across GPUs, overhead doesn't."""
    if not is_model_cached(repo_id):
        raise VramEstimateError(
            f"'{repo_id}' is not pulled - run `fllame model pull {repo_id}` first."
        )
    weights_gb = local_estimate_vram_gb(repo_id)
    if weights_gb is None:
        raise VramEstimateError(f"'{repo_id}' has no cached .safetensors weight files.")
    config_path = cached_file(repo_id, "config.json")
    if config_path is None:
        raise VramEstimateError(f"'{repo_id}' has no cached config.json.")
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError) as e:
        raise VramEstimateError(f"'{repo_id}': unreadable config.json: {e}") from e

    text_config = config.get("text_config") or config.get("llm_config") or config
    model_dtype = str(text_config.get("torch_dtype") or config.get("torch_dtype") or "")
    model_dtype = model_dtype or str(text_config.get("dtype") or config.get("dtype") or "")
    tp = tensor_parallel_size

    parts = [
        VramPart(
            "Weights",
            weights_gb / tp,
            "cached .safetensors files"
            + (f" / {tp} GPUs (--tensor-parallel-size)" if tp > 1 else ""),
        ),
        _kv_cache_part(
            text_config,
            kv_bytes=_kv_cache_bytes_per_element(kv_cache_dtype, model_dtype),
            kv_dtype_label=kv_cache_dtype
            if kv_cache_dtype != "auto"
            else f"auto = {model_dtype or 'bf16, assumed'}",
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tp,
        ),
    ]
    state = _linear_attention_state_part(
        text_config,
        state_bytes=_DTYPE_BYTES.get(model_dtype, 2),
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tp,
    )
    if state is not None:
        parts.append(state)
    parts.append(
        VramPart(
            "Runtime overhead",
            RUNTIME_OVERHEAD_GB,
            "CUDA context, activations and CUDA graphs - a fixed estimate",
        )
    )

    notes = []
    if tp > 1:
        notes.append(f"Per GPU, across {tp} GPUs.")
    if "vision_config" in config:
        notes.append("The vision encoder's activations for image inputs are not included.")
    return VramEstimate(parts=parts, notes=notes)


def _kv_cache_bytes_per_element(kv_cache_dtype: str, model_dtype: str) -> int:
    """vLLM's "auto" stores the KV cache in the model's own dtype, i.e.
    the unquantized one - weight quantization doesn't shrink it."""
    dtype = kv_cache_dtype.lower()
    if dtype == "auto":
        return _DTYPE_BYTES.get(model_dtype, 2)
    if dtype.startswith("fp8"):
        return 1
    if dtype in _DTYPE_BYTES:
        return _DTYPE_BYTES[dtype]
    raise VramEstimateError(f"unsupported --kv-cache-dtype {kv_cache_dtype!r}")


def _kv_cache_part(
    text_config: dict,
    *,
    kv_bytes: int,
    kv_dtype_label: str,
    max_model_len: int,
    max_num_seqs: int,
    tensor_parallel_size: int,
) -> VramPart:
    """Only attention layers hold a per-token KV cache: a
    linear-attention layer keeps a fixed per-sequence state instead
    (see `_linear_attention_state_part`), and a sliding-window layer
    caches at most its window. MLA caches one compressed latent per
    token, replicated on every GPU rather than split by head."""
    try:
        layer_count = int(text_config["num_hidden_layers"])
        layer_types = text_config.get("layer_types")
        interval = text_config.get("full_attention_interval")
        if layer_types:
            full_layers = sum(t == "full_attention" for t in layer_types)
            sliding_layers = sum(t == "sliding_attention" for t in layer_types)
        elif interval:
            full_layers, sliding_layers = layer_count // int(interval), 0
        else:
            full_layers, sliding_layers = layer_count, 0

        if text_config.get("kv_lora_rank"):
            latent = int(text_config["kv_lora_rank"]) + int(text_config["qk_rope_head_dim"])
            elements = latent
            per_layer = f"({latent} MLA latent)"
        else:
            heads = int(text_config["num_attention_heads"])
            kv_heads = int(text_config.get("num_key_value_heads") or heads)
            head_dim = int(text_config.get("head_dim") or int(text_config["hidden_size"]) // heads)
            kv_heads_per_gpu = max(1, kv_heads // tensor_parallel_size)
            elements = 2 * kv_heads_per_gpu * head_dim
            per_layer = f"{kv_heads_per_gpu} KV heads x {head_dim} head dim x 2 (K+V)"

        window = min(int(text_config.get("sliding_window") or max_model_len), max_model_len)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        raise VramEstimateError(
            "can't compute the KV cache from this model's config.json "
            f"(unrecognized architecture, missing {e})"
        ) from e

    bytes_per_token_per_layer = elements * kv_bytes
    cached_tokens_per_seq = full_layers * max_model_len + sliding_layers * window
    total_bytes = bytes_per_token_per_layer * cached_tokens_per_seq * max_num_seqs

    layers = f"{full_layers} attention layers"
    if sliding_layers:
        layers += f" at {max_model_len} tokens + {sliding_layers} sliding-window layers at {window}"
    else:
        layers += f" x {max_model_len} tokens (--max-model-len)"
    formula = (
        f"{per_layer} x {kv_bytes} byte(s) ({kv_dtype_label}) "
        f"= {bytes_per_token_per_layer / 1024:.1f} KiB per token per layer; "
        f"x {layers}; x {max_num_seqs} sequences (--max-num-seqs)"
    )
    return VramPart("KV cache", total_bytes / _GIB, formula)


def _linear_attention_state_part(
    text_config: dict, *, state_bytes: int, max_num_seqs: int, tensor_parallel_size: int
) -> VramPart | None:
    """Gated DeltaNet (Qwen3-Next style) linear-attention layers: a
    recurrent state plus a short convolution window per sequence, fixed
    in size whatever the context length."""
    try:
        layer_count = int(text_config["num_hidden_layers"])
        v_heads = int(text_config["linear_num_value_heads"])
        k_heads = int(text_config["linear_num_key_heads"])
        k_dim = int(text_config["linear_key_head_dim"])
        v_dim = int(text_config["linear_value_head_dim"])
        conv_kernel = int(text_config["linear_conv_kernel_dim"])
    except (KeyError, TypeError, ValueError):
        return None

    layer_types = text_config.get("layer_types")
    interval = text_config.get("full_attention_interval")
    if layer_types:
        linear_layers = sum(t == "linear_attention" for t in layer_types)
    elif interval:
        linear_layers = layer_count - layer_count // int(interval)
    else:
        return None

    recurrent = v_heads * k_dim * v_dim
    conv = (conv_kernel - 1) * (2 * k_heads * k_dim + v_heads * v_dim)
    per_seq_bytes = linear_layers * (recurrent + conv) * state_bytes / tensor_parallel_size
    formula = (
        f"{linear_layers} linear-attention layers x ({recurrent} recurrent + {conv} conv) "
        f"elements x {state_bytes} bytes; x {max_num_seqs} sequences (--max-num-seqs)"
    )
    return VramPart("Linear-attention state", per_seq_bytes * max_num_seqs / _GIB, formula)
