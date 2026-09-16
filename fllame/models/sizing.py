"""A coarse memory budget, used only to default `model scan`'s
`--max-size` - not a guarantee a model actually fits when served.
"""

from __future__ import annotations

from fllame.domain.hardware import HardwareProfile
from fllame.models.architecture import ModelArchitecture

# On unified memory, host OS/vLLM overhead comes out of the same pool
# as model weights, unlike discrete VRAM.
_UNIFIED_MEMORY_OS_RESERVE_GB = 8.0

# Covers KV cache/activation slop this estimate doesn't account for.
_MEMORY_SAFETY_MARGIN = 0.85

# Concurrent full-speed requests to size --max-model-len's default
# against - not a --max-num-seqs cap or anything enforced at runtime,
# just the assumption behind this one default. Chosen from real
# agentic-workload testing: throughput held up to about this many
# concurrent max-length requests before degrading (still completes
# past it, just slower).
_DEFAULT_SIZING_CONCURRENCY = 8
# Below this, a "usable" context window isn't worth offering - too
# short to hold a real conversation or task, so --max-model-len is left
# unset (and recipe build hard-fails) rather than injecting it.
_MIN_USABLE_MAX_MODEL_LEN = 4096
# Coarse fixed allowance for activation memory and other overhead
# beyond weights + KV cache, deliberately not modeled per-architecture.
_ACTIVATION_OVERHEAD_GB = 2.0

_BYTES_PER_GB = 1e9
# fp16/bf16 KV cache dtype assumed: 2 bytes per element, times 2 for
# both K and V.
_KV_CACHE_DTYPE_BYTES = 2
_KV_TENSORS_PER_TOKEN = 2


def memory_budget_gb(profile: HardwareProfile) -> float | None:
    """Sizes against one GPU's worth of memory even with several
    present - a multi-GPU tensor-parallel budget is a different
    question."""
    if profile.chip_family == "grace_blackwell":
        return profile.ram_gb
    if profile.vram_gb_per_gpu is not None:
        return profile.vram_gb_per_gpu
    return profile.ram_gb


def usable_memory_gb(*, budget_gb: float, unified_memory: bool) -> float:
    usable_gb = max(0.0, budget_gb - _UNIFIED_MEMORY_OS_RESERVE_GB) if unified_memory else budget_gb
    return usable_gb * _MEMORY_SAFETY_MARGIN


def kv_cache_bytes_per_token(arch: ModelArchitecture) -> int:
    return (
        _KV_TENSORS_PER_TOKEN
        * arch.num_layers
        * arch.num_kv_heads
        * arch.head_dim
        * _KV_CACHE_DTYPE_BYTES
    )


def max_context_length_for_budget(
    *,
    arch: ModelArchitecture,
    weights_gb: float,
    total_budget_gb: float,
    concurrency: int = _DEFAULT_SIZING_CONCURRENCY,
) -> int:
    """The largest max-model-len that fits `total_budget_gb` (already
    the *effective* gpu-memory-utilization fraction times the box's raw
    memory - the caller has already done that multiplication) after
    `weights_gb` and a fixed activation/overhead allowance, sized for
    `concurrency` simultaneous max-length sequences - never above the
    model's own true architectural ceiling (`arch.max_context_length`),
    never negative.
    """
    kv_cache_budget_gb = total_budget_gb - weights_gb - _ACTIVATION_OVERHEAD_GB
    if kv_cache_budget_gb <= 0:
        return 0

    max_tokens_total = kv_cache_budget_gb * _BYTES_PER_GB / kv_cache_bytes_per_token(arch)
    context_length = int(max_tokens_total / concurrency)
    return max(0, min(context_length, arch.max_context_length))
