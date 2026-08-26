"""A coarse, per-quantization ceiling on model size for a memory budget -
used only to narrow `fllame model scan`'s search, not a guarantee a
model actually fits when served. This is deliberately not the
recipe-level estimator (params + quantization + max-model-len +
concurrency, for weights and KV cache both) tracked as still-deferred in
CLAUDE.md - that one is for confirming a specific recipe fits; this one
is for pointing a search in the right direction at all. Adapted from
Profit4Cloud's brainzz-documents admin UI, trimmed to fllame's
vLLM-only scope (no training/LoRA headroom, no llama.cpp quantizations).
"""

from __future__ import annotations

from fllame.domain.hardware import HardwareProfile

_BYTES_PER_PARAM: dict[str, float] = {
    "awq": 0.5,
    "gptq": 0.5,
    "fp8": 1.0,
    "fp4": 0.5,
    "nvfp4": 0.5,
}
_DEFAULT_BYTES_PER_PARAM = 1.0

# Reserved for the host OS and vLLM's own non-weight overhead before
# estimating max model size on unified memory, where that overhead comes
# out of the same pool the model weights do. Discrete VRAM needs no such
# reserve - it isn't shared with the host at all.
_UNIFIED_MEMORY_OS_RESERVE_GB = 8.0

# Safety margin on top of whatever's left, covering KV cache/activation
# slop this estimate doesn't otherwise account for.
_MEMORY_SAFETY_MARGIN = 0.85


def memory_budget_gb(profile: HardwareProfile) -> float | None:
    """The single-GPU memory pool `max_params_billion` should size
    against - `vram_gb_per_gpu` for a discrete GPU, `ram_gb` for a
    unified-memory chip (grace_blackwell) since there's no separate VRAM
    figure to report there. `None` if neither is known. Sizes against
    one GPU's worth of memory even when more than one is present - a
    multi-GPU tensor-parallel budget is a different, harder question
    this doesn't attempt.
    """
    if profile.chip_family == "grace_blackwell":
        return profile.ram_gb
    if profile.vram_gb_per_gpu is not None:
        return profile.vram_gb_per_gpu
    return profile.ram_gb


def max_params_billion(*, budget_gb: float, unified_memory: bool, quantization: str) -> float:
    usable_gb = max(0.0, budget_gb - _UNIFIED_MEMORY_OS_RESERVE_GB) if unified_memory else budget_gb
    bytes_per_param = _BYTES_PER_PARAM.get(quantization.lower(), _DEFAULT_BYTES_PER_PARAM)
    return (usable_gb * _MEMORY_SAFETY_MARGIN) / bytes_per_param
