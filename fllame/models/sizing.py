"""A coarse memory budget for a machine's hardware, used only to default
`fllame model scan`'s `--max-size` ceiling, not a guarantee a model
actually fits when served. This is deliberately not the recipe-level
estimator (params + quantization + max-model-len + concurrency, for
weights and KV cache both) tracked as still-deferred in CLAUDE.md - that
one is for confirming a specific recipe fits; this one is for pointing a
search in the right direction at all. `fllame/models/discovery.py`
compares each candidate's own Hub-reported byte size directly against
this budget, so - unlike an earlier version of this module - there's no
per-quantization bytes-per-param guess here: a candidate's real,
already-quantized size is known directly from Hub metadata, and needs no
conversion.
"""

from __future__ import annotations

from fllame.domain.hardware import HardwareProfile

# Reserved for the host OS and vLLM's own non-weight overhead before
# estimating max model size on unified memory, where that overhead comes
# out of the same pool the model weights do. Discrete VRAM needs no such
# reserve - it isn't shared with the host at all.
_UNIFIED_MEMORY_OS_RESERVE_GB = 8.0

# Safety margin on top of whatever's left, covering KV cache/activation
# slop this estimate doesn't otherwise account for.
_MEMORY_SAFETY_MARGIN = 0.85


def memory_budget_gb(profile: HardwareProfile) -> float | None:
    """The single-GPU memory pool `usable_memory_gb` should size against
    - `vram_gb_per_gpu` for a discrete GPU, `ram_gb` for a unified-memory
    chip (grace_blackwell) since there's no separate VRAM figure to
    report there. `None` if neither is known. Sizes against one GPU's
    worth of memory even when more than one is present - a multi-GPU
    tensor-parallel budget is a different, harder question this doesn't
    attempt.
    """
    if profile.chip_family == "grace_blackwell":
        return profile.ram_gb
    if profile.vram_gb_per_gpu is not None:
        return profile.vram_gb_per_gpu
    return profile.ram_gb


def usable_memory_gb(*, budget_gb: float, unified_memory: bool) -> float:
    """`budget_gb`, minus the unified-memory OS reserve where it
    applies, minus the general safety margin - the default
    `fllame model scan --max-size` when the flag isn't given explicitly.
    """
    usable_gb = max(0.0, budget_gb - _UNIFIED_MEMORY_OS_RESERVE_GB) if unified_memory else budget_gb
    return usable_gb * _MEMORY_SAFETY_MARGIN
