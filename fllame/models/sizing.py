"""A coarse memory budget, used only to default `model scan`'s
`--max-size` - not a guarantee a model actually fits when served.
"""

from __future__ import annotations

from fllame.domain.hardware import HardwareProfile

# On unified memory, host OS/vLLM overhead comes out of the same pool
# as model weights, unlike discrete VRAM.
_UNIFIED_MEMORY_OS_RESERVE_GB = 8.0

# Covers KV cache/activation slop this estimate doesn't account for.
_MEMORY_SAFETY_MARGIN = 0.85


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
