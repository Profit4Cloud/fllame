"""What one scan of the local machine found - a snapshot, not a live
handle. Nothing here is persisted: scanning is cheap enough that callers
(the CLI today, model-filtering/OOM-guard code later) just re-scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HardwareProfile:
    gpu_name: str | None
    gpu_count: int
    vram_gb_per_gpu: float | None
    ram_gb: float | None
    chip_family: str  # "grace_blackwell" | "nvidia" | "none"
    supported_quantizations: list[str] = field(default_factory=list)
    scanned_at: str = ""

    @property
    def has_gpu(self) -> bool:
        return self.gpu_name is not None
