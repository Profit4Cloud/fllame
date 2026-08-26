"""Detects GPU(s) and RAM on the local machine via `nvidia-smi` and
`/proc/meminfo`. NVIDIA/CUDA only, matching fllame's vLLM-only scope - see
CLAUDE.md, "Explicitly deferred".
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from fllame.domain.hardware import HardwareProfile

# GPU name substrings for NVIDIA's unified-memory Grace-Blackwell chips,
# which support additional quantizations (fp4/nvfp4) generic NVIDIA GPUs
# don't.
_GRACE_BLACKWELL_MARKERS = ("GB10", "GB200")

_BASE_QUANTIZATIONS = ["awq", "gptq", "fp8"]
_GRACE_BLACKWELL_EXTRA_QUANTIZATIONS = ["fp4", "nvfp4"]

_DEFAULT_MEMINFO_PATH = Path("/proc/meminfo")


def scan_hardware() -> HardwareProfile:
    gpu_info = _scan_nvidia_gpus()
    ram_gb = _ram_gb()
    scanned_at = datetime.now(UTC).isoformat()

    if gpu_info is None:
        return HardwareProfile(
            gpu_name=None,
            gpu_count=0,
            vram_gb_per_gpu=None,
            ram_gb=ram_gb,
            chip_family="none",
            supported_quantizations=[],
            scanned_at=scanned_at,
        )

    name, count, vram_gb_per_gpu = gpu_info
    chip_family = _chip_family(name)
    return HardwareProfile(
        gpu_name=name,
        gpu_count=count,
        vram_gb_per_gpu=vram_gb_per_gpu,
        ram_gb=ram_gb,
        chip_family=chip_family,
        supported_quantizations=_supported_quantizations(chip_family),
        scanned_at=scanned_at,
    )


def _chip_family(gpu_name: str) -> str:
    if any(marker in gpu_name for marker in _GRACE_BLACKWELL_MARKERS):
        return "grace_blackwell"
    return "nvidia"


def _supported_quantizations(chip_family: str) -> list[str]:
    if chip_family == "grace_blackwell":
        return [*_BASE_QUANTIZATIONS, *_GRACE_BLACKWELL_EXTRA_QUANTIZATIONS]
    return list(_BASE_QUANTIZATIONS)


def _scan_nvidia_gpus() -> tuple[str, int, float] | None:
    """The first GPU's name and per-GPU VRAM, plus how many GPUs were
    found. Assumes a homogeneous set of GPUs, same as `nvidia-smi`'s
    per-line output gives no cheaper way to summarize a mixed one.
    """
    if shutil.which("nvidia-smi") is None:
        return None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    name, mem_mib = (part.strip() for part in lines[0].split(","))
    return name, len(lines), float(mem_mib) / 1024


def _ram_gb(meminfo_path: Path = _DEFAULT_MEMINFO_PATH) -> float | None:
    try:
        text = meminfo_path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            kib = int(line.split()[1])
            return kib / (1024 * 1024)
    return None
