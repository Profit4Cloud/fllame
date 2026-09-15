"""Translates a Recipe into a docker-compose service definition."""

from __future__ import annotations

from pathlib import Path

from fllame.domain.hardware import HardwareProfile
from fllame.domain.recipe import Recipe
from fllame.domain.vllm_command import has_flag

_CONTAINER_HF_HOME = "/root/.cache/huggingface"

_GPU_MEMORY_UTILIZATION_FLAG = "--gpu-memory-utilization"
# Never 1.0 - vLLM given the whole budget leaves no room for driver/CUDA
# context overhead. Matches vLLM's own out-of-the-box default rather
# than pushing closer to 1.0, since that default is already the
# commonly-run-safe figure across discrete GPUs, not just a placeholder.
_MAX_GPU_MEMORY_UTILIZATION = 0.92
# Unlike discrete VRAM, unified memory is shared with the OS and every
# other process on the box - a flat percentage isn't enough headroom on
# a small system and is needlessly conservative on a large one, so this
# is a fixed reservation instead.
_UNIFIED_MEMORY_SYSTEM_RESERVE_GB = 5.0


def default_gpu_memory_utilization(profile: HardwareProfile) -> float:
    """How much of the GPU's memory to hand vLLM when the recipe's own
    command doesn't already set `--gpu-memory-utilization` - left
    unset, vLLM defaults to reserving the memory itself, which on a
    unified-memory chip means the OS as a whole, not just the model,
    starves.

    A discrete GPU gets the flat cap outright. Unified memory instead
    reserves a fixed amount for the OS/everything else running on the
    box, still never above the same flat cap (a very large
    unified-memory system's reserved fraction can otherwise exceed it).
    """
    if profile.chip_family == "grace_blackwell" and profile.ram_gb:
        reserved_fraction = (profile.ram_gb - _UNIFIED_MEMORY_SYSTEM_RESERVE_GB) / profile.ram_gb
        return min(_MAX_GPU_MEMORY_UTILIZATION, max(0.0, reserved_fraction))
    return _MAX_GPU_MEMORY_UTILIZATION


def _host_volume_source(hf_cache_dir: Path) -> str:
    """`${HOME}/...`-relative when possible, so the generated
    compose.yaml still works after being copied to another machine or
    account - literal only when `hf_cache_dir` sits outside the home
    directory entirely (no portable relative form)."""
    try:
        relative = hf_cache_dir.relative_to(Path.home())
    except ValueError:
        return str(hf_cache_dir)
    return "${HOME}" if str(relative) == "." else f"${{HOME}}/{relative.as_posix()}"


def cache_volume_host_path(service: dict) -> str | None:
    """The host side of `service`'s HF cache bind mount, or `None`."""
    for volume in service.get("volumes", []):
        host, _, container = volume.partition(":")
        if container == _CONTAINER_HF_HOME:
            return host
    return None


class VllmServingBackend:
    name = "vllm"

    def build_service(
        self, recipe: Recipe, *, hf_cache_dir: Path, hardware: HardwareProfile
    ) -> dict:
        # HF_HUB_CACHE set explicitly: huggingface_hub otherwise derives
        # it as HF_HOME/hub, one level below where the volume actually
        # mounts, and finds nothing.
        env = {
            "HF_HOME": _CONTAINER_HF_HOME,
            "HF_HUB_CACHE": _CONTAINER_HF_HOME,
            "HF_HUB_OFFLINE": "1",
            **recipe.env,
        }
        serve_args = list(recipe.serve_args)
        if not has_flag(serve_args, _GPU_MEMORY_UTILIZATION_FLAG):
            serve_args += [
                _GPU_MEMORY_UTILIZATION_FLAG,
                f"{default_gpu_memory_utilization(hardware):.2f}",
            ]
        service: dict = {
            "image": recipe.image,
            "entrypoint": ["vllm", "serve"],
            "command": [recipe.repo_id, *serve_args],
            "ports": [f"{recipe.port}:{recipe.port}"],
            "environment": [f"{key}={value}" for key, value in env.items()],
            "volumes": [f"{_host_volume_source(hf_cache_dir)}:{_CONTAINER_HF_HOME}"],
            "ipc": "host",
            "gpus": "all",
        }

        return service


def generate_dockerfile(recipe: Recipe) -> str | None:
    """`FROM <image>` plus one `RUN <preinstall line>` per entry, in
    order - never joined with `&&` into a single `RUN`, so an unchanged
    earlier step stays cache-hit on a later rebuild even if a later one
    changes. `None` when there's nothing to install: a no-op Dockerfile
    would only obscure that this recipe runs the base image verbatim.
    """
    if not recipe.preinstall:
        return None
    lines = [f"FROM {recipe.image}", *(f"RUN {step}" for step in recipe.preinstall)]
    return "\n".join(lines) + "\n"
