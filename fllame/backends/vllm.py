"""Translates a Recipe into a docker-compose service definition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fllame.domain.hardware import HardwareProfile
from fllame.domain.recipe import Recipe
from fllame.domain.vllm_command import extract_flag_value, has_flag
from fllame.models.architecture import read_architecture
from fllame.models.cache import local_estimate_vram_gb
from fllame.models.sizing import (
    _DEFAULT_SIZING_CONCURRENCY,
    DEFAULT_SIZING_CONFIG,
    SizingConfig,
    max_context_length_for_budget,
    memory_budget_gb,
)

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


_MAX_MODEL_LEN_FLAG = "--max-model-len"
_TENSOR_PARALLEL_SIZE_FLAG = "--tensor-parallel-size"


@dataclass(frozen=True)
class _MaxModelLenResolution:
    # What to inject into serve_args, or None: the recipe set its own,
    # the architecture is unsupported, or the hardware budget is
    # unknown - every "don't touch it" case looks the same to the
    # caller.
    value: int | None
    # A human-readable message only when a supported architecture's
    # minimum usable context genuinely doesn't fit - None in every
    # other case, including "unsupported/unknown".
    shortfall: str | None


def _effective_gpu_memory_utilization(recipe: Recipe, hardware: HardwareProfile) -> float:
    """Whatever value will actually end up on the vllm serve command
    line for --gpu-memory-utilization - the recipe's own explicit value
    if it set one, else the same computed default build_service would
    inject. The max-model-len budget must be sized against this same
    number, not recomputed independently, since they're not
    independent decisions."""
    explicit = extract_flag_value(recipe.serve_args, _GPU_MEMORY_UTILIZATION_FLAG)
    return float(explicit) if explicit is not None else default_gpu_memory_utilization(hardware)


def _resolve_max_model_len(
    recipe: Recipe, hardware: HardwareProfile, sizing_config: SizingConfig
) -> _MaxModelLenResolution:
    if has_flag(recipe.serve_args, _MAX_MODEL_LEN_FLAG):
        return _MaxModelLenResolution(value=None, shortfall=None)

    weights_gb = local_estimate_vram_gb(recipe.repo_id)
    arch = read_architecture(recipe.repo_id)
    if weights_gb is None or arch is None:
        return _MaxModelLenResolution(value=None, shortfall=None)

    budget_gb = memory_budget_gb(hardware)
    if budget_gb is None:
        return _MaxModelLenResolution(value=None, shortfall=None)

    utilization = _effective_gpu_memory_utilization(recipe, hardware)
    total_budget_gb = utilization * budget_gb
    context_length = max_context_length_for_budget(
        arch=arch,
        weights_gb=weights_gb,
        total_budget_gb=total_budget_gb,
        concurrency=_DEFAULT_SIZING_CONCURRENCY,
        activation_overhead_gb=sizing_config.activation_overhead_gb,
    )

    if context_length < sizing_config.min_usable_max_model_len:
        shortfall = (
            f"'{recipe.repo_id}' doesn't fit a usable context window on this hardware even "
            f"at fllame's minimum ({sizing_config.min_usable_max_model_len} tokens) - weights "
            f"alone need {weights_gb:.1f} GB, leaving too little of the {total_budget_gb:.1f} GB "
            f"budget (at --gpu-memory-utilization {utilization:.2f}) for KV cache at "
            f"{_DEFAULT_SIZING_CONCURRENCY}-way concurrency."
        )
        return _MaxModelLenResolution(value=None, shortfall=shortfall)

    return _MaxModelLenResolution(value=context_length, shortfall=None)


def max_model_len_shortfall(
    recipe: Recipe, hardware: HardwareProfile, sizing_config: SizingConfig = DEFAULT_SIZING_CONFIG
) -> str | None:
    """What `recipe build` hard-fails on - the same shortfall detection
    `build_service`'s own injection already skips silently, exposed
    separately so both paths share one source of truth."""
    return _resolve_max_model_len(recipe, hardware, sizing_config).shortfall


def tensor_parallel_size_mismatch_warning(recipe: Recipe, hardware: HardwareProfile) -> str | None:
    """None when there's nothing worth flagging (no GPU count detected
    at all, or the recipe's tensor-parallel-size already matches it).
    vLLM's own default is 1 when the recipe doesn't set one.

    `gpus: all` and no fllame-set `--tensor-parallel-size` are both
    unconditional - this only warns, it never changes either."""
    if hardware.gpu_count <= 0:
        return None

    value = extract_flag_value(recipe.serve_args, _TENSOR_PARALLEL_SIZE_FLAG) or "1"
    tensor_parallel_size = int(value)
    if tensor_parallel_size == hardware.gpu_count:
        return None

    if tensor_parallel_size > hardware.gpu_count:
        return (
            f"warning: '{recipe.handle}' sets --tensor-parallel-size {tensor_parallel_size}, "
            f"but only {hardware.gpu_count} GPU(s) were detected - vLLM will fail to start "
            "until this recipe's --tensor-parallel-size matches what's actually present."
        )
    return (
        f"warning: '{recipe.handle}' sets --tensor-parallel-size {tensor_parallel_size}, but "
        f"{hardware.gpu_count} GPU(s) were detected - `gpus: all` still hands the container "
        f"every one of them, so {hardware.gpu_count - tensor_parallel_size} will sit unused. "
        "Raise --tensor-parallel-size in the recipe's command to use them."
    )


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
        self,
        recipe: Recipe,
        *,
        hf_cache_dir: Path,
        hardware: HardwareProfile,
        sizing_config: SizingConfig = DEFAULT_SIZING_CONFIG,
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
        max_model_len = _resolve_max_model_len(recipe, hardware, sizing_config).value
        if max_model_len is not None:
            serve_args += [_MAX_MODEL_LEN_FLAG, str(max_model_len)]
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
