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
# Unlike discrete VRAM, unified memory is shared with the OS and every
# other process on the box - a flat percentage isn't enough headroom on
# a small system and is needlessly conservative on a large one, so this
# is a fixed reservation instead.
_UNIFIED_MEMORY_SYSTEM_RESERVE_GB = 5.0


def default_gpu_memory_utilization(
    profile: HardwareProfile, sizing_config: SizingConfig = DEFAULT_SIZING_CONFIG
) -> float:
    """How much of the GPU's memory to hand vLLM when the recipe's own
    command doesn't already set `--gpu-memory-utilization` - left
    unset, vLLM defaults to reserving the memory itself, which on a
    unified-memory chip means the OS as a whole, not just the model,
    starves.

    A discrete GPU gets `sizing_config.max_gpu_memory_utilization`
    outright. Unified memory instead reserves a fixed amount for the
    OS/everything else running on the box, still never above that same
    ceiling (a very large unified-memory system's reserved fraction can
    otherwise exceed it).
    """
    max_utilization = sizing_config.max_gpu_memory_utilization
    if profile.chip_family == "grace_blackwell" and profile.ram_gb:
        reserved_fraction = (profile.ram_gb - _UNIFIED_MEMORY_SYSTEM_RESERVE_GB) / profile.ram_gb
        return min(max_utilization, max(0.0, reserved_fraction))
    return max_utilization


_MAX_MODEL_LEN_FLAG = "--max-model-len"
_TENSOR_PARALLEL_SIZE_FLAG = "--tensor-parallel-size"


@dataclass(frozen=True)
class _MaxModelLenResolution:
    # What to inject into serve_args, or None: the recipe set its own,
    # the architecture is unsupported, the hardware budget is unknown,
    # or there's a shortfall - every "don't inject anything" case looks
    # the same to `build_service`.
    value: int | None
    # A human-readable message only when a supported architecture's
    # minimum usable context genuinely doesn't fit - None in every
    # other case, including "unsupported/unknown".
    shortfall: str | None
    # Informational only, never blocks anything: set whenever `value`
    # ends up None for a reason other than the recipe setting its own
    # --max-model-len (which is an expected, silent, no-note-needed
    # case) - otherwise "nothing got injected" and "why" are both
    # invisible, which is exactly what makes this hard to debug.
    note: str | None = None


@dataclass(frozen=True)
class _GpuMemoryUtilizationResolution:
    # Always set: the recipe's own valid explicit value, or a computed
    # default - `--max-model-len` sizing needs *some* number to size
    # against even when the recipe's own value turns out to be invalid
    # (0.0 in that case - the error below is what actually stops
    # anything from running with it).
    value: float
    # Non-None only when the recipe's own explicit value is unsafe/
    # unparseable - what `recipe build` hard-fails on.
    error: str | None


def _resolve_gpu_memory_utilization(
    recipe: Recipe, hardware: HardwareProfile, sizing_config: SizingConfig
) -> _GpuMemoryUtilizationResolution:
    explicit = extract_flag_value(recipe.serve_args, _GPU_MEMORY_UTILIZATION_FLAG)
    if explicit is None:
        return _GpuMemoryUtilizationResolution(
            value=default_gpu_memory_utilization(hardware, sizing_config), error=None
        )

    try:
        parsed = float(explicit)
    except ValueError:
        return _GpuMemoryUtilizationResolution(
            value=0.0,
            error=(
                f"'{recipe.handle}' sets --gpu-memory-utilization to '{explicit}', which isn't "
                "a number - fix it in the recipe's command."
            ),
        )

    if parsed <= 0 or parsed > sizing_config.max_gpu_memory_utilization:
        return _GpuMemoryUtilizationResolution(
            value=0.0,
            error=(
                f"'{recipe.handle}' sets --gpu-memory-utilization to {parsed}, outside the safe "
                f"range (0, {sizing_config.max_gpu_memory_utilization}] - a value this high can "
                "let vLLM claim this machine's entire GPU/unified memory pool. Lower it in the "
                "recipe's command, or raise fllame's own ceiling with `fllame config "
                "set-max-gpu-memory-utilization` if you're sure it's safe on this hardware."
            ),
        )

    return _GpuMemoryUtilizationResolution(value=parsed, error=None)


def gpu_memory_utilization_error(
    recipe: Recipe, hardware: HardwareProfile, sizing_config: SizingConfig = DEFAULT_SIZING_CONFIG
) -> str | None:
    """What `recipe build` hard-fails on when the recipe's own explicit
    --gpu-memory-utilization is unsafe or unparseable - `build_service`
    itself leaves that value untouched either way (it never silently
    overrides an explicit choice), so this is the only thing that
    actually stops such a recipe from being built."""
    return _resolve_gpu_memory_utilization(recipe, hardware, sizing_config).error


def validate_gpu_memory_utilization(
    service: dict, sizing_config: SizingConfig = DEFAULT_SIZING_CONFIG
) -> str | None:
    """None when `service`'s own `command` already carries a safe
    --gpu-memory-utilization - checked directly against compose.yaml as
    it stands on disk at `serve` time, independent of whether `recipe
    build` last wrote it correctly, since compose.yaml is explicitly
    meant to be hand-editable and this one value is safety-critical
    enough (an unset or too-high value can let vLLM claim a
    unified-memory machine's *entire* memory pool, not just crash its
    own container) to never simply trust because it was there before.
    """
    command = [str(token) for token in (service.get("command") or [])]
    value = extract_flag_value(command, _GPU_MEMORY_UTILIZATION_FLAG)

    if value is None:
        return (
            "refusing to serve: compose.yaml has no --gpu-memory-utilization set - an unset "
            "value can let vLLM claim this machine's entire GPU/unified memory pool. Run "
            "`fllame recipe build` to regenerate compose.yaml, or set one by hand."
        )

    try:
        parsed = float(value)
    except ValueError:
        return (
            f"refusing to serve: compose.yaml's --gpu-memory-utilization ('{value}') isn't a "
            "number. Fix it by hand, or run `fllame recipe build` to regenerate compose.yaml."
        )

    if parsed <= 0 or parsed > sizing_config.max_gpu_memory_utilization:
        return (
            f"refusing to serve: compose.yaml's --gpu-memory-utilization ({parsed}) is outside "
            f"the safe range (0, {sizing_config.max_gpu_memory_utilization}] - fix it by hand, "
            "run `fllame recipe build` to regenerate compose.yaml, or raise fllame's own "
            "ceiling with `fllame config set-max-gpu-memory-utilization` if you're sure it's "
            "safe on this hardware."
        )

    return None


def _resolve_max_model_len(
    recipe: Recipe, hardware: HardwareProfile, sizing_config: SizingConfig
) -> _MaxModelLenResolution:
    if has_flag(recipe.serve_args, _MAX_MODEL_LEN_FLAG):
        return _MaxModelLenResolution(value=None, shortfall=None)

    weights_gb = local_estimate_vram_gb(recipe.repo_id)
    if weights_gb is None:
        return _MaxModelLenResolution(
            value=None,
            shortfall=None,
            note=(
                f"note: couldn't determine '{recipe.repo_id}''s cached weight size (not fully "
                "cached, or no .safetensors files) - --max-model-len left unset, vLLM will use "
                "its own default."
            ),
        )

    arch = read_architecture(recipe.repo_id)
    if arch is None:
        return _MaxModelLenResolution(
            value=None,
            shortfall=None,
            note=(
                f"note: '{recipe.repo_id}''s config.json isn't in a recognized architecture "
                "shape - --max-model-len left unset, vLLM will use its own default."
            ),
        )

    budget_gb = memory_budget_gb(hardware)
    if budget_gb is None:
        return _MaxModelLenResolution(
            value=None,
            shortfall=None,
            note=(
                "note: this machine's GPU/RAM budget couldn't be determined - --max-model-len "
                "left unset, vLLM will use its own default."
            ),
        )

    utilization = _resolve_gpu_memory_utilization(recipe, hardware, sizing_config).value
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


def max_model_len_note(
    recipe: Recipe, hardware: HardwareProfile, sizing_config: SizingConfig = DEFAULT_SIZING_CONFIG
) -> str | None:
    """What `recipe build` prints (never blocks on) when --max-model-len
    couldn't be computed for a reason other than the recipe setting its
    own - without this, "nothing got injected" and "why" are both
    invisible, which makes a real skip indistinguishable from a bug."""
    return _resolve_max_model_len(recipe, hardware, sizing_config).note


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
                f"{default_gpu_memory_utilization(hardware, sizing_config):.2f}",
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
