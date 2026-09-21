"""Translates a Recipe into a docker-compose service definition."""

from __future__ import annotations

from pathlib import Path

from fllame.domain.hardware import HardwareProfile
from fllame.domain.recipe import Recipe
from fllame.domain.vllm_command import extract_flag_value, has_flag

_CONTAINER_HF_HOME = "/root/.cache/huggingface"

_GPU_MEMORY_UTILIZATION_FLAG = "--gpu-memory-utilization"
_TENSOR_PARALLEL_SIZE_FLAG = "--tensor-parallel-size"


def validate_gpu_memory_utilization(service: dict) -> str | None:
    """None when `service`'s own `command` already carries a present,
    parseable, in-range --gpu-memory-utilization - checked directly
    against compose.yaml as it stands on disk at `serve` time,
    independent of whether `recipe build` last wrote it correctly,
    since compose.yaml is explicitly meant to be hand-editable and this
    one value is safety-critical enough (an unset value can let vLLM
    claim a unified-memory machine's *entire* memory pool) to never
    simply trust because it was there before.

    Only rejects what's *obviously* wrong (missing, unparseable, <= 0,
    or > 1 - utilization can't exceed 100%, a mathematical fact, not a
    policy opinion) - an explicit, in-range recipe value is otherwise
    trusted outright, whatever it is.
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

    if parsed <= 0 or parsed > 1.0:
        return (
            f"refusing to serve: compose.yaml's --gpu-memory-utilization ({parsed}) is an "
            "obviously invalid value - it must be greater than 0 and no more than 1. Fix it by "
            "hand, or run `fllame recipe build` to regenerate compose.yaml."
        )

    return None


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
        self, recipe: Recipe, *, hf_cache_dir: Path, default_gpu_memory_utilization: float
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
            # The recipe's own value, if it set one, is trusted outright
            # - fllame never second-guesses it here (see
            # `validate_gpu_memory_utilization` for the one check that
            # still applies, at `serve` time, against obviously-broken
            # values regardless of where they came from).
            serve_args += [
                _GPU_MEMORY_UTILIZATION_FLAG,
                f"{default_gpu_memory_utilization:.2f}",
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
