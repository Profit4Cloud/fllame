"""Translates a Recipe into a docker-compose service definition."""

from __future__ import annotations

import shlex
from pathlib import Path

from fllame.domain.recipe import Recipe

_CONTAINER_HF_HOME = "/root/.cache/huggingface"


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

    def build_service(self, recipe: Recipe, *, hf_cache_dir: Path) -> dict:
        # HF_HUB_CACHE set explicitly: huggingface_hub otherwise derives
        # it as HF_HOME/hub, one level below where the volume actually
        # mounts, and finds nothing.
        env = {
            "HF_HOME": _CONTAINER_HF_HOME,
            "HF_HUB_CACHE": _CONTAINER_HF_HOME,
            "HF_HUB_OFFLINE": "1",
            **recipe.env,
        }
        service: dict = {
            "image": recipe.image,
            "ports": [f"{recipe.port}:{recipe.port}"],
            "environment": [f"{key}={value}" for key, value in env.items()],
            "volumes": [f"{_host_volume_source(hf_cache_dir)}:{_CONTAINER_HF_HOME}"],
            "ipc": "host",
            "gpus": "all",
        }

        vllm_command = shlex.join(["vllm", "serve", recipe.repo_id, *recipe.serve_args])

        if recipe.preinstall:
            # `recipe.preinstall` entries are joined in verbatim, as
            # shell text; `vllm_command` is shlex-escaped so a repo_id/
            # flag value containing shell syntax stays a literal
            # argument. `exec` replaces the shell with vLLM so it
            # becomes PID 1 and receives SIGTERM directly.
            segments = [*recipe.preinstall, f"exec {vllm_command}"]
            service["entrypoint"] = ["sh", "-c"]
            service["command"] = [" && ".join(segments)]
        else:
            service["entrypoint"] = ["vllm", "serve"]
            service["command"] = [recipe.repo_id, *recipe.serve_args]

        return service
