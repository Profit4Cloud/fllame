"""Persisted CLI settings, managed via `fllame config`, not hand-edited."""

from __future__ import annotations

import yaml

from fllame import config

# If --gpu-memory-utilization is not set, vLLM will use all available GPU memory, possibly
# crashing unified systems. Therefore, it is always injected into compose.yaml, either
# explicitly from the recipe, or the default below - so this setting is never left unset.
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.92

# `recipe build` pins `latest` to the release it points at, so following
# it never makes a built recipe drift.
_DEFAULT_IMAGE = "vllm/vllm-openai:latest"


def _read() -> dict:
    path = config.config_file_path()
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _write(data: dict) -> None:
    path = config.config_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def get_default_image() -> str:
    return _read().get("default_image", _DEFAULT_IMAGE)


def set_default_image(image: str) -> None:
    data = _read()
    data["default_image"] = image
    _write(data)


def get_default_gpu_memory_utilization() -> float:
    return _read().get("default_gpu_memory_utilization", _DEFAULT_GPU_MEMORY_UTILIZATION)


def set_default_gpu_memory_utilization(value: float) -> None:
    data = _read()
    data["default_gpu_memory_utilization"] = value
    _write(data)
