"""Persisted CLI settings, managed via `fllame config`, not hand-edited."""

from __future__ import annotations

import yaml

from fllame import config


def _read() -> dict:
    path = config.config_file_path()
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _write(data: dict) -> None:
    path = config.config_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def get_default_image() -> str | None:
    return _read().get("default_image")


def set_default_image(image: str) -> None:
    data = _read()
    data["default_image"] = image
    _write(data)


def get_min_usable_max_model_len() -> int | None:
    return _read().get("min_usable_max_model_len")


def set_min_usable_max_model_len(tokens: int) -> None:
    data = _read()
    data["min_usable_max_model_len"] = tokens
    _write(data)


def get_activation_overhead_gb() -> float | None:
    return _read().get("activation_overhead_gb")


def set_activation_overhead_gb(gb: float) -> None:
    data = _read()
    data["activation_overhead_gb"] = gb
    _write(data)


def get_max_gpu_memory_utilization() -> float | None:
    return _read().get("max_gpu_memory_utilization")


def set_max_gpu_memory_utilization(value: float) -> None:
    data = _read()
    data["max_gpu_memory_utilization"] = value
    _write(data)
