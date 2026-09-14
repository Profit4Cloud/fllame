"""Persisted CLI settings - today just `default_image` - managed via
`fllame config`, not hand-edited.
"""

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
