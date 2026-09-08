"""A small persisted settings file for cross-invocation CLI defaults -
today just `default_image`, the Docker image a recipe falls back to when
it doesn't pin its own (see `Recipe.image`, `fllame config`). Distinct
from `recipes_dir` (hand-edited recipe data) and `state_dir` (generated
compose state): this is fllame's own settings, meant to be managed via
`fllame config`, not hand-edited directly.
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
