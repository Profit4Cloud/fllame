"""Compiles one recipe into its own `compose.yaml` - fully overwritten
on every command that needs it, not hand-edited under normal use.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from fllame.backends.base import ServingBackend
from fllame.domain.hardware import HardwareProfile
from fllame.domain.recipe import Recipe
from fllame.models.sizing import DEFAULT_SIZING_CONFIG, SizingConfig


def generate_compose(
    recipe: Recipe,
    *,
    backend: ServingBackend,
    hf_cache_dir: Path,
    hardware: HardwareProfile,
    sizing_config: SizingConfig = DEFAULT_SIZING_CONFIG,
) -> dict:
    service = backend.build_service(
        recipe, hf_cache_dir=hf_cache_dir, hardware=hardware, sizing_config=sizing_config
    )
    return {"services": {recipe.handle: service}}


def write_compose_file(compose: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(compose, sort_keys=False))
