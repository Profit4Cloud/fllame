"""Seam for a future non-vLLM backend; only `VllmServingBackend` exists."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fllame.domain.hardware import HardwareProfile
from fllame.domain.recipe import Recipe
from fllame.models.sizing import DEFAULT_SIZING_CONFIG, SizingConfig


class ServingBackend(Protocol):
    name: str

    def build_service(
        self,
        recipe: Recipe,
        *,
        hf_cache_dir: Path,
        hardware: HardwareProfile,
        sizing_config: SizingConfig = DEFAULT_SIZING_CONFIG,
    ) -> dict:
        """The `services.<handle>` value for this recipe's compose.yaml."""
        ...
