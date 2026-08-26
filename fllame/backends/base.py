"""The seam a future non-vLLM backend would implement. Exactly one
implementation exists today (`VllmServingBackend`) - see CLAUDE.md,
"Explicitly deferred".
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fllame.domain.recipe import Recipe


class ServingBackend(Protocol):
    name: str

    def build_service(self, recipe: Recipe, *, hf_cache_dir: Path) -> dict:
        """A docker-compose service definition (the value under
        `services.<handle>`) for running this recipe."""
        ...
