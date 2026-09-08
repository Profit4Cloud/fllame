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

    def render_dockerfile(self, recipe: Recipe) -> str | None:
        """Dockerfile content to build a custom image for this recipe,
        or `None` if it can run from its `image` directly - a recipe
        with no preinstall step needs no Dockerfile at all."""
        ...
