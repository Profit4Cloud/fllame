"""The seam a future non-vLLM backend would implement. Exactly one
implementation exists today (`VllmServingBackend`) - see CLAUDE.md,
"Explicitly deferred".
"""

from __future__ import annotations

from typing import Protocol

from fllame.domain.recipe import Recipe


class ServingBackend(Protocol):
    name: str

    def build_argv(self, recipe: Recipe, *, port: int) -> list[str]:
        """The command line to launch this recipe, as argv."""
        ...
