"""Translates a Recipe into a `vllm serve` invocation. Assumes `vllm` is
already installed in the current environment - fllame does not vendor or
install it.
"""

from __future__ import annotations

from fllame.domain.recipe import Recipe


class VllmServingBackend:
    name = "vllm"

    def build_argv(self, recipe: Recipe, *, port: int) -> list[str]:
        return ["vllm", "serve", recipe.repo_id, "--port", str(port), *recipe.serve_args]
