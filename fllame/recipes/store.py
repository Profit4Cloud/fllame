"""Loads Recipes from a directory of hand-edited YAML files. The recipe
directory is meant to live in the operator's own git repo, not fllame's.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from fllame.domain.recipe import Recipe, RecipeError


class RecipeStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def list_handles(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(p.stem for p in self.directory.glob("*.yaml"))

    def load(self, handle: str) -> Recipe:
        path = self.directory / f"{handle}.yaml"
        if not path.is_file():
            raise RecipeError(f"no recipe found for '{handle}' (expected {path})")
        data = yaml.safe_load(path.read_text()) or {}
        return Recipe.from_dict(handle, data)

    def load_all(self) -> list[Recipe]:
        return [self.load(handle) for handle in self.list_handles()]
