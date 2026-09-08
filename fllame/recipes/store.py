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

    def next_available_handle(self, base_handle: str) -> str:
        """`base_handle` itself if free, else `base_handle`_2, _3, ...
        Never overwrites an existing recipe - not even one for the same
        repo_id, since a second recipe for the same model (a different
        quantization, a different command tuning) is a legitimate,
        separate thing to keep.
        """
        if not (self.directory / f"{base_handle}.yaml").is_file():
            return base_handle
        n = 2
        while (self.directory / f"{base_handle}_{n}.yaml").is_file():
            n += 1
        return f"{base_handle}_{n}"

    def save(self, recipe: Recipe) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{recipe.handle}.yaml"
        path.write_text(recipe.to_yaml())

    def remove(self, handle: str) -> None:
        path = self.directory / f"{handle}.yaml"
        if not path.is_file():
            raise RecipeError(f"no recipe found for '{handle}' (expected {path})")
        path.unlink()
