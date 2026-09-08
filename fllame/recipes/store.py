"""Loads Recipes from a directory of hand-edited YAML files. The recipe
directory is meant to live in the operator's own git repo, not fllame's.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from fllame.domain.recipe import Recipe, RecipeError


def autofix_whitespace(text: str) -> str:
    r"""A narrow cleanup pass for the most common accidental YAML
    breakage from a text editor - CRLF line endings, tab indentation
    (which YAML forbids outright - most often introduced when an
    editor auto-indents a pasted block with tabs standing in for
    spaces at the same intended depth, the case this is really for),
    and trailing whitespace. Deliberately not a structural fix that
    changes a key's indentation *level* to what it "should" be - that
    would mean guessing the file's intended nesting. Full schema
    validation (`Recipe.from_dict`) still runs after this either way,
    so the rare case where tab-expansion happens to shift structure
    (e.g. a stray leading tab on an otherwise unindented line) is still
    caught rather than silently accepted - this isn't a guarantee of a
    correct parse, just a better shot at one before giving up.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(2)
    return "\n".join(line.rstrip() for line in text.split("\n"))


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
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            raise RecipeError(f"recipe '{handle}': invalid YAML: {e}") from e
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise RecipeError(
                f"recipe '{handle}': must be a YAML mapping (key: value pairs), "
                f"got {type(data).__name__}"
            )
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
