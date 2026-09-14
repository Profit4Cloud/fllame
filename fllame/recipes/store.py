"""Loads Recipes from a directory of hand-edited YAML files, one
`recipe.yaml` per handle's own subfolder.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from fllame.domain.recipe import Recipe, RecipeError
from fllame.domain.vllm_command import join_command_lines

# Matches an unindented `key:` line - where a `command:` section ends
# if not followed by a blank line first.
_TOP_LEVEL_KEY = re.compile(r"^[A-Za-z_][\w-]*:(\s|$)")
_QUOTED = re.compile(r"^(['\"]).*\1$")
_BLOCK_SCALAR_INDICATOR = re.compile(r"^[|>][+-]?$")


def _extract_command_section(text: str) -> tuple[str, str | None]:
    """Splits raw recipe text into (everything else, `command`'s raw
    value), never handing `command` to `yaml.safe_load`: its rendering
    is a YAML literal block scalar, which needs consistent indentation
    on every line to stay valid at all - easy to break by hand.
    Extracting and parsing it with `join_command_lines` instead sidesteps
    that fragility, at the cost of `command` not being interruptible by
    another field without a blank line first.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("command:")), None)
    if start is None:
        return text, None

    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.strip() or _TOP_LEVEL_KEY.match(line):
            end = i
            break

    command_lines = lines[start:end]
    remaining = lines[:start] + lines[end:]
    return "\n".join(remaining), _parse_lenient_command(command_lines)


def _parse_lenient_command(command_lines: list[str]) -> str:
    first_line_value = command_lines[0][len("command:") :].strip()
    rest = command_lines[1:]

    if first_line_value == "" or _BLOCK_SCALAR_INDICATOR.match(first_line_value):
        content_lines = rest
    else:
        if _QUOTED.match(first_line_value):
            first_line_value = first_line_value[1:-1]
        content_lines = [first_line_value, *rest]

    return join_command_lines(content_lines)


def autofix_whitespace(text: str) -> str:
    """Fixes CRLF endings, tab indentation (which YAML forbids), and
    trailing whitespace - not a structural fix that guesses a key's
    intended indentation *level*. `Recipe.from_dict` still validates
    afterward either way."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(2)
    return "\n".join(line.rstrip() for line in text.split("\n"))


class RecipeStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def list_handles(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(p.parent.name for p in self.directory.glob("*/recipe.yaml"))

    def load(self, handle: str) -> Recipe:
        path = self.directory / handle / "recipe.yaml"
        if not path.is_file():
            raise RecipeError(f"no recipe found for '{handle}' (expected {path})")

        yaml_text, command = _extract_command_section(path.read_text())
        try:
            data = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            raise RecipeError(f"recipe '{handle}': invalid YAML: {e}") from e
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise RecipeError(
                f"recipe '{handle}': must be a YAML mapping (key: value pairs), "
                f"got {type(data).__name__}"
            )
        if command is not None:
            data["command"] = command
        return Recipe.from_dict(handle, data)

    def load_all(self) -> list[Recipe]:
        return [self.load(handle) for handle in self.list_handles()]

    def next_available_handle(self, base_handle: str) -> str:
        """`base_handle` itself if free, else `_2`, `_3`, ... - never
        overwrites, even for a second recipe on the same repo_id."""
        if not (self.directory / base_handle / "recipe.yaml").is_file():
            return base_handle
        n = 2
        while (self.directory / f"{base_handle}_{n}" / "recipe.yaml").is_file():
            n += 1
        return f"{base_handle}_{n}"

    def save(self, recipe: Recipe) -> None:
        directory = self.directory / recipe.handle
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "recipe.yaml").write_text(recipe.to_yaml())

    def remove(self, handle: str) -> None:
        directory = self.directory / handle
        path = directory / "recipe.yaml"
        if not path.is_file():
            raise RecipeError(f"no recipe found for '{handle}' (expected {path})")
        shutil.rmtree(directory)
