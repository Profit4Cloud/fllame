"""Loads Recipes from a directory of hand-edited YAML files, one
`recipe.yaml` per handle's own subfolder (which also holds that
handle's generated `compose.yaml` - see `fllame/config.py`'s
`recipe_dir`). The recipe directory is meant to live in the operator's
own git repo, not fllame's.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from fllame.domain.recipe import Recipe, RecipeError
from fllame.domain.vllm_command import join_command_lines

# A top-level `key:` line - used to find where a `command:` section
# ends when it isn't followed by a blank line or EOF first (see
# `_extract_command_section`).
_TOP_LEVEL_KEY = re.compile(r"^[A-Za-z_][\w-]*:(\s|$)")
_QUOTED = re.compile(r"^(['\"]).*\1$")
_BLOCK_SCALAR_INDICATOR = re.compile(r"^[|>][+-]?$")


def _extract_command_section(text: str) -> tuple[str, str | None]:
    """Splits raw recipe file text into (everything else, the `command`
    field's raw value) at the first line starting with `command:`.

    `command` is deliberately never handed to `yaml.safe_load` as part
    of the rest of the document: its own multi-line rendering (see
    `Recipe.to_dict`) is a YAML literal block scalar, which requires
    consistent, sufficient indentation on every continuation line to
    remain valid YAML at all - an easy thing to break by hand (e.g.
    stripping what looks like meaningless leading whitespace) that
    would otherwise fail YAML parsing outright, not just this one
    field. Extracting it here and parsing it with `join_command_lines`
    instead sidesteps that fragility entirely: only its own line-based
    grammar applies, indentation and trailing `\\` continuations optional.

    The command section runs from the `command:` line to the next
    blank line, the next unindented `key:` line, or EOF - so `command`
    doesn't strictly have to be the last field (though `RecipeStore.save`
    always writes it last), just not interrupted by another field
    without a blank line separating them.
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
        """`base_handle` itself if free, else `base_handle`_2, _3, ...
        Never overwrites an existing recipe - not even one for the same
        repo_id, since a second recipe for the same model (a different
        quantization, a different command tuning) is a legitimate,
        separate thing to keep.
        """
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
        """Deletes HANDLE's whole folder - both `recipe.yaml` and
        whatever generated `compose.yaml` sits next to it, since the two
        live together (see `fllame/config.py`'s `recipe_dir`)."""
        directory = self.directory / handle
        path = directory / "recipe.yaml"
        if not path.is_file():
            raise RecipeError(f"no recipe found for '{handle}' (expected {path})")
        shutil.rmtree(directory)
