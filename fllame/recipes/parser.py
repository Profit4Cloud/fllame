"""Parses a pasted `export KEY=VALUE` + `RUN <command>` +
`vllm serve <repo_id> <args...>` block - the shape a recipe typically
comes in when copied from a model card or vLLM's own docs - into the
pieces a Recipe needs. Rejects anything that isn't literally one of
those three line shapes.

The `vllm serve` line may span several physical lines with trailing
`\\` continuations, exactly as it's often shown on a model card or
recipes.vllm.ai - joined into one logical line before parsing (see
`domain.vllm_command.join_line_continuations`).

`export` values and the `vllm serve` line itself are rejected if they
contain a shell metacharacter: those are meant to become a single argv
token/env var value, so fllame parses this text itself rather than
handing it to a real shell, and a pasted `$(cat /etc/passwd)` there must
never become a literal wrong string silently baked into a recipe. A
`RUN` line is different in kind - it's meant to genuinely be a shell
command (a preinstall step run before `vllm serve`, e.g. `RUN pip
install -U transformers`) - so it's stored and later run verbatim, not
rejected for containing shell syntax that would be perfectly legitimate
there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from fllame.domain.vllm_command import (
    VllmCommandError,
    join_line_continuations,
    parse_vllm_serve_command,
    split_shell_safe,
)

_EXPORT_PATTERN = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_RUN_PATTERN = re.compile(r"^RUN\s+(.+)$")
_ENV_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


class RecipePasteError(ValueError):
    """A pasted recipe block couldn't be parsed, or contained something
    fllame won't evaluate (shell substitution, command chaining, ...).
    """


@dataclass(frozen=True)
class ParsedRecipe:
    repo_id: str
    command: str
    env: dict[str, str] = field(default_factory=dict)
    preinstall: list[str] = field(default_factory=list)


def parse_pasted_recipe(text: str) -> ParsedRecipe:
    env: dict[str, str] = {}
    repo_id: str | None = None
    command: str | None = None
    preinstall: list[str] = []

    for raw_line in join_line_continuations(text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        export_match = _EXPORT_PATTERN.match(line)
        if export_match:
            key, value = export_match.groups()
            env[key] = _parse_single_token(f"env var '{key}'", value)
            continue

        run_match = _RUN_PATTERN.match(line)
        if run_match:
            preinstall.append(run_match.group(1))
            continue

        if line == "vllm serve" or line.startswith("vllm serve "):
            if command is not None:
                raise RecipePasteError("more than one `vllm serve` line - paste exactly one")
            try:
                repo_id, _ = parse_vllm_serve_command(line)
            except VllmCommandError as e:
                raise RecipePasteError(str(e)) from e
            command = line
            continue

        raise RecipePasteError(
            "unrecognized line (only `export KEY=VALUE` lines, `RUN <command>` "
            f"lines, and one `vllm serve ...` line are accepted): {line!r}"
        )

    if command is None or repo_id is None:
        raise RecipePasteError("no `vllm serve <repo_id> ...` line found in the paste")

    return ParsedRecipe(repo_id=repo_id, command=command, env=env, preinstall=preinstall)


def parse_env_line(line: str) -> tuple[str, str]:
    """Parses a bare `KEY=VALUE` line - the shape `recipe add`'s guided
    dialogue collects env vars in, no `export` keyword needed since
    that step is only ever env vars, unlike the mixed-line paste grammar
    `parse_pasted_recipe` handles.
    """
    match = _ENV_PATTERN.match(line)
    if not match:
        raise RecipePasteError(f"not a KEY=VALUE line: {line!r}")
    key, value = match.groups()
    return key, _parse_single_token(f"env var '{key}'", value)


def _parse_single_token(where: str, value: str) -> str:
    try:
        tokens = split_shell_safe(where, value)
    except VllmCommandError as e:
        raise RecipePasteError(str(e)) from e
    if len(tokens) != 1:
        raise RecipePasteError(
            f"{where}: value must be a single token (quote it if it has spaces): {value!r}"
        )
    return tokens[0]
