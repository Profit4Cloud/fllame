"""Parses `recipe add` input: a `vllm serve` command given as arguments,
and the `KEY=VALUE` env lines its guided dialogue collects. Both reject
shell metacharacters - fllame parses them, never a real shell.
"""

from __future__ import annotations

import re
import shlex

from fllame.domain.vllm_command import (
    VllmCommandError,
    join_command_lines,
    parse_vllm_serve_command,
    split_shell_safe,
)

_ENV_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


class RecipePasteError(ValueError):
    pass


def parse_command_args(args: list[str]) -> str:
    """One logical `vllm serve` line from `recipe add`'s arguments. A
    single argument is a quoted paste and is taken as-is; re-quoting it
    with `shlex.join` would wrap the whole command in quotes. Several
    arguments are the shell's own split of an unquoted command."""
    text = args[0] if len(args) == 1 else shlex.join(args)
    command = join_command_lines(text.splitlines())
    if command != "vllm serve" and not command.startswith("vllm serve "):
        raise RecipePasteError(
            "Only a `vllm serve REPO_ID ...` command can be passed here. "
            "For a custom image, env vars or preinstall commands, run "
            "`fllame recipe add` without arguments."
        )
    try:
        parse_vllm_serve_command(command)
    except VllmCommandError as e:
        raise RecipePasteError(str(e)) from e
    return command


def parse_env_line(line: str) -> tuple[str, str]:
    """Bare `KEY=VALUE`, no `export` keyword - the shape `recipe add`'s
    guided dialogue collects env vars in."""
    match = _ENV_PATTERN.match(line)
    if not match:
        raise RecipePasteError(f"not a KEY=VALUE line: {line!r}")
    key, value = match.groups()
    try:
        tokens = split_shell_safe(f"env var '{key}'", value)
    except VllmCommandError as e:
        raise RecipePasteError(str(e)) from e
    if len(tokens) != 1:
        raise RecipePasteError(
            f"env var '{key}': value must be a single token (quote it if it has spaces): {value!r}"
        )
    return key, tokens[0]
