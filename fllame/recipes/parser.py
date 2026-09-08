"""Parses a pasted `export KEY=VALUE` + `RUN <command>` +
`vllm serve <repo_id> <args...>` block - the shape a recipe typically
comes in when copied from a model card or vLLM's own docs - into the
pieces a Recipe needs. Rejects anything that isn't literally one of
those three line shapes.

`export`/`vllm serve` values are rejected if they contain a shell
metacharacter: those are meant to become a single argv token/env var
value, so fllame parses this text itself rather than handing it to a
real shell, and a pasted `$(cat /etc/passwd)` there must never become a
literal wrong string silently baked into a recipe. A `RUN` line is
different in kind - it's meant to genuinely be a shell command (a
preinstall step run before `vllm serve`, e.g. `RUN pip install -U
transformers`) - so it's stored and later run verbatim, not rejected for
containing shell syntax that would be perfectly legitimate there.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

_EXPORT_PATTERN = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_RUN_PATTERN = re.compile(r"^RUN\s+(.+)$")
_FLAG_PATTERN = re.compile(r"^--[A-Za-z][\w-]*(=.*)?$")
_DANGEROUS_CHARS = re.compile(r"[;&|`\n\r]|\$\(")


class RecipePasteError(ValueError):
    """A pasted recipe block couldn't be parsed, or contained something
    fllame won't evaluate (shell substitution, command chaining, ...).
    """


@dataclass(frozen=True)
class ParsedRecipe:
    repo_id: str
    env: dict[str, str] = field(default_factory=dict)
    serve_args: list[str] = field(default_factory=list)
    preinstall: list[str] = field(default_factory=list)


def parse_pasted_recipe(text: str) -> ParsedRecipe:
    env: dict[str, str] = {}
    repo_id: str | None = None
    serve_args: list[str] = []
    preinstall: list[str] = []

    for raw_line in text.splitlines():
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
            if repo_id is not None:
                raise RecipePasteError("more than one `vllm serve` line - paste exactly one")
            tokens = _split_line("`vllm serve` line", line)
            if len(tokens) < 3:
                raise RecipePasteError("`vllm serve` line is missing a model repo id")
            repo_id = tokens[2]
            serve_args = tokens[3:]
            continue

        raise RecipePasteError(
            "unrecognized line (only `export KEY=VALUE` lines, `RUN <command>` "
            f"lines, and one `vllm serve ...` line are accepted): {line!r}"
        )

    if repo_id is None:
        raise RecipePasteError("no `vllm serve <repo_id> ...` line found in the paste")

    for token in serve_args:
        if token.startswith("-") and not _FLAG_PATTERN.match(token):
            raise RecipePasteError(f"malformed flag: {token!r}")

    return ParsedRecipe(repo_id=repo_id, env=env, serve_args=serve_args, preinstall=preinstall)


def _parse_single_token(where: str, value: str) -> str:
    tokens = _split_line(where, value)
    if len(tokens) != 1:
        raise RecipePasteError(
            f"{where}: value must be a single token (quote it if it has spaces): {value!r}"
        )
    return tokens[0]


def _split_line(where: str, text: str) -> list[str]:
    if _DANGEROUS_CHARS.search(text):
        raise RecipePasteError(
            f"{where}: contains a character fllame won't evaluate (;, &, |, `, or $(...)) "
            "- paste a literal value, not a shell command/substitution"
        )
    try:
        return shlex.split(text)
    except ValueError as e:
        raise RecipePasteError(f"{where}: unbalanced quoting: {e}") from e
