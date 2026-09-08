"""Parsing and validation for a single `vllm serve <repo_id> <args...>`
command line - shared between `Recipe`'s own `command` field (see
`domain/recipe.py`) and the pasted-recipe parser
(`recipes/parser.py`), since both need the same answer to "is this a
well-formed vllm serve invocation, and what's its repo_id/args".

Tokenized with `shlex`, never handed to a real shell: this text only
ever becomes literal argv elements in a generated docker-compose
`command:` list, so a value like `; rm -rf /` is rejected outright
rather than risk it being mistaken for something fllame would evaluate.
"""

from __future__ import annotations

import re
import shlex

_FLAG_PATTERN = re.compile(r"^--[A-Za-z][\w-]*(=.*)?$")
_DANGEROUS_CHARS = re.compile(r"[;&|`\r]|\$\(")
# Matches a `\`-continued line break, including the surrounding
# indentation on both sides, so joining collapses it to exactly one
# space rather than leaving behind whatever indentation the original
# multi-line paste happened to use.
_LINE_CONTINUATION = re.compile(r"[ \t]*\\[ \t]*\n[ \t]*")


class VllmCommandError(ValueError):
    """A `vllm serve ...` command line is malformed, or contains
    something fllame won't evaluate (shell substitution, command
    chaining, ...).
    """


def join_line_continuations(text: str) -> str:
    """Collapses a `\\`-then-newline line continuation into a single
    space, the same way a real shell would - so a command copied
    verbatim from a model card or recipes.vllm.ai, backslashes and all,
    parses as the one logical line it represents.
    """
    return _LINE_CONTINUATION.sub(" ", text)


def split_shell_safe(where: str, text: str) -> list[str]:
    """Tokenizes `text` with `shlex`, rejecting shell metacharacters
    (`;`, `&`, `|`, `` ` ``, `$(...)`) first - used for anything that's
    meant to become a single argv token/env var value, not a real
    command (see `parse_vllm_serve_command` for the one case that's
    deliberately exempt from this).
    """
    if _DANGEROUS_CHARS.search(text):
        raise VllmCommandError(
            f"{where}: contains a character fllame won't evaluate (;, &, |, `, or $(...)) "
            "- paste a literal value, not a shell command/substitution"
        )
    try:
        return shlex.split(text)
    except ValueError as e:
        raise VllmCommandError(f"{where}: unbalanced quoting: {e}") from e


def parse_vllm_serve_command(command: str) -> tuple[str, list[str]]:
    """Parses a `vllm serve <repo_id> <args...>` string (line
    continuations joined first) into `(repo_id, serve_args)`.
    """
    joined = join_line_continuations(command).strip()
    if joined != "vllm serve" and not joined.startswith("vllm serve "):
        raise VllmCommandError(f"not a `vllm serve <repo_id> ...` command: {command!r}")

    tokens = split_shell_safe("`vllm serve` command", joined)
    if len(tokens) < 3:
        raise VllmCommandError("`vllm serve` command is missing a model repo id")

    repo_id = tokens[2]
    serve_args = tokens[3:]
    for token in serve_args:
        if token.startswith("-") and not _FLAG_PATTERN.match(token):
            raise VllmCommandError(f"malformed flag: {token!r}")
    return repo_id, serve_args


def extract_port(serve_args: list[str], *, default: int = 8000) -> int:
    """The `--port` value from a parsed `vllm serve` args list, falling
    back to `default` (vLLM's own default) when absent - so the
    container's host port mapping matches whatever vLLM will actually
    bind to, whether or not the recipe's command spells `--port` out.
    """
    for i, token in enumerate(serve_args):
        if token == "--port" and i + 1 < len(serve_args):
            return int(serve_args[i + 1])
        if token.startswith("--port="):
            return int(token.split("=", 1)[1])
    return default
