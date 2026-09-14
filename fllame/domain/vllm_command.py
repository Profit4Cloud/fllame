"""Parsing/validation for a `vllm serve <repo_id> <args...>` line,
shared between `Recipe.command` and the pasted-recipe parser.
"""

from __future__ import annotations

import re
import shlex

_FLAG_PATTERN = re.compile(r"^--[A-Za-z][\w-]*(=.*)?$")
_DANGEROUS_CHARS = re.compile(r"[;&|`\r]|\$\(")
_LINE_CONTINUATION = re.compile(r"[ \t]*\\[ \t]*\n[ \t]*")


class VllmCommandError(ValueError):
    pass


def join_line_continuations(text: str) -> str:
    """Collapses a `\\`-newline continuation to one space, the way a
    real shell would."""
    return _LINE_CONTINUATION.sub(" ", text)


def join_command_lines(lines: list[str]) -> str:
    r"""Joins lines into one logical `vllm serve` command. Lenient on
    purpose - unlike `join_line_continuations` - since these come from
    hand-edited recipe files: each line's leading/trailing whitespace is
    stripped, and a trailing `\` is optional (plenty of real examples
    show one flag per line with no continuation marker at all). Blank
    and `#`-comment lines are dropped.
    """
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            stripped = stripped[:-1].rstrip()
        cleaned.append(stripped)
    return " ".join(cleaned)


def split_shell_safe(where: str, text: str) -> list[str]:
    """Tokenizes with `shlex`, rejecting shell metacharacters first."""
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


def render_multiline_command(repo_id: str, serve_args: list[str]) -> str:
    """One flag per line, each ending in `\\` except the last - the
    shape recipes.vllm.ai and model cards typically show a command in.
    """
    lines = [f"vllm serve {repo_id}"]
    for token in serve_args:
        if token.startswith("--"):
            lines.append(token)
        else:
            lines[-1] += f" {shlex.quote(token)}"
    return " \\\n".join(lines)


def extract_port(serve_args: list[str], *, default: int = 8000) -> int:
    for i, token in enumerate(serve_args):
        if token == "--port" and i + 1 < len(serve_args):
            return int(serve_args[i + 1])
        if token.startswith("--port="):
            return int(token.split("=", 1)[1])
    return default
