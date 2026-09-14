"""`command` holds the whole `vllm serve <repo_id> <args...>` line rather
than separate `repo_id`/`args` fields, so it can be copied straight out
of the recipe file and run by hand. `repo_id`/`serve_args`/`port` are
derived from it on access, not stored a second time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from fllame.domain.vllm_command import (
    VllmCommandError,
    extract_port,
    parse_vllm_serve_command,
    render_multiline_command,
)


class RecipeError(ValueError):
    """A recipe file is missing a required field or otherwise malformed."""


@dataclass(frozen=True)
class Recipe:
    handle: str
    command: str
    # None means "use fllame's configured default image", resolved when
    # a Recipe becomes a compose service - not persisted, so a later
    # `fllame config set-default-image` applies to recipes that didn't
    # pin their own.
    image: str | None = None
    backend: str = "vllm"
    description: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    # Whole command lines, not tokenized like `command`'s args - shell
    # text run verbatim (may contain its own quoting, `&&`, etc.).
    preinstall: list[str] = field(default_factory=list)

    @property
    def repo_id(self) -> str:
        return parse_vllm_serve_command(self.command)[0]

    @property
    def serve_args(self) -> list[str]:
        return parse_vllm_serve_command(self.command)[1]

    @property
    def port(self) -> int:
        return extract_port(self.serve_args)

    @staticmethod
    def from_dict(handle: str, data: dict) -> Recipe:
        if "command" not in data:
            raise RecipeError(
                f"recipe '{handle}': missing required field 'command' "
                "(the `vllm serve <repo_id> <args...>` line)"
            )
        command = data["command"]
        try:
            parse_vllm_serve_command(command)
        except VllmCommandError as e:
            raise RecipeError(f"recipe '{handle}': 'command': {e}") from e

        declared_handle = data.get("handle")
        if declared_handle is not None and declared_handle != handle:
            raise RecipeError(
                f"recipe file '{handle}.yaml' declares handle '{declared_handle}', "
                "which must match the filename"
            )

        backend = data.get("backend", "vllm")
        if backend != "vllm":
            raise RecipeError(
                f"recipe '{handle}': backend '{backend}' is not supported - "
                "fllame only ships a vLLM backend today"
            )

        env = dict(data.get("env") or {})
        if "HF_HOME" in env:
            raise RecipeError(
                f"recipe '{handle}': 'env' must not set HF_HOME - fllame manages the HF "
                "cache mount and its in-container path itself"
            )
        if "HF_HUB_OFFLINE" in env:
            raise RecipeError(
                f"recipe '{handle}': 'env' must not set HF_HUB_OFFLINE - fllame always "
                "sets it itself; edit the generated compose file directly if a specific "
                "model genuinely needs network access"
            )
        if "HF_HUB_CACHE" in env:
            raise RecipeError(
                f"recipe '{handle}': 'env' must not set HF_HUB_CACHE - fllame points it "
                "at the same in-container path the cache mount and HF_HOME both use"
            )

        preinstall = list(data.get("preinstall") or [])
        for step in preinstall:
            if not isinstance(step, str) or not step.strip():
                raise RecipeError(
                    f"recipe '{handle}': 'preinstall' entries must be non-empty strings, "
                    f"got {step!r}"
                )

        return Recipe(
            handle=handle,
            command=command,
            image=data.get("image"),
            backend=backend,
            description=data.get("description"),
            env=env,
            preinstall=preinstall,
        )

    def to_dict(self) -> dict:
        """Inverse of `from_dict`. `command` is always rendered one flag
        per line via `render_multiline_command`, regardless of how it
        was originally authored."""
        data: dict = {}
        if self.description:
            data["description"] = self.description
        if self.image:
            data["image"] = self.image
        if self.env:
            data["env"] = self.env
        if self.preinstall:
            data["preinstall"] = self.preinstall
        data["command"] = render_multiline_command(self.repo_id, self.serve_args)
        return data

    def to_yaml(self) -> str:
        """No line-wrap width limit, or the default YAML dumper would
        fold a long line mid-flag; `command` forced to literal block
        style (`|`) when multi-line, or PyYAML's default folding would
        blank-line-separate each flag instead of one-per-line."""
        data = self.to_dict()
        if "\n" in data["command"]:
            data["command"] = _LiteralStr(data["command"])
        return yaml.safe_dump(data, sort_keys=False, width=float("inf"))


class _LiteralStr(str):
    """Marks one string to dump in YAML's literal block style (`|`),
    not every string in the document."""


def _literal_str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(_LiteralStr, _literal_str_representer, Dumper=yaml.SafeDumper)
