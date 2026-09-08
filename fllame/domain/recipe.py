"""A recipe is the curated, hand-editable answer to "how do I serve this
model" for one model handle - the thing an operator commits to git and
reviews in a PR, not a runtime artifact.

Its `command` field holds the whole `vllm serve <repo_id> <args...>`
invocation - not split into separate `repo_id`/`args` YAML keys - so it
can be copied straight out of the recipe file and run by hand (`vllm
serve ...` on a box with vLLM installed) with no reassembly. Written
out (`to_dict`/`to_yaml`, used by both `RecipeStore.save` and `recipe
show`) as a canonical one-flag-per-line block regardless of how it was
originally authored. `repo_id`/`serve_args`/`port` are derived
properties for the rest of fllame (`model pull`, the compose backend,
...), parsed from `command` on access rather than stored a second time.
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

_VALID_GPUS = ("all", "none")


class RecipeError(ValueError):
    """A recipe file is missing a required field or otherwise malformed."""


@dataclass(frozen=True)
class Recipe:
    handle: str
    # The whole `vllm serve <repo_id> <args...>` line, exactly as
    # authored - see the module docstring for why this isn't split into
    # separate fields.
    command: str
    # `None` means "use fllame's configured default image" (`fllame
    # config`), resolved at the point a Recipe becomes a compose service -
    # not persisted into the recipe file, so a later `fllame config
    # set-default-image` change applies to every recipe that didn't pin
    # its own.
    image: str | None = None
    backend: str = "vllm"
    description: str | None = None
    # Docker GPU reservation: "all" (every GPU on the host) or "none"
    # (CPU-only). Anything more granular - specific device IDs, a count -
    # isn't supported yet; see CLAUDE.md, "Explicitly deferred".
    gpus: str = "all"
    env: dict[str, str] = field(default_factory=dict)
    # Shell commands run, in order, before `vllm serve` - e.g. the "extra
    # install" step some recipes need on top of the base image (a newer
    # `transformers`, a plugin package). Each entry is a whole command
    # line, not a token list like `command`'s args: unlike a `vllm serve`
    # flag value, a preinstall command is genuinely meant to be shell
    # text (it may legitimately contain its own quoting, `&&`, etc.), so
    # it's stored and later run verbatim rather than tokenized.
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

        gpus = data.get("gpus", "all")
        if gpus not in _VALID_GPUS:
            raise RecipeError(
                f"recipe '{handle}': 'gpus' must be one of {_VALID_GPUS}, got '{gpus}'"
            )

        env = dict(data.get("env") or {})
        if "HF_HOME" in env:
            raise RecipeError(
                f"recipe '{handle}': 'env' must not set HF_HOME - fllame manages the HF "
                "cache mount and its in-container path itself"
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
            gpus=gpus,
            env=env,
            preinstall=preinstall,
        )

    def to_dict(self) -> dict:
        """The inverse of `from_dict` - also what `recipe show` prints
        (with `image` already resolved to fllame's configured default,
        see `cli._resolve_image`), so the file on disk and the resolved
        view share one shape: `command` last and on its own, since it's
        the part meant to be copied out and run by hand. Rendered as a
        canonical multi-line block (see `render_multiline_command`) -
        one flag per line - regardless of how `command` happened to be
        authored (a single line, different spacing, ...), so the same
        recipe always looks the same on disk/in `recipe show`.
        """
        data: dict = {}
        if self.description:
            data["description"] = self.description
        if self.image:
            data["image"] = self.image
        if self.gpus != "all":
            data["gpus"] = self.gpus
        if self.env:
            data["env"] = self.env
        if self.preinstall:
            data["preinstall"] = self.preinstall
        data["command"] = render_multiline_command(self.repo_id, self.serve_args)
        return data

    def to_yaml(self) -> str:
        """`to_dict()` rendered with no line-wrap width limit (the
        default YAML dumper would otherwise fold a long line mid-flag),
        and `command` forced to YAML's literal block style (`|`) when
        it's actually multi-line, so each `--flag` lands on its own
        physical line rather than PyYAML's default single-quoted
        folding (which would visually blank-line-separate them instead).
        """
        data = self.to_dict()
        if "\n" in data["command"]:
            data["command"] = _LiteralStr(data["command"])
        return yaml.safe_dump(data, sort_keys=False, width=float("inf"))


class _LiteralStr(str):
    """A marker type telling `_literal_str_representer` to dump this
    particular string in YAML's literal block style (`|`) - forcing it
    only for this one value, not every string in the document.
    """


def _literal_str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(_LiteralStr, _literal_str_representer, Dumper=yaml.SafeDumper)
