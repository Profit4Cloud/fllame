"""A recipe is the curated, hand-editable answer to "how do I serve this
model" for one model handle - the thing an operator commits to git and
reviews in a PR, not a runtime artifact.

Its `command` field is stored (and shown by `recipe show`) as the whole
`vllm serve <repo_id> <args...>` line, verbatim - not split into
separate `repo_id`/`args` YAML keys - so it can be copied straight out
of the recipe file and run by hand (`vllm serve ...` on a box with vLLM
installed) with no reassembly. `repo_id`/`serve_args`/`port` are still
available as derived properties for the rest of fllame (`model pull`,
the compose backend, ...), parsed from `command` on access rather than
stored a second time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from fllame.domain.vllm_command import VllmCommandError, extract_port, parse_vllm_serve_command

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
        the part meant to be copied out and run by hand.
        """
        data: dict = {}
        if self.description:
            data["description"] = self.description
        if self.image:
            data["image"] = self.image
        data["gpus"] = self.gpus
        if self.env:
            data["env"] = self.env
        if self.preinstall:
            data["preinstall"] = self.preinstall
        data["command"] = self.command
        return data

    def to_yaml(self) -> str:
        """`to_dict()` rendered with no line-wrap width limit: the whole
        point of a copy-pasteable `command` is that it's one physical
        line in the file - the default YAML dumper would otherwise fold
        a long one mid-flag.
        """
        return yaml.safe_dump(self.to_dict(), sort_keys=False, width=float("inf"))
