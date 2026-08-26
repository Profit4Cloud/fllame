"""A recipe is the curated, hand-editable answer to "how do I serve this
model" for one model handle - the thing an operator commits to git and
reviews in a PR, not a runtime artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_VALID_GPUS = ("all", "none")


class RecipeError(ValueError):
    """A recipe file is missing a required field or otherwise malformed."""


@dataclass(frozen=True)
class Recipe:
    handle: str
    repo_id: str
    image: str
    backend: str = "vllm"
    description: str | None = None
    port: int = 8000
    # Docker GPU reservation: "all" (every GPU on the host) or "none"
    # (CPU-only). Anything more granular - specific device IDs, a count -
    # isn't supported yet; see CLAUDE.md, "Explicitly deferred".
    gpus: str = "all"
    env: dict[str, str] = field(default_factory=dict)
    serve_args: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(handle: str, data: dict) -> Recipe:
        if "repo_id" not in data:
            raise RecipeError(f"recipe '{handle}': missing required field 'repo_id'")
        if "image" not in data:
            raise RecipeError(
                f"recipe '{handle}': missing required field 'image' "
                "(the Docker image to run, e.g. 'vllm/vllm-openai:v0.27.1')"
            )

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

        return Recipe(
            handle=handle,
            repo_id=data["repo_id"],
            image=data["image"],
            backend=backend,
            description=data.get("description"),
            port=data.get("port", 8000),
            gpus=gpus,
            env=env,
            serve_args=list(data.get("serve_args") or []),
        )
