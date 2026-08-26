"""A recipe is the curated, hand-editable answer to "how do I serve this
model" for one model handle - the thing an operator commits to git and
reviews in a PR, not a runtime artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class RecipeError(ValueError):
    """A recipe file is missing a required field or otherwise malformed."""


@dataclass(frozen=True)
class Recipe:
    handle: str
    repo_id: str
    backend: str = "vllm"
    description: str | None = None
    port: int = 8000
    env: dict[str, str] = field(default_factory=dict)
    serve_args: list[str] = field(default_factory=list)
    # Reserved for a future container/Helm export target; `fllame serve`
    # does not use this today - see CLAUDE.md, "Explicitly deferred".
    image: str | None = None

    @staticmethod
    def from_dict(handle: str, data: dict) -> Recipe:
        if "repo_id" not in data:
            raise RecipeError(f"recipe '{handle}': missing required field 'repo_id'")

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

        return Recipe(
            handle=handle,
            repo_id=data["repo_id"],
            backend=backend,
            description=data.get("description"),
            port=data.get("port", 8000),
            env=dict(data.get("env") or {}),
            serve_args=list(data.get("serve_args") or []),
            image=data.get("image"),
        )
