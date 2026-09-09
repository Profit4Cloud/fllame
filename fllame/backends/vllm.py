"""Translates a Recipe into a docker-compose service definition that runs
`vllm serve` in a container. Assumes the model is already fully present
in the mounted HF cache (fllame's CLI guarantees this before ever
generating a service - see `fllame/models/puller.py`) rather than letting
vLLM's own auto-download run inside the container.

Every generated service also gets `HF_HUB_OFFLINE=1` unconditionally,
for the same reason: the model is always already fully downloaded by
the time this runs, so vLLM has no legitimate need to reach the Hub
itself, and letting it try anyway trades a guaranteed-fast local
resolution for a network call that can fail slowly or silently. Not a
recipe/`env` concern - `Recipe.from_dict` rejects a recipe that tries
to set `HF_HUB_OFFLINE` itself, the same way it already rejects
`HF_HOME` - if a specific model genuinely needs the network for
something beyond its own repo_id (e.g. a linked tokenizer/base-model
repo), that's a hand-edit-the-generated-compose-file situation.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from fllame.domain.recipe import Recipe

# Where the OpenAI-compatible vLLM image expects its HF cache; the host
# cache directory is bind-mounted here.
_CONTAINER_HF_HOME = "/root/.cache/huggingface"


class VllmServingBackend:
    name = "vllm"

    def build_service(self, recipe: Recipe, *, hf_cache_dir: Path) -> dict:
        service: dict = {
            "image": recipe.image,
            "ports": [f"{recipe.port}:{recipe.port}"],
            "environment": {
                "HF_HOME": _CONTAINER_HF_HOME,
                "HF_HUB_OFFLINE": "1",
                **recipe.env,
            },
            "volumes": [f"{hf_cache_dir}:{_CONTAINER_HF_HOME}"],
        }

        # No explicit `--port` inserted here: recipe.port is only for
        # the host mapping above, derived from whatever's already in
        # serve_args (or vLLM's own default) - not re-added, or a
        # recipe whose command already sets --port would end up with
        # it twice.
        vllm_command = shlex.join(["vllm", "serve", recipe.repo_id, *recipe.serve_args])

        if recipe.preinstall:
            # No custom image/Dockerfile: the preinstall step runs as
            # part of the container's own command, through a shell,
            # every time it starts (there's no build step to cache it
            # in) - the whole point of this recipe field being a
            # compose-only concept: one self-contained
            # docker-compose.yml per recipe, nothing else to build or
            # manage alongside it. `vllm_command` above is already
            # shell-escaped (`shlex.join`), so a repo_id/flag value
            # containing shell syntax stays a literal argument rather
            # than being interpreted as more shell; `recipe.preinstall`
            # entries are joined in verbatim since they're genuinely
            # meant to be shell text (see `Recipe.preinstall`'s
            # docstring). `exec` on the final command replaces the
            # shell process with vLLM's own, so it becomes PID 1 and
            # receives `docker stop`'s SIGTERM directly instead of it
            # being swallowed by an intermediate shell.
            segments = [*recipe.preinstall, f"exec {vllm_command}"]
            service["entrypoint"] = ["sh", "-c"]
            service["command"] = [" && ".join(segments)]
        else:
            # Set explicitly rather than relying on the image's own
            # ENTRYPOINT/CMD, whatever a given vllm/vllm-openai tag
            # happens to bake in - this way behavior doesn't depend on
            # image internals we don't control.
            service["entrypoint"] = ["vllm", "serve"]
            service["command"] = [recipe.repo_id, *recipe.serve_args]

        if recipe.gpus == "all":
            service["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
                    }
                }
            }
        return service
