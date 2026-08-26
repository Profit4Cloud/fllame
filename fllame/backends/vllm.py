"""Translates a Recipe into a docker-compose service definition that runs
`vllm serve` in a container. Assumes the model is already fully present
in the mounted HF cache (fllame's CLI guarantees this before ever
generating a service - see `fllame/models/puller.py`) rather than letting
vLLM's own auto-download run inside the container.
"""

from __future__ import annotations

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
            # Set explicitly rather than relying on the image's own
            # ENTRYPOINT/CMD, whatever a given vllm/vllm-openai tag
            # happens to bake in - this way behavior doesn't depend on
            # image internals we don't control.
            "entrypoint": ["vllm", "serve"],
            "command": [recipe.repo_id, "--port", str(recipe.port), *recipe.serve_args],
            "ports": [f"{recipe.port}:{recipe.port}"],
            "environment": {"HF_HOME": _CONTAINER_HF_HOME, **recipe.env},
            "volumes": [f"{hf_cache_dir}:{_CONTAINER_HF_HOME}"],
        }
        if recipe.gpus == "all":
            service["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
                    }
                }
            }
        return service
