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

Two more hard defaults, unconditional for the same reason (a sane
baseline for a vLLM container specifically, not something a recipe
should have to opt into): `ipc: host` gives vLLM's own multiprocessing
workers (tensor-parallel workers, NCCL) access to the host's shared
memory rather than Docker's own default `/dev/shm` (usually 64MB),
which is a routine cause of a worker crashing outright once a recipe
goes beyond a single-GPU, single-process setup; and `gpus: "all"`
reserves every GPU on the host, unconditionally, the Compose
Specification's own shorthand for what `docker run --gpus all` does -
not a recipe-level knob at all. Overriding either - pinning specific
device IDs, or swapping `ipc: host` for an explicit `shm_size:` - is a
hand-edit-the-generated-compose-file situation, entirely the
operator's own responsibility and never verified by fllame; see the
README's "Advanced" section.

The HF cache bind mount's host side is written as `${HOME}/...` rather
than a literal absolute path whenever `hf_cache_dir` sits under the
current user's home directory (the default, out-of-the-box location) -
see `_host_volume_source`. Docker Compose interpolates `${HOME}` itself,
from whatever shell environment `docker compose` runs in, so the
generated `compose.yaml` stays correct after being copied to a
different machine or run under a different account, rather than baking
in the one user's home directory it happened to be generated under.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from fllame.domain.recipe import Recipe

# Where the OpenAI-compatible vLLM image expects its HF cache; the host
# cache directory is bind-mounted here.
_CONTAINER_HF_HOME = "/root/.cache/huggingface"


def _host_volume_source(hf_cache_dir: Path) -> str:
    """The host side of the HF cache bind mount. Written as `${HOME}/...`
    when `hf_cache_dir` sits under the current user's home directory -
    the default, out-of-the-box location, since HF_HOME/HF_HUB_CACHE
    fall back to it - rather than as a literal absolute path, so the
    same generated compose.yaml stays correct after being copied
    elsewhere or run by a different account (see the module docstring).
    Falls back to the literal path when the cache lives somewhere
    `${HOME}` can't express - a custom HF_HOME/HF_HUB_CACHE pointed
    outside the home directory entirely.
    """
    try:
        relative = hf_cache_dir.relative_to(Path.home())
    except ValueError:
        return str(hf_cache_dir)
    return "${HOME}" if str(relative) == "." else f"${{HOME}}/{relative.as_posix()}"


def cache_volume_host_path(service: dict) -> str | None:
    """The host side of `service`'s HF cache bind mount (see
    `_host_volume_source`/`build_service`), or `None` if `service` has
    no volumes at all or none of them mount `_CONTAINER_HF_HOME` - lets
    a caller compare an already-generated `compose.yaml`'s cache mount
    against what would be generated now, without needing to know the
    volume list's internal shape itself.
    """
    for volume in service.get("volumes", []):
        host, _, container = volume.partition(":")
        if container == _CONTAINER_HF_HOME:
            return host
    return None


class VllmServingBackend:
    name = "vllm"

    def build_service(self, recipe: Recipe, *, hf_cache_dir: Path) -> dict:
        # `hf_cache_dir` (the host side of the volume below) is always
        # `HF_HUB_CACHE`, not `HF_HOME` - by default one directory
        # *under* it (`HF_HOME/hub`). Setting only `HF_HOME` here would
        # leave the container's own huggingface_hub computing its
        # `HF_HUB_CACHE` as `HF_HOME/hub`, one level below where the
        # mount actually lands, so it would never find anything cached
        # regardless of what's really on disk. Setting `HF_HUB_CACHE`
        # explicitly to the same path the volume is mounted at removes
        # that implicit derivation entirely.
        env = {
            "HF_HOME": _CONTAINER_HF_HOME,
            "HF_HUB_CACHE": _CONTAINER_HF_HOME,
            "HF_HUB_OFFLINE": "1",
            **recipe.env,
        }
        service: dict = {
            "image": recipe.image,
            "ports": [f"{recipe.port}:{recipe.port}"],
            # A list of `KEY=VALUE` strings rather than a `KEY: VALUE`
            # mapping - both are valid Compose syntax for the same
            # thing, this is just the more familiar shell-like shape.
            "environment": [f"{key}={value}" for key, value in env.items()],
            "volumes": [f"{_host_volume_source(hf_cache_dir)}:{_CONTAINER_HF_HOME}"],
            # vLLM's own multiprocessing workers (tensor-parallel, NCCL)
            # need more shared memory than Docker's tiny default
            # `/dev/shm` - see the module docstring.
            "ipc": "host",
            # Unconditional, the Compose Specification's own shorthand
            # for `docker run --gpus all` - not a recipe-level knob (see
            # the module docstring). Pin specific device IDs by
            # hand-editing the generated compose.yaml directly.
            "gpus": "all",
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
            # compose.yaml per recipe, nothing else to build or
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

        return service
