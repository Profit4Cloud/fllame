# fllame

A headless CLI for vLLM serving. A recipe is a plain YAML file you write
and commit to your own repo (like a Helm `values.yaml` or an Ollama
Modelfile) describing which model, which Docker image, and which `vllm
serve` flags to use. `fllame serve HANDLE` resolves one into a
`docker compose` service and runs it - no web UI, nothing installed on
the host beyond fllame and Docker.

## Install

Requires Python 3.12+ and Docker (with the `compose` plugin) on `PATH`.
fllame runs the official `vllm/vllm-openai` image; it doesn't vendor
vLLM itself.

```
pipx install .
```

Developing fllame: `poetry install`, then `eval $(poetry env activate)`.
Then you can run the fllame commands below within the activated environment.

## Commands

- fllame recipe list # List every recipe handle.
- fllame recipe show HANDLE # Print HANDLE's resolved recipe as YAML.
- fllame recipe add [VLLM_SERVE_LINE...] [--image IMAGE] [--pull] [--build] # Create a recipe from a pasted vllm serve line, or a guided dialogue if none is given.
- fllame recipe build HANDLE [--yes] # Write (or overwrite) HANDLE's compose.yaml (and Dockerfile, if it has preinstall) - the only command that does - then validate it with a real docker build/pull.
- fllame recipe edit HANDLE # Open HANDLE's recipe.yaml in $EDITOR and re-validate on save.
- fllame recipe remove HANDLE [--yes] # Delete HANDLE's whole recipe folder.
- fllame hardware scan # Detect this machine's NVIDIA GPU(s)/RAM and supported quantizations.
- fllame model pull REPO_ID # Download REPO_ID into the Hugging Face cache.
- fllame model list # List models currently present in the local cache (no network).
- fllame model scan [--query QUERY] [--quant QUANT] [--max-size SIZE] [--min-params N] [--max-params N] [--limit N] # Search the Hub for candidate models, ranked by hardware fit/downloads/recency.
- fllame model update [REPO_ID] [--apply] # Check cached model(s) against the Hub for a newer revision; --apply re-pulls anything stale.
- fllame config show # Print the currently configured default Docker image.
- fllame config set-default-image IMAGE # Set the default image recipes fall back to; offers to update existing compose.yaml files still using the old default.
- fllame serve HANDLE [--yes] # Launch HANDLE's recipe via docker compose up -d (always detached) and print a docker logs command to follow it; never touches the network.
- fllame status # Show every recipe's container state via docker compose ps.
- fllame stop HANDLE # Stop HANDLE's container via docker compose stop.

Every command also takes `-h`/`--help`.

## Storage location

Recipes live in `~/.config/fllame/recipes/<handle>/recipe.yaml` (override
with `FLLAME_RECIPES_DIR`). That same folder gets that handle's
`compose.yaml` too (and a `Dockerfile`, for a recipe with `preinstall`),
written only by `recipe build`/`recipe add --build`. `serve`/`status`/
`stop` never touch either, so a hand edit to one is safe to keep
indefinitely - `serve` will warn if it no longer matches what was last
built, rather than silently ignoring the edit or overwriting it.

`recipe build` also drops a small `.fllame-build.yaml` in the same
folder, recording what it last built - purely local-machine bookkeeping
for that warning, not meant to be portable or backed up.

fllame's scope ends once Docker and vLLM are running correctly on this
one machine: it's single-machine, single-user by design. Backing up or
versioning the recipes directory - `.fllame-build.yaml` included - is
entirely your own responsibility; fllame doesn't manage or assume any
of that itself.

## Recipe format

| Field         | Required | Meaning |
|---------------|----------|---------|
| `command`     | yes      | the whole `vllm serve <repo_id> <args...>` invocation, not split into separate keys - `repo_id` and the host port mapping are derived from it |
| `image`       | no       | Docker image to run, e.g. `vllm/vllm-openai:v0.27.1` - omit to use `fllame config`'s default, or `vllm/vllm-openai:latest` if no default is set |
| `env`         | no       | environment variables; must not set `HF_HOME`, `HF_HUB_CACHE`, or `HF_HUB_OFFLINE`, which fllame manages itself |
| `preinstall`  | no       | shell commands run, in order, as `Dockerfile` `RUN` lines when `recipe build` builds this recipe's local image - not re-run at `serve` time |

## Advanced

Every generated `compose.yaml` gets `HF_HUB_OFFLINE=1`, `gpus: all`, and
`ipc: host` unconditionally - hard defaults, not recipe fields. To
override one (network access for a linked repo, pinning specific GPU
device IDs, an explicit `shm_size:`), edit the generated `compose.yaml`
directly - fllame never verifies that edit, so keeping it correct is on
you (`serve` will flag that it no longer matches the last build, but
that's a heads-up, not a check that the edit itself is sound). The same
goes for a `Dockerfile`, for a recipe with `preinstall` - hand-edit it
and re-run `recipe build` to pick the change up.

`recipe build` also adds `--gpu-memory-utilization` to the `vllm serve`
command whenever the recipe's own `command` doesn't already set one -
left unset, vLLM happily reserves the whole GPU for itself, which on a
unified-memory machine means starving the OS, not just other processes
on the GPU. The value is `0.92` (matching vLLM's own out-of-the-box
default) on a discrete GPU, or on unified memory, whichever is lower of
`0.92` and the fraction of total memory left after reserving 5 GB for
the OS/everything else on the box. Pin your own value in the recipe's
`command` (`--gpu-memory-utilization 0.8`) to override it.

`recipe build` similarly computes `--max-model-len` by default, sized
against the model's cached weight size, its `config.json` architecture,
and the same effective `--gpu-memory-utilization`, on the assumption of
8 concurrent full-length requests - left unset, vLLM can default to the
model's full trained context length, which can need far more KV-cache
memory than is actually available. The computed value never exceeds
the model's own architectural context ceiling. If the model's
architecture isn't recognized, or the fitted context length would fall
below fllame's usable minimum, `--max-model-len` is left to be set by
hand in the recipe's `command` - the latter case makes `recipe build`
fail with the numbers involved, rather than serving an unusably short
context silently. `recipe build` also warns (without blocking) when a
recipe's `--tensor-parallel-size` doesn't match the number of GPUs
actually detected, since `gpus: all` and any tensor-parallel-size stay
exactly as configured either way.

## Development

```
poetry install
poetry run pytest
poetry run ruff check .
```

See `CLAUDE.md` for the architecture this sits on and what's
deliberately not built yet.
