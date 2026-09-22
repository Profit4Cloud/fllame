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
- fllame config show # Print fllame's currently configured settings.
- fllame config set-default-image IMAGE # Set the default image recipes fall back to; offers to update existing compose.yaml/Dockerfile files still using the old default.
- fllame config set-default-gpu-memory-utilization VALUE # Set the --gpu-memory-utilization value recipe build injects when a recipe doesn't set its own (default: 0.92); offers to update existing compose.yaml files still using the old default.
- fllame serve HANDLE [--yes] # Launch HANDLE's recipe via docker compose up -d (always detached) and print a docker logs command to follow it; never touches the network.
- fllame status # Show every recipe's container state via docker compose ps.
- fllame stop HANDLE # Stop HANDLE's container via docker compose stop.

Every command also takes `-h`/`--help`.

## Storage location

Recipes live in `~/.config/fllame/recipes/<handle>/recipe.yaml` (override with `FLLAME_RECIPES_DIR`). That same folder also stores `compose.yaml` and `Dockerfile` if generated during build. Both are safe to hand-edit - fllame never overwrites them on its own.

## Recipe format

| Field         | Required | Meaning |
|---------------|----------|---------|
| `command`     | yes      | the whole `vllm serve <repo_id> <args...>` invocation, not split into separate keys - `repo_id` and the host port mapping are derived from it |
| `image`       | no       | Docker image to run, e.g. `vllm/vllm-openai:v0.27.1` - omit to use `fllame config`'s default, or `vllm/vllm-openai:latest` if no default is set |
| `env`         | no       | environment variables; must not set `HF_HOME`, `HF_HUB_CACHE`, or `HF_HUB_OFFLINE`, which fllame manages itself |
| `preinstall`  | no       | shell commands run, in order, as `Dockerfile` `RUN` lines when `recipe build` builds this recipe's local image - not re-run at `serve` time |

## Advanced

Every generated `compose.yaml` gets `HF_HUB_OFFLINE=1`, `gpus: all`, and `ipc: host` unconditionally - hard defaults, not recipe fields. To override one (network access for a linked repo, pinning specific GPU device IDs, an explicit `shm_size:`), edit the generated `compose.yaml` directly. Fllame never respects your edits, but doesn't verify them, so you must know what you are doing. The same
goes for a `Dockerfile`, which is generated for recipes with `preinstall`.

Every generated `compose.yaml` always has `--gpu-memory-utilization` set to aconfigurable default. It is `0.92` by default, but this can be changed using `fllame config set-default-gpu-memory-utilization <value>`. This default is only used when there is no explicit value in
the recipe.

## Development

```
poetry install
poetry run pytest
poetry run ruff check .
```

See `CLAUDE.md` for the architecture this sits on and what's
deliberately not built yet.
