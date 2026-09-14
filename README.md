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
- fllame recipe build HANDLE [--yes] # Write (or overwrite) HANDLE's compose.yaml - the only command that does.
- fllame recipe edit HANDLE # Open HANDLE's recipe.yaml in $EDITOR and re-validate on save.
- fllame recipe remove HANDLE [--yes] # Delete HANDLE's whole recipe folder.
- fllame hardware scan # Detect this machine's NVIDIA GPU(s)/RAM and supported quantizations.
- fllame model pull REPO_ID # Download REPO_ID into the Hugging Face cache.
- fllame model list # List models currently present in the local cache (no network).
- fllame model scan [--query QUERY] [--quant QUANT] [--max-size SIZE] [--min-params N] [--max-params N] [--limit N] # Search the Hub for candidate models, ranked by hardware fit/downloads/recency.
- fllame model update [REPO_ID] [--apply] # Check cached model(s) against the Hub for a newer revision; --apply re-pulls anything stale.
- fllame config show # Print the currently configured default Docker image.
- fllame config set-default-image IMAGE # Set the default image recipes fall back to; offers to update existing compose.yaml files still using the old default.
- fllame serve HANDLE [--detach] [--yes] # Launch HANDLE's recipe via docker compose up; never touches the network.
- fllame status # Show every recipe's container state via docker compose ps.
- fllame stop HANDLE # Stop HANDLE's container via docker compose stop.

Every command also takes `-h`/`--help`.

## Storage location

Recipes live in `~/.config/fllame/recipes/<handle>/recipe.yaml` (override
with `FLLAME_RECIPES_DIR`). That same folder gets that handle's
`compose.yaml` too, written only by `recipe build`/`recipe add --build`.
`serve`/`status`/`stop` never touch it, so a hand-edited `compose.yaml` is
safe to keep indefinitely.

## Recipe format

| Field         | Required | Meaning |
|---------------|----------|---------|
| `command`     | yes      | the whole `vllm serve <repo_id> <args...>` invocation, not split into separate keys - `repo_id` and the host port mapping are derived from it |
| `image`       | no       | Docker image to run, e.g. `vllm/vllm-openai:v0.27.1` - omit to use `fllame config`'s default, or `vllm/vllm-openai:latest` if no default is set |
| `env`         | no       | environment variables; must not set `HF_HOME`, `HF_HUB_CACHE`, or `HF_HUB_OFFLINE`, which fllame manages itself |
| `preinstall`  | no       | shell commands run, in order, before `vllm serve` |

## Advanced

Every generated `compose.yaml` gets `HF_HUB_OFFLINE=1`, `gpus: all`, and
`ipc: host` unconditionally - hard defaults, not recipe fields. To
override one (network access for a linked repo, pinning specific GPU
device IDs, an explicit `shm_size:`), edit the generated `compose.yaml`
directly - fllame never verifies that edit, so keeping it correct is on
you.

## Development

```
poetry install
poetry run pytest
poetry run ruff check .
```

See `CLAUDE.md` for the architecture this sits on and what's
deliberately not built yet.
