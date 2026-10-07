# fllame

A headless CLI for vLLM serving. A recipe is a plain YAML file you write
and commit to your own repo (like a Helm `values.yaml` or an Ollama
Modelfile) describing which model, which Docker image, and which `vllm
serve` flags to use. `fllame serve RECIPE_ID` resolves one into a
`docker compose` service and runs it - no web UI, nothing installed on
the host beyond fllame and Docker.

## Prerequisites

- Python 3.12+.
- Docker with the `compose` plugin on `PATH`.
- An NVIDIA GPU with the NVIDIA Container Toolkit.

## Install

Install [pipx](https://pipx.pypa.io) first if you don't have it.

```
git clone https://github.com/Profit4Cloud/fllame.git
cd fllame
pipx install .
```

Developing fllame: `poetry install`, then `eval $(poetry env activate)`.
Then you can run the fllame commands below within the activated environment.

## Commands

- fllame recipe list # List every RECIPE_ID.
- fllame recipe show RECIPE_ID # Print RECIPE_ID's resolved recipe as YAML.
- fllame recipe add [VLLM_SERVE_LINE...] [--pull] [--build] # Create a recipe from a vllm serve command, or a guided dialogue if none is given.
- fllame recipe build RECIPE_ID [--yes] # Write (or overwrite) RECIPE_ID's compose.yaml (and Dockerfile, if it has preinstall) - the only command that does - then validate it with a real docker build/pull. Skips the pull if the image is already downloaded.
- fllame recipe vram RECIPE_ID [--max-model-len N] [--max-num-seqs N] [--details] # Estimate VRAM for RECIPE_ID's recipe.
- fllame recipe edit RECIPE_ID # Open RECIPE_ID's recipe.yaml in $EDITOR and re-validate on save.
- fllame recipe remove RECIPE_ID [--yes] # Delete RECIPE_ID's whole recipe folder.
- fllame hardware scan # Detect this machine's NVIDIA GPU(s)/RAM and supported quantizations.
- fllame model pull REPO_ID # Download REPO_ID into the Hugging Face cache.
- fllame model list # List models currently present in the local cache (no network).
- fllame model scan [--query QUERY] [--quant QUANT] [--max-size SIZE] [--min-params N] [--max-params N] [--limit N] # Search the Hub, ranked by downloads and size.
- fllame model update [REPO_ID] [--apply] # Check cached model(s) against the Hub for a newer revision; --apply re-pulls anything stale.
- fllame config show # Print fllame's currently configured settings.
- fllame config set-default-image IMAGE # Set the default image recipes fall back to (default: vllm/vllm-openai:latest); offers to update existing compose.yaml/Dockerfile files still using the old default.
- fllame config set-default-gpu-memory-utilization VALUE # Set the --gpu-memory-utilization value recipe build injects when a recipe doesn't set its own (default: 0.92); offers to update existing compose.yaml files still using the old default.
- fllame serve RECIPE_ID [--yes] # Launch RECIPE_ID's recipe via docker compose up -d (always detached) and print the commands to check readiness and follow the logs; never touches the network.
- fllame bench RECIPE_ID [--concurrency 1,4,8,16,32] [--num-prompts N,...] [--input-len N] [--output-len N] # Run a vllm bench serve concurrency sweep inside RECIPE_ID's running container, print a results table, and save a reproducible run to RECIPE_ID's bench/<timestamp>/ folder.
- fllame status [RECIPE_ID] [--watch] # Show each recipe's container state via docker compose ps, plus whether a running server can generate. --watch re-checks every 5s until every running server is ready or in error.
- fllame stop [RECIPE_ID] # Stop RECIPE_ID's container via docker compose stop. Without RECIPE_ID, stop every running one.

Every command also takes `-h`/`--help`.

## Storage location

Recipes live in `~/.config/fllame/recipes/<RECIPE_ID>/recipe.yaml` (override with `FLLAME_RECIPES_DIR`). That same folder also stores `compose.yaml` and `Dockerfile` if generated during build. Both are safe to hand-edit - fllame never overwrites them on its own.

## Adding a recipe

Pass a `vllm serve` command to `recipe add`. Quotes are optional, and the command may span several lines:

```bash
fllame recipe add vllm serve org/repo --max-model-len 8192

fllame recipe add "vllm serve org/repo
  --max-model-len 8192
  --tensor-parallel-size 1"
```

Without quotes, end each line but the last with `\`. The recipe uses the default image. For a custom image, env vars or preinstall commands, run `fllame recipe add` without arguments to start a dialogue.

## Docker image versions

The default image is `vllm/vllm-openai:latest`. `recipe build` resolves `latest` to the release it points at, e.g. `vllm/vllm-openai:v0.31.0`, and writes that into `compose.yaml` or the `Dockerfile`. A built recipe therefore never drifts to another vLLM version. Rebuilding picks up the newest release. If no release tag matches, the build stops with an error. Then set an image with a version tag.

Without network access, `recipe build` uses the newest `vX.Y.Z` version already downloaded, and says so. If none is downloaded, it stops with an error.

## Recipe format

| Field         | Required | Meaning |
|---------------|----------|---------|
| `command`     | yes      | the whole `vllm serve <repo_id> <args...>` invocation, not split into separate keys - `repo_id` and the host port mapping are derived from it |
| `image`       | no       | Docker image to run, e.g. `vllm/vllm-openai:v0.27.1` - omit to use `fllame config`'s default |
| `env`         | no       | environment variables; must not set `HF_HOME`, `HF_HUB_CACHE`, or `HF_HUB_OFFLINE`, which fllame manages itself |
| `preinstall`  | no       | shell commands run, in order, as `Dockerfile` `RUN` lines when `recipe build` builds this recipe's local image - not re-run at `serve` time |

## Advanced

Every generated `compose.yaml` gets `gpus: all`, `ipc: host`, and the env vars listed under [Environment Variables](#environment-variables). To override one (network access for a linked repo, pinning specific GPU device IDs, an explicit `shm_size:`), edit the generated `compose.yaml` directly. Fllame always respects your edits, but doesn't verify them, so you must know what you are doing. The same goes for a `Dockerfile`, which is generated for recipes with `preinstall`.

Every generated `compose.yaml` always has `--gpu-memory-utilization` set to a configurable default. It is `0.92` by default, but this can be changed using `fllame config set-default-gpu-memory-utilization <value>`. This default is only used when there is no explicit value in
the recipe.

## Environment Variables

Every generated `compose.yaml` sets these env vars:

| Variable                        | Why                                         |
|---------------------------------|---------------------------------------------|
| `HF_HUB_OFFLINE=1`              | No Hugging Face downloads or update checks  |
| `HF_HOME`, `HF_HUB_CACHE`       | Point to the mounted local cache.           |
| `VLLM_NO_USAGE_STATS=1`         | Disable vLLM usage stats.                   |
| `HF_HUB_DISABLE_TELEMETRY=1`    | Disable Hugging Face telemetry.             |
| `DO_NOT_TRACK=1`                | Opt out for other libraries.                |
| `RAY_USAGE_STATS_ENABLED=0`     | Disable Ray usage stats.                    |
| `PYTHONUNBUFFERED=1`            | Show logs immediately in `docker logs`.     |

A recipe's `env` can override the telemetry vars.

### Enterprise use

Check these before serving in a shared or production environment:

- **Media URLs.** Requests with `image_url` make vLLM fetch remote URLs. Restrict hosts with `--allowed-media-domains`. Block redirects with `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0`.
- **Network exposure.** The port binds to all interfaces, without authentication. Bind it to `127.0.0.1` in `compose.yaml`. Put a reverse proxy, like nginx, in front. Also require a token with `--api-key`.
- **Dev mode.** Never set `VLLM_SERVER_DEV_MODE=1`. It exposes unsafe debug endpoints.
- **`--trust-remote-code`.** Runs Python code from the model repo. Only use it for repos you trust.
- **HF cache.** fllame trusts everything in the HF cache folders. Control who can write there.
- **Offline check.** Add a network with `internal: true` to `compose.yaml`. Serve once to prove no network is needed. Ports are not published then, so check `docker logs`.
- **`ipc: host`.** Shares the host's IPC namespace, for PyTorch shared memory. For stronger isolation, replace it with `shm_size:`, e.g. `16g`.

## Development

```
poetry install
poetry run pytest
poetry run ruff check .
```

See `CLAUDE.md` for the architecture this sits on and what's
deliberately not built yet.
