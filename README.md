# fllame

A headless CLI for vLLM serving. `fllame serve <handle>` resolves a
hand-edited recipe (which HF repo, which Docker image, which `vllm serve`
flags) and runs it as a Docker container via `docker compose` - no web
UI, no dashboard, nothing to click through, and nothing installed on the
host beyond fllame itself.

Recipes are plain YAML files you write and commit to your own repo, the
same way you'd track a Helm `values.yaml` or an Ollama Modelfile. fllame
resolves them into a docker-compose service and runs it.

## Install

Requires Python 3.12+ and Docker (with the `compose` plugin) on `PATH`.
fllame does not install or vendor vLLM itself - it runs the official
`vllm/vllm-openai` image.

To get a `fllame` command available everywhere, install it with
[pipx](https://pipx.pypa.io/) (installs into its own isolated venv and
puts a shim on your `PATH` - no activating anything needed afterward,
and it doesn't matter which directory you're in):

```
pipx install .
```

`poetry install` (below) only creates a project-local virtualenv - it's
what you want if you're developing fllame itself, but the resulting
`fllame` command only exists inside that venv, reachable via `poetry run
fllame ...` or after activating it with `eval $(poetry env activate)`
(Poetry 2.0+ - the old `poetry shell` needs a separate plugin now). If
you just ran `poetry install` and got `fllame: command not found`,
that's why - use `pipx install .` instead for everyday use.

Developing fllame and using it as a normal global command at the same
time: `pipx install --editable .` (or `pipx install -e .`). It's the
same global `fllame` command as above, but the venv points straight at
this checkout instead of a frozen copy - edit the source and the very
next `fllame` invocation picks it up, no reinstall. `pipx install .`
(no `-e`) copies the code at install time, so it won't see later edits.

## Quick start

Recipes live in `~/.config/fllame/recipes/<handle>.yaml` by default
(override with `FLLAME_RECIPES_DIR`). One file per model handle - see
`examples/recipes/llama-3-8b-instruct.yaml` for a full example:

```yaml
# ~/.config/fllame/recipes/llama-3-8b-instruct.yaml
repo_id: meta-llama/Meta-Llama-3-8B-Instruct
image: vllm/vllm-openai:v0.27.1
description: Llama 3 8B Instruct, single-GPU default profile
port: 8000
gpus: all
serve_args:
  - --gpu-memory-utilization=0.9
  - --max-model-len=8192
```

Then:

```
fllame hardware scan                           # what this box can run
fllame recipe list
fllame recipe show llama-3-8b-instruct         # resolved compose service, as YAML
fllame model pull llama-3-8b-instruct          # download into the HF cache, standalone
fllame model list                              # what's actually cached locally
fllame serve llama-3-8b-instruct               # pulls if needed, then runs in the foreground
fllame serve llama-3-8b-instruct --detach      # same, but backgrounded
fllame serve llama-3-8b-instruct --offline     # never touch the network - fail if not cached
fllame status                                  # docker compose ps
fllame stop llama-3-8b-instruct                # docker compose stop
```

`fllame hardware scan` detects NVIDIA GPU(s) via `nvidia-smi` (name, count,
VRAM per GPU) and RAM via `/proc/meminfo`, and reports which vLLM
quantizations that hardware supports. It's a live scan, not a persisted
value - nothing to keep in sync. It does not check a recipe against the
detected hardware before `fllame serve` runs it; see CLAUDE.md for why.

`fllame serve` always downloads the model into Hugging Face's own cache
first (via `huggingface_hub.snapshot_download`, a no-op if it's already
there) before starting the container - vLLM's own auto-download inside
the container is never relied on. That same cache directory (wherever
`HF_HOME`/`HF_HUB_CACHE` resolves to) is bind-mounted into the container,
so the download only ever happens once, on the host.

For a fully offline demo: run `fllame model pull <handle>` while online,
then `fllame serve <handle> --offline` later with no network at all.
`--offline` forces `local_files_only` on the download check (fails fast
with a clear error if the model isn't fully cached, rather than hoping a
plain download call happens to fall back to cache quickly on a
genuinely offline machine) and sets `HF_HUB_OFFLINE=1` on the container
itself, so vLLM doesn't attempt any network call either.

`fllame serve`/`status`/`stop` all regenerate
`$FLLAME_STATE_DIR/docker-compose.yml` (default
`~/.local/state/fllame/docker-compose.yml`) from every recipe on file
before running a `docker compose` command against it. That file is a
generated artifact fllame fully owns - don't hand-edit it, edit the
recipe instead.

## Recipe format

| Field         | Required | Meaning |
|---------------|----------|---------|
| `repo_id`     | yes      | HF repo id (or local path) - what's passed to `vllm serve` and downloaded via `fllame model pull`/`serve` |
| `image`       | yes      | the Docker image to run, e.g. `vllm/vllm-openai:v0.27.1` |
| `backend`     | no       | must be `vllm` if set - the only backend fllame ships today |
| `description` | no       | free text, shown by `recipe show` |
| `port`        | no       | default `8000`; used for both the container's `--port` and the host port mapping |
| `gpus`        | no       | `all` (default) or `none` - whether the container gets a GPU reservation |
| `env`         | no       | environment variables set on the container; must not set `HF_HOME`, which fllame manages itself |
| `serve_args`  | no       | extra flags appended to `vllm serve <repo_id>` verbatim - don't include `--port` here, use the `port` field |

## Development

```
poetry install
poetry run pytest
poetry run ruff check .
```

See `CLAUDE.md` for the architecture this sits on and what's deliberately
not built yet.
