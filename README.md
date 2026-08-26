# fllame

A headless CLI for vLLM serving. `fllame serve <handle>` resolves a
hand-edited recipe (which HF repo, which `vllm serve` flags) and launches
vLLM - no web UI, no dashboard, nothing to click through.

Recipes are plain YAML files you write and commit to your own repo, the
same way you'd track a Helm `values.yaml` or an Ollama Modelfile. fllame
just resolves and runs them.

## Install

Requires Python 3.11+ and a working `vllm` installation on `PATH` (fllame
does not install or vendor vLLM itself).

```
poetry install
```

## Quick start

Recipes live in `~/.config/fllame/recipes/<handle>.yaml` by default
(override with `FLLAME_RECIPES_DIR`). One file per model handle - see
`examples/recipes/llama-3-8b-instruct.yaml` for a full example:

```yaml
# ~/.config/fllame/recipes/llama-3-8b-instruct.yaml
repo_id: meta-llama/Meta-Llama-3-8B-Instruct
description: Llama 3 8B Instruct, single-GPU default profile
port: 8000
env:
  HF_HOME: /data/hf-cache
serve_args:
  - --gpu-memory-utilization=0.9
  - --max-model-len=8192
```

Then:

```
fllame recipe list
fllame recipe show llama-3-8b-instruct
fllame serve llama-3-8b-instruct              # foreground, execs vllm
fllame serve llama-3-8b-instruct --detach      # background, tracked
fllame status
fllame stop llama-3-8b-instruct
```

## Recipe format

| Field         | Required | Meaning |
|---------------|----------|---------|
| `repo_id`     | yes      | HF repo id (or local path) passed to `vllm serve` |
| `backend`     | no       | must be `vllm` if set - the only backend fllame ships today |
| `description` | no       | free text, shown by `recipe show` |
| `port`        | no       | default `8000`; override per-invocation with `fllame serve --port` |
| `env`         | no       | environment variables set for the `vllm serve` subprocess |
| `serve_args`  | no       | extra flags appended to `vllm serve <repo_id>` verbatim - don't include `--port` here, use the `port` field |
| `image`       | no       | reserved for a future container/Helm export target; unused by `fllame serve` today |

## Development

```
poetry install
poetry run pytest
poetry run ruff check .
```

See `CLAUDE.md` for the architecture this sits on and what's deliberately
not built yet.
