# Repo orientation

fllame is a headless CLI for serving models with vLLM: a curated recipe
registry (model handle -> HF repo id + `vllm serve` flags), a thin
wrapper that resolves and launches vLLM from those recipes, and local
state tracking for anything started in the background. No web UI, no
admin server, no database server to run.

It is a fresh, narrow rewrite - not a fork of, and not code-shared with,
Profit4Cloud's `brainzz-documents` repo (an unrelated enterprise platform
project that also happens to serve models via vLLM/llama.cpp, among
other things). Any resemblance in domain shape - recipes, hardware-aware
defaults - is intentional inspiration, not shared code.

## Layout

```
fllame/            # The package. CLI (Typer), domain types, backends, stores.
  cli.py             # Entry point - `recipe`, `serve`, `status`, `stop` commands.
  domain/recipe.py   # The `Recipe` type and its validation.
  backends/           # `ServingBackend` seam; `vllm.py` is the only implementation.
  recipes/store.py   # Reads Recipes from a directory of hand-edited YAML files.
  state/store.py     # SQLite-backed tracking of backgrounded servers.
tests/              # pytest, one module per fllame/ module above.
examples/recipes/   # Sample recipe files, for reference - not loaded at runtime.
```

## Workflow conventions

- **Never commit to `main`.** Branch, then open a PR. `main` only moves
  via merge.
- **Discuss architecture before writing non-trivial code.** Sketch the
  design first (as a doc, in conversation, or both) and get it agreed
  before implementing anything that isn't a small, obvious fix.
- **Flag genuinely open decisions instead of inventing an answer.**
  Several things here are deliberately not implemented because the
  decision belongs to the project owner, not to whoever's writing the
  code (see "Explicitly deferred" below). Add the interface/hook so the
  decision can slot in later; don't guess a policy.
- **Docs must stand alone.** No references to "our conversation," no
  pointers to a PR that might get discarded, English only. Someone
  reading a doc in five years with zero conversational context should be
  able to follow it.

## Architecture, in one paragraph

A `Recipe` (`fllame/domain/recipe.py`) is a validated, immutable record
of how to serve one model handle: HF repo id, port, env vars, and extra
`vllm serve` flags. Recipes are loaded from plain YAML files
(`fllame/recipes/store.py`) that live in the *operator's* own directory,
not inside fllame - they're meant to be hand-edited and git-tracked the
same way a Helm `values.yaml` or an Ollama Modelfile is, not stored in a
database. `ServingBackend` (`fllame/backends/`) turns a `Recipe` into an
argv; `VllmServingBackend` is the only implementation and the only one
fllame ships. `fllame serve` either `exec`s that argv directly
(foreground - e.g. as a systemd unit's `ExecStart`) or backgrounds it and
records the PID/port in a local SQLite state file
(`fllame/state/store.py`) so `fllame status`/`fllame stop` can find it
again. That SQLite file is ephemeral machine state, not configuration -
deliberately not something an operator would hand-edit.

## Explicitly deferred (implemented as an interface/hook, not a concrete answer)

- **Non-vLLM backends** (llama.cpp, MLX, ...) - `ServingBackend` exists
  as a seam precisely so one could be added later, but fllame is
  vLLM-only by design today. Don't add a second implementation without a
  decision from the project owner about which framework and why.
- **The `image` recipe field and any container/Helm export** - recorded
  in the schema but unused by `fllame serve`, which assumes a local
  `vllm` install on `PATH`. Compiling a recipe into a
  [production-stack](https://github.com/vllm-project/production-stack)
  Helm `values.yaml` is the intended graduation route for multi-node/
  cluster use, but the compilation logic - and which of
  production-stack's routing/autoscaling options to default to - isn't
  designed yet.
- **Grafana/OpenWebUI integration recipes** - mentioned as a goal for
  this project; no scaffolding exists yet.
- **Hardware-aware recipe selection** - recipes are looked up by handle
  alone today; picking between multiple recipes for the same handle
  based on detected GPU/VRAM is not implemented.
- **Any multi-user or remote-access concern** (auth, RBAC, a server
  process) - fllame is a local CLI for a trusted single operator by
  design, not a service. If that assumption ever needs to change, that's
  a new decision, not an extension of the current code.

## Merged so far

- CLI scaffold: `recipe list`/`recipe show`, `serve` (foreground exec or
  `--detach`), `status`, `stop`.
- `Recipe` domain type + YAML-directory-backed `RecipeStore`.
- `VllmServingBackend`, the sole `ServingBackend` implementation.
- SQLite-backed `StateStore` for backgrounded servers.
