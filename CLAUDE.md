# Repo orientation

fllame is a headless CLI for serving models with vLLM: a curated recipe
registry (model handle -> HF repo id + Docker image + `vllm serve`
flags), a thin wrapper that compiles each recipe into its own self-
contained docker-compose folder and drives it, and a Hugging Face
download step that always runs before serving. No web UI, no admin
server, no database server to run, and nothing installed on the host
beyond fllame and Docker itself.

It is a fresh, narrow rewrite - not a fork of, and not code-shared with,
Profit4Cloud's `brainzz-documents` repo (an unrelated enterprise platform
project that also happens to serve models via vLLM/llama.cpp, among
other things). Any resemblance in domain shape - recipes, hardware-aware
defaults - is intentional inspiration, not shared code.

## Layout

```
fllame/            # The package. CLI (Typer), domain types, backends, generators.
  cli.py             # Entry point - `recipe`, `hardware`, `model`, `serve`, `status`, `stop`.
  domain/recipe.py   # The `Recipe` type and its validation.
  domain/hardware.py # The `HardwareProfile` type - a scan snapshot, never persisted.
  backends/           # `ServingBackend` seam; `vllm.py` is the only implementation,
                       # turning a Recipe into a docker-compose service definition.
  recipes/store.py   # Reads/writes Recipes from a directory of hand-edited YAML files.
  recipes/parser.py  # Parses a pasted export/vllm-serve block for `recipe add`.
  recipes/naming.py  # Derives a recipe handle from a repo_id.
  hardware/scanner.py # Live NVIDIA GPU/RAM detection (`nvidia-smi`, `/proc/meminfo`).
  compose/generator.py # Compiles one recipe into its own compose folder.
  models/puller.py   # Downloads a model into HF's own cache via huggingface_hub.
  models/cache.py    # Lists/verifies what's in that cache - a filesystem scan, no network.
  models/discovery.py # Searches the HF Hub for candidate models, ranked.
  models/sizing.py   # The coarse hardware memory budget `--max-size` defaults to.
  models/updater.py  # Checks a cached model against the Hub for a newer revision.
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
- **Keep this file current, not cumulative.** When behavior changes,
  update the section that describes that behavior in place - don't
  append a dated changelog entry. Git history is the changelog; this
  file should only ever describe the system as it is right now.

## Setup vs. running - a core principle

fllame's commands split cleanly into two phases, enforced as a hard
rule, not a preference:

- **Setup** - `recipe add`/`edit`/`remove`/`build`, `model pull`/`scan`/
  `update`, `hardware scan`, `config`. Network access belongs here.
  `pull_model` (`models/puller.py`) is called only from `model pull`,
  `recipe add --pull`, and `model update --apply`.
- **Running** - `serve`, `status`, `stop`. Must never touch the network,
  under any circumstance. `serve` only ever *verifies* what's already on
  disk (`models/cache.py`'s `is_model_cached`/`local_estimate_vram_gb`,
  pure filesystem scans, never `pull_model`) and fails with a pointer to
  `fllame model pull <repo_id>` if the model isn't cached - it never
  falls back to downloading, not even a "local-files-only" call. Calling
  a function with no networking code in it at all is a stronger
  guarantee than parameterizing the download function off, which is one
  accidental refactor away from a network call creeping back in.

Any future feature that needs the network for anything - a Hub lookup, a
version check, a richer VRAM estimate - belongs in a setup-phase
command, never inlined into `serve`/`status`/`stop`'s own code path. This
is what makes "pull everything while online, then run entirely offline"
an actual guarantee - `HF_HUB_OFFLINE=1` on the container (see
"Architecture" below) extends it to vLLM itself. `model update` exists
specifically because `serve` can structurally never check Hub staleness
itself - it's the only place that ever can.

## Layering - a second, orthogonal core principle

Independent of the network boundary above, fllame has a strict one-way
dependency direction between its two data concepts:

- **`model`** (`fllame/models/*`) is the lowest layer - no module here
  imports from `fllame.domain.recipe`, `fllame.recipes`, or `cli.py`.
  Every `model` command (`pull`/`list`/`scan`/`update`) takes a Hugging
  Face repo_id directly, with no notion that a recipe exists.
- **`recipe`** (`fllame.domain.recipe`, `fllame.recipes.*`) sits above
  `model` conceptually (it finds a model via `Recipe.repo_id`, parsed
  from `command`) but its own code doesn't import from `fllame.models`
  either - `cli.py`, one layer above both, is what wires them together
  (`recipe add --pull` loads a `Recipe`, reads its `repo_id`, then calls
  `pull_model` with it as a separate step - not `recipes/store.py`
  reaching into `models/` itself).
- **`serve`** (`cli.py`) consumes both, at the top: the recipe folder
  structure directly (to find/regenerate `compose.yaml`), the model only
  indirectly through the recipe's `repo_id`.

Concretely: `model pull`/`model update` never call `_load_or_exit`, touch
`RecipeStore`, or import from `fllame.domain.recipe`/`fllame.recipes` -
not even for a convenience shortcut. A `model` command's argument is
always a repo_id, never a recipe handle; resolving "which repo_id does
this recipe need" is `recipe show`/`recipe add --pull`'s job, one layer
up.

## Architecture

- **`Recipe`** (`domain/recipe.py`) - a validated, immutable record: a
  Docker `image` (optional, falls back to `fllame config`'s configured
  default), `gpus` (`all`|`none`), `env`, `preinstall` (shell commands
  run before `vllm serve`), and one `command` field holding the whole
  `vllm serve <repo_id> <args...>` line - not split into `repo_id`/args/
  port keys. `repo_id`/`serve_args`/`port` are derived properties parsed
  from `command` on access (`domain/vllm_command.py`), not stored twice.
  Rendered back out (`to_dict`/`to_yaml`) as a canonical one-flag-per-
  line block regardless of how it was authored, so it's exactly what
  gets copied out and run by hand.
- **Storage** (`recipes/store.py`) - one `recipe.yaml` per handle, in its
  own folder (`config.py`'s `recipe_dir(handle)`,
  `$FLLAME_RECIPES_DIR/<handle>/` by default) - the operator's own
  directory, hand-edited and git-tracked like a Helm `values.yaml`. That
  same folder holds the generated `compose.yaml` (see below), side by
  side. `RecipeStore.load` extracts `command`'s raw text before handing
  the rest to `yaml.safe_load`, so its own indentation/trailing-`\`
  leniency (`join_command_lines`) doesn't fight YAML's block-scalar
  strictness. `save`/`remove`/`next_available_handle` (a handle
  collision gets `_2`, `_3`, ... - never overwrites) round it out.
- **`recipe add`** - either a pasted `vllm serve ...` one-liner
  (`recipes/parser.py`, with `ignore_unknown_options=True` so any flag
  works verbatim as trailing args) or, given none, a guided dialogue
  (image, preinstall, env, command - each its own labeled step). Either
  way, anything outside that shape - a shell metacharacter/substitution
  in `env`/`command` - is a hard parse error; fllame parses this text
  itself rather than handing it to a real shell. `preinstall`/`RUN`
  lines are the deliberate exception, taken verbatim as shell text - with
  one narrow substitution: a preinstall line starting with `uv pip
  install` becomes plain `pip install` (the vllm/vllm-openai image's own
  Python environment isn't the uv-managed venv `uv pip install`
  expects), echoing a console note when it fires. Only `recipe add` does
  this - never `recipe edit`, `serve`, or `recipe build`.
- **`ServingBackend`** (`backends/`) - `VllmServingBackend` is the only
  implementation, turning a `Recipe` into a plain `compose.yaml` service
  (image, ports, volumes, `gpus`/`ipc`). With `preinstall` set, it wraps
  the container's command in a shell instead: `entrypoint: [sh, -c]`,
  running `<preinstall> && exec vllm serve ...` - `exec` so vLLM becomes
  PID 1 and receives `docker stop`'s SIGTERM directly. `preinstall`
  reruns on every container start (not cached like a Docker layer) - the
  cost of there being nothing else to build or manage.
- **`compose/generator.py`** - compiles one recipe into its own
  self-contained `compose.yaml` (own compose project, `fllame-<handle>`),
  written into that same folder. Fully regenerated by `serve`/`status`/
  `stop`/`recipe build` on every invocation; `recipe add`/`edit`/`remove`
  never write it. Being self-contained, the folder can be copied
  elsewhere and driven with plain `docker compose up -d`, no fllame
  involved. `recipe remove` deletes the whole folder.
- **The setup/running boundary, concretely**: before `docker compose up`,
  `serve` calls `is_model_cached` (a pure filesystem scan) - never
  `pull_model` - and fails with a pointer to `fllame model pull` if the
  model isn't there. That cache directory is bind-mounted into the
  container (its host side written as `${HOME}/...` when possible,
  rather than baking in one home directory, so `compose.yaml` stays
  correct on another machine/account). The container's `HF_HOME` and
  `HF_HUB_CACHE` are both set explicitly to that mount path -
  `HF_HUB_CACHE` isn't left to `huggingface_hub`'s own default derivation
  from `HF_HOME` (`HF_HOME/hub`, one directory below where the mount
  lands), so vLLM inside the container sees exactly the cache fllame just
  verified. `HF_HUB_OFFLINE=1` is unconditional on every service.
  `Recipe.env` rejects all three (`HF_HOME`/`HF_HUB_CACHE`/
  `HF_HUB_OFFLINE`) - fllame manages them, not the recipe. `gpus: "all"`/
  `ipc: "host"` are the same kind of hard default. Override any of these
  by hand-editing the generated `compose.yaml` (see README's "Advanced"
  section) - `serve`/`status`/`stop` fully regenerate it on every run, so
  such an edit only survives until the next one.
- **`HardwareProfile`** (`domain/hardware.py`) - a live NVIDIA GPU/RAM
  scan (`nvidia-smi`, `/proc/meminfo`), cheap enough to redo on demand
  rather than persist and risk going stale.
- **`model scan`** - `models/sizing.py` turns a `HardwareProfile` into a
  coarse memory budget; `models/discovery.py` searches the Hub within a
  `--max-size` ceiling (defaulted from that budget), ranked by
  estimated-VRAM fit/downloads/recency, using the Hub's own safetensors
  metadata where available. GGUF is excluded outright; MLX isn't (no
  serving story either way, just not evaluated). `discovery.py` itself
  has no dependency on `HardwareProfile`, so `--quant`/`--max-size` given
  explicitly skips hardware scanning entirely.
- **`serve`'s pre-flight checks** - `_warn_if_vram_likely_insufficient`
  compares `local_estimate_vram_gb` (real on-disk `.safetensors` size, no
  Hub lookup - a cruder, different estimate than `model scan`'s) against
  the hardware budget; `_warn_if_cache_location_changed` compares the
  cache directory about to be mounted against what's already baked into
  an existing `compose.yaml`. Both warn-and-confirm (`-y`/`--yes` skips
  the prompt, not the check) and silently skip whenever they don't have a
  confident answer - neither is a blocking verdict.
- **`model update [REPO_ID] [--apply]`** - the only place fllame ever
  asks the Hub whether a cached model is stale (`models/updater.py`'s
  `check_for_update`: local commit hash vs. `model_info(...).sha`).
  Check-only by default; `--apply` re-pulls stale repos through the same
  `pull_model` `model pull` uses (a no-op transfer if nothing changed).
- **Download error handling** - all three `pull_model` call sites
  (`model pull`, `model update --apply`, `recipe add --pull`) catch
  `HfHubHTTPError`/`RequestException` (a transient Hub failure) and
  `PermissionError` (a shared-cache ownership mismatch, e.g. from
  something else writing into the cache as a different user) separately,
  reporting an actionable message - resume by rerunning; the exact
  `chown` fix - instead of a raw traceback.
- **`config set-default-image`** - changing the default already applies
  to any recipe that doesn't pin its own `image`, on its next
  `serve`/`recipe build` (no migration needed). It also offers to
  text-substitute the old default for the new one across any existing
  `compose.yaml` whose `image:` is a literal, exact match - never a
  regeneration, so hand edits elsewhere in those files survive.

## Explicitly deferred (implemented as an interface/hook, not a concrete answer)

- **Non-vLLM backends** (llama.cpp, MLX, ...) - `ServingBackend` exists
  as a seam precisely so one could be added later, but fllame is
  vLLM-only by design today. Don't add a second implementation without a
  decision from the project owner about which framework and why.
- **Sibling services alongside a recipe's `compose.yaml`** (Grafana,
  OpenWebUI, ...) - each recipe's folder holds only that one
  fllame-managed service and fully overwrites it on every run. A
  hand-written compose file pulling recipes' generated ones in via
  Compose's `include:` is the likely shape, but isn't built - don't have
  `serve`/`status`/`stop` silently merge into or preserve unrelated
  hand-added services until that's designed.
- **A Helm export target for multi-node scale** - compiling a recipe
  into a [production-stack](https://github.com/vllm-project/production-stack)
  `values.yaml` is the intended graduation route beyond a single Docker
  host, but neither the compilation logic nor which routing/autoscaling
  options to default to is designed yet.
- **Hardware-aware recipe selection** - recipes are looked up by handle
  alone; picking between multiple recipes for the same handle based on
  detected GPU/VRAM is not implemented.
- **A real, recipe-level VRAM estimator** (params + quantization +
  `--max-model-len` + concurrency, for weights and KV cache both) -
  likely surfacing as `fllame recipe vram-usage`. Not attempted yet; a
  prior project's equivalent turned out to be a flat safety margin, not a
  real estimate, and isn't worth copying. Don't confuse this with
  `models/sizing.py`'s cruder, search-narrowing ceiling - the two are
  intentionally separate and shouldn't merge without this being designed
  properly first.
- **A real, blocking pre-flight OOM guard on `serve`** - blocked on the
  estimator above: a wrong "fits" verdict from a weights-only figure
  would be worse than none. `_warn_if_vram_likely_insufficient` (see
  "Architecture") is the deliberately narrower thing that exists today -
  a heads-up, not this verdict, and shouldn't be mistaken for it.
- **Injecting `--gpu-memory-utilization` at serve time from the hardware
  scan.** Decided, not yet built: not a `Recipe`/`serve_args` concern - a
  recipe is never required to set it, and nothing normalizes it into the
  file. The intended design is for `serve` to compute a safe value from
  `scan_hardware()` at launch time and add it to the generated command,
  not persist it - recomputed per machine. Until built, a recipe that
  omits it gets whatever vLLM's own default is; one that sets it
  explicitly is used as-is.
- **Any multi-user or remote-access concern** (auth, RBAC, a server
  process) - fllame is a local CLI for a trusted single operator by
  design, not a service. If that assumption ever needs to change, that's
  a new decision, not an extension of the current code.
