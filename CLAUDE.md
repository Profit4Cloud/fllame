# Repo orientation

fllame is a headless CLI for serving models with vLLM: a recipe registry
(handle -> HF repo id + Docker image + `vllm serve` flags), a compiler
from recipe to self-contained docker-compose folder, and a download step
that always runs before serving. No web UI, no admin server, no
database. Nothing on the host beyond fllame and Docker.

Fresh, narrow rewrite - not a fork of, not code-shared with,
Profit4Cloud's `brainzz-documents` (an unrelated platform that also
serves via vLLM/llama.cpp). Domain-shape resemblance (recipes,
hardware-aware defaults) is inspiration, not shared code.

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

- **Never commit to `main`.** Branch, then PR. `main` moves only via
  merge.
- **Discuss architecture before non-trivial code.** Agree the design
  first, in a doc or conversation, before implementing anything beyond a
  small, obvious fix.
- **Flag open decisions, don't invent answers.** Several things here are
  deliberately unimplemented - the call is the project owner's (see
  "Explicitly deferred"). Add the hook, not a guessed policy.
- **Docs stand alone.** No "our conversation," no pointers to a
  discardable PR, English only. Readable with zero conversational
  context.
- **Keep this file current, not cumulative.** Update the relevant
  section in place when behavior changes - don't append a changelog
  entry. Git history is the changelog.

## Setup vs. running - a core principle

Two phases, hard rule:

- **Setup** - `recipe add`/`edit`/`remove`/`build`, `model pull`/`scan`/
  `update`, `hardware scan`, `config`. Network allowed. `pull_model`
  (`models/puller.py`) is called only from `model pull`, `recipe add
  --pull`, `model update --apply`.
- **Running** - `serve`, `status`, `stop`. Never touch the network.
  `serve` only *verifies* disk state (`models/cache.py`'s
  `is_model_cached`/`local_estimate_vram_gb`, pure filesystem scans,
  never `pull_model`); fails with a `fllame model pull <repo_id>` pointer
  if uncached - never falls back to downloading. A function with no
  networking code can't regress into one; a flag-gated download function
  can.

Any network-needing feature - Hub lookup, version check, richer VRAM
estimate - goes in setup, never `serve`/`status`/`stop`. This is what
makes "pull online, run fully offline" a guarantee - `HF_HUB_OFFLINE=1`
on the container extends it to vLLM. `model update` exists because
`serve` structurally can't check Hub staleness itself.

## Layering - a second, orthogonal core principle

Separate, one-way dependency direction:

- **`model`** (`fllame/models/*`) - lowest layer. No module here imports
  `fllame.domain.recipe`, `fllame.recipes`, or `cli.py`. Every `model`
  command takes a repo_id directly, no notion of recipes.
- **`recipe`** (`fllame.domain.recipe`, `fllame.recipes.*`) - above
  `model` conceptually (finds a model via `Recipe.repo_id`), but its own
  code doesn't import `fllame.models` either. `cli.py` wires the two:
  `recipe add --pull` loads a `Recipe`, reads `repo_id`, calls
  `pull_model` as a separate step.
- **`serve`** (`cli.py`) - consumes both: the recipe folder directly, the
  model only via the recipe's `repo_id`.

Concretely: `model pull`/`model update` never touch `RecipeStore` or
import from `fllame.domain.recipe`/`fllame.recipes`. Argument is always a
repo_id, never a handle; resolving "which repo_id" is `recipe
show`/`recipe add --pull`'s job.

## Architecture

- **`Recipe`** (`domain/recipe.py`) - immutable record: `image`
  (optional, falls back to `fllame config`'s default), `gpus`
  (`all`|`none`), `env`, `preinstall` (shell commands before `vllm
  serve`), and `command` - the whole `vllm serve <repo_id> <args...>`
  line, not split into keys. `repo_id`/`serve_args`/`port` are derived
  from `command` (`domain/vllm_command.py`), not stored twice. Rendered
  (`to_dict`/`to_yaml`) as one-flag-per-line - exactly copy-pasteable.
- **Storage** (`recipes/store.py`) - one `recipe.yaml` per handle folder
  (`config.py`'s `recipe_dir(handle)`, default
  `$FLLAME_RECIPES_DIR/<handle>/`) - operator-owned, hand-edited,
  git-tracked like a Helm `values.yaml`. Same folder holds the generated
  `compose.yaml`. `RecipeStore.load` extracts `command`'s raw text before
  `yaml.safe_load`, so its whitespace/trailing-`\` leniency doesn't fight
  YAML's block-scalar strictness. `save`/`remove`/`next_available_handle`
  (collision -> `_2`, `_3`... never overwrite) round it out.
- **`recipe add`** - a pasted `vllm serve ...` one-liner
  (`recipes/parser.py`, `ignore_unknown_options=True`) or, given none, a
  guided dialogue (image, preinstall, env, command). A shell
  metacharacter/substitution in `env`/`command` is a hard parse error -
  fllame parses this itself, never hands it to a shell. `preinstall`/
  `RUN` lines are verbatim shell text, with one fix: a leading `uv pip
  install` becomes `pip install` (vllm/vllm-openai's Python env isn't the
  uv venv it expects), with a console note. Only `recipe add` does this -
  not `edit`, `serve`, or `build`.
- **`ServingBackend`** (`backends/`) - `VllmServingBackend`, the only
  implementation. Turns a `Recipe` into a `compose.yaml` service (image,
  ports, volumes, `gpus`/`ipc`). With `preinstall`: `entrypoint: [sh,
  -c]` running `<preinstall> && exec vllm serve ...` - `exec` so vLLM is
  PID 1, gets SIGTERM directly. `preinstall` reruns every start (not
  image-cached) - the cost of nothing else to build.
- **`compose/generator.py`** - compiles a recipe into its own
  `compose.yaml` (own project, `fllame-<handle>`), same folder.
  Regenerated by `serve`/`status`/`stop`/`recipe build` every invocation;
  `recipe add`/`edit`/`remove` never write it. Self-contained: copy the
  folder, run `docker compose up -d`, no fllame needed. `recipe remove`
  deletes the whole folder.
- **Setup/running, concretely**: before `docker compose up`, `serve`
  calls `is_model_cached` (filesystem scan, never `pull_model`) - fails
  with a `model pull` pointer if uncached. The cache dir bind-mounts into
  the container (`${HOME}/...`-relative when possible, for portability).
  Container `HF_HOME` and `HF_HUB_CACHE` both point at that mount - not
  left to `huggingface_hub`'s own `HF_HOME/hub` derivation, which lands
  one directory below the mount and would find nothing. `HF_HUB_OFFLINE=1`
  is unconditional. `Recipe.env` rejects all three - fllame owns them.
  `gpus: "all"`/`ipc: "host"` are the same kind of hard default. Override
  by hand-editing `compose.yaml` (README's "Advanced") - but
  `serve`/`status`/`stop` regenerate it every run, so an edit lasts one
  run.
- **`HardwareProfile`** (`domain/hardware.py`) - live NVIDIA GPU/RAM scan
  (`nvidia-smi`, `/proc/meminfo`). Cheap enough to redo on demand; never
  persisted.
- **`model scan`** - `models/sizing.py` turns a `HardwareProfile` into a
  memory budget; `models/discovery.py` searches the Hub within a
  `--max-size` ceiling (defaulted from that budget), ranked by
  estimated-VRAM fit/downloads/recency, from Hub safetensors metadata.
  GGUF excluded; MLX not (no serving story, unevaluated). `discovery.py`
  has no `HardwareProfile` dependency - explicit `--quant`/`--max-size`
  skips hardware scanning.
- **`serve`'s pre-flight checks** - `_warn_if_vram_likely_insufficient`:
  `local_estimate_vram_gb` (on-disk `.safetensors` size, no Hub lookup -
  cruder than `model scan`'s estimate) vs. hardware budget.
  `_warn_if_cache_location_changed`: the cache dir about to mount vs.
  what's already baked into `compose.yaml`. Both warn-and-confirm (`-y`
  skips the prompt, not the check), silent when unconfident. Neither
  blocks.
- **`model update [REPO_ID] [--apply]`** - the only place fllame asks the
  Hub if a cached model is stale (`models/updater.py`: local commit hash
  vs. `model_info(...).sha`). Check-only by default; `--apply` re-pulls
  stale repos via `pull_model` (no-op transfer if unchanged).
- **Download errors** - all three `pull_model` call sites (`model pull`,
  `model update --apply`, `recipe add --pull`) catch `HfHubHTTPError`/
  `RequestException` (transient Hub failure) and `PermissionError`
  (shared-cache ownership mismatch) separately - actionable message
  (rerun to resume; exact `chown` fix), not a raw traceback.
- **`config set-default-image`** - applies automatically to any recipe
  without a pinned `image`, next `serve`/`recipe build`. Also offers a
  literal text-substitute of old->new default across existing
  `compose.yaml` files with an exact-match `image:` - never a
  regeneration, so other hand edits survive.

## Explicitly deferred (implemented as an interface/hook, not a concrete answer)

- **Non-vLLM backends** (llama.cpp, MLX, ...) - `ServingBackend` is a
  seam for this, but fllame is vLLM-only today. No second implementation
  without an owner decision on which and why.
- **Sibling services** (Grafana, OpenWebUI, ...) alongside a recipe's
  `compose.yaml` - each folder holds one fllame-managed service, fully
  overwritten every run. A hand-written file pulling recipes in via
  Compose `include:` is the likely shape, not built. Don't have
  `serve`/`status`/`stop` merge or preserve unrelated services until
  designed.
- **A Helm export target for multi-node scale** - compiling a recipe to
  a [production-stack](https://github.com/vllm-project/production-stack)
  `values.yaml` is the intended route beyond one Docker host. Neither the
  compilation logic nor default routing/autoscaling options are
  designed.
- **Hardware-aware recipe selection** - recipes are looked up by handle
  alone. Picking between multiple recipes per handle by detected
  GPU/VRAM isn't implemented.
- **A real, recipe-level VRAM estimator** (params + quantization +
  `--max-model-len` + concurrency, weights and KV cache) - likely
  `fllame recipe vram-usage`. Not attempted; a prior project's version
  was a flat safety margin, not worth copying. Not the same as
  `models/sizing.py`'s cruder, search-narrowing ceiling - keep them
  separate until this is designed.
- **A blocking pre-flight OOM guard on `serve`** - blocked on the
  estimator above: a wrong "fits" verdict beats none.
  `_warn_if_vram_likely_insufficient` (see "Architecture") is a
  heads-up, not this verdict.
- **Injecting `--gpu-memory-utilization` from the hardware scan at serve
  time.** Decided, not built: not a `Recipe`/`serve_args` field - `serve`
  should compute it from `scan_hardware()` at launch and append to the
  command, recomputed per machine, never persisted. Until then: vLLM's
  own default applies unless the recipe sets it explicitly.
- **Multi-user or remote-access concerns** (auth, RBAC, a server
  process) - fllame is a local CLI for one trusted operator, not a
  service. Changing that is a new decision, not an extension.
