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
  models/cache.py    # Lists what's in that cache - a filesystem scan, no network.
  models/discovery.py # Searches the HF Hub for candidate models, ranked.
  models/sizing.py   # The coarse hardware memory budget `--max-size` defaults to.
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
of how to serve one model handle: a Docker image, GPU reservation, env
vars, a preinstall step, and - as one field, `command` - the whole
`vllm serve <repo_id> <args...>` invocation, not split into
`repo_id`/args/port keys - rendered (`Recipe.to_dict`/`to_yaml`, used by
both `RecipeStore.save` and `recipe show`) as a canonical one-flag-
per-line block (`domain/vllm_command.py`'s `render_multiline_command`)
regardless of how it was originally authored, so it's exactly what gets
copied out and run by hand. `repo_id`/`serve_args`/`port` are derived
properties parsed from it on access (`fllame/domain/vllm_command.py`),
not stored a second time. `RecipeStore.load` never hands `command`'s
raw text to `yaml.safe_load` at all (`_extract_command_section`
carves it out first, from the `command:` line to the next blank
line/unindented key/EOF) - a YAML literal block scalar needs
consistent, sufficient indentation on every line to stay valid YAML at
all, which is exactly what's easy to break by hand (deleting what
looks like meaningless leading whitespace); parsing it instead with
`domain/vllm_command.py`'s own `join_command_lines` (leading/trailing
whitespace and a trailing `\` all optional, blank/`#`-comment lines
dropped) avoids that fragility entirely. Recipes are loaded
from plain YAML files (`fllame/recipes/store.py`) that live in the
*operator's* own directory, not inside fllame - they're meant to be
hand-edited and git-tracked the same way a Helm `values.yaml` or an
Ollama Modelfile is, not stored in a database. `RecipeStore` also
writes them: `save()` (used by `recipe add`), `remove()`, and
`next_available_handle()` (handle collision -> `_2`, `_3`, ... - never
overwrites, even for a second recipe on the same repo_id, since that's
a legitimate way to keep more than one tuning of a model around).
`fllame/recipes/parser.py` turns that same text - zero or more `export
KEY=VALUE` lines, zero or more `RUN <command>` lines, exactly one `vllm
serve <repo_id> <args...>` line, the shape a recipe typically comes in
from a model card - into that data, when `recipe add` gets it as
trailing CLI arguments joined via `shlex.join`
(`ignore_unknown_options=True` on that one command's `context_settings`,
so a pasted `vllm serve ... --flag value` works verbatim as trailing
args without every flag needing to be a recognized fllame option - the
exact case that motivated adding this path: a user tried the natural
"run this as if it were a shell command" shape first). With no trailing
arguments instead, `recipe add` walks through a guided dialogue instead
of parsing a single pasted blob - image, then preinstall commands, then
env vars, then the `vllm serve` command, each its own labeled step
(`cli.py`'s `_read_block`/`_read_command`, plus `parser.py`'s
`parse_env_line` for the bare `KEY=VALUE` shape that step collects, no
`export` keyword needed since the step is unambiguous on its own).
Anything else on a line, or a shell metacharacter/substitution in an
`export` value or the `vllm serve` line, is a hard parse error either
way - fllame parses this text itself rather than handing it to a real
shell, so it never silently evaluates something dangerous; a `RUN`
line/the dialogue's preinstall step are the deliberate exception, taken
verbatim as shell text since that's what they genuinely are.
`fllame/recipes/naming.py`
derives the handle recipe `add` uses from the repo_id (the part after
the last `/`, slugified). `ServingBackend`
(`fllame/backends/`) turns a `Recipe` into a docker-compose service
definition (image, entrypoint/command, ports, volumes, GPU reservation)
- always a plain, self-contained `docker-compose.yml`, no separate
build step. When `preinstall` is non-empty, `VllmServingBackend` (the
only implementation, and the only one fllame ships) wraps the
container's own command in a shell instead: `entrypoint: [sh, -c]`,
`command: ["<preinstall> && exec vllm serve ..."]` - the vLLM portion
is still built with `shlex.join` (so a repo_id/flag value containing
shell syntax stays a literal argument, not smuggled shell), while
`preinstall` entries are joined in as-is, since they're genuinely meant
to be shell text (see `Recipe.preinstall`'s docstring); `exec` on the
final command replaces the shell with vLLM itself, so it becomes PID 1
and receives `docker stop`'s SIGTERM directly. The trade-off, deliberate:
`preinstall` reruns on every container start (not cached the way a
Docker image layer would be) in exchange for there being nothing to
build or manage beyond the one compose file.
`fllame/compose/generator.py` compiles one recipe into its own compose
folder (`fllame/config.py`'s `recipe_state_dir(handle)`,
`$FLLAME_STATE_DIR/recipes/<handle>/` by default) as its own compose
project (`compose_project_name(handle)`, `fllame-<handle>`), fully
owned and regenerated by fllame on every command that needs it,
entirely independent of every other recipe's; being self-contained,
a folder can also be copied elsewhere and driven with plain `docker
compose up -d`, no fllame CLI involved. `fllame serve`/`status`/`stop`
shell out to `docker compose` against HANDLE's own folder (`up`/`ps`/
`stop`) - `status` loops over every recipe, one invocation each - so
container lifecycle state is whatever Docker already tracks; fllame
keeps none of its own. `recipe remove`
deletes both the recipe file and its generated folder (not a container
still running under it). Before
`serve` ever calls `docker compose up`, it calls
`fllame/models/puller.py` (`huggingface_hub.snapshot_download`) to
guarantee the model is fully present in HF's own cache - vLLM's own
auto-download inside the container is never relied on - and that same
host cache directory is bind-mounted into the container.
`fllame/models/cache.py` (`huggingface_hub.scan_cache_dir`) is the
read-only counterpart, backing `fllame model list`.

`serve --offline` forces the download step's `local_files_only=True`
(failing fast with a clear error if the model isn't fully cached, rather
than a plain download call's network-then-fallback-to-cache behavior,
which isn't fast or fully deterministic on a genuinely offline machine)
and sets `HF_HUB_OFFLINE=1` on that one generated service's environment
- an invocation-time concern applied in `cli.py` when it (re)writes the
compose file, not a property threaded through `Recipe`/`ServingBackend`.
This is what makes "pull while online, `serve --offline` later with no
network at all" a real guarantee rather than a hope.

`HardwareProfile` (`fllame/domain/hardware.py`) is a fourth, distinct
kind of data: neither hand-edited config nor container state, just the
result of a live scan (`fllame/hardware/scanner.py`, NVIDIA GPU via
`nvidia-smi` + RAM via `/proc/meminfo`) that's cheap enough to redo each
time it's needed rather than persist and risk going stale.

`fllame model scan` composes two independent pieces: `models/sizing.py`
turns a `HardwareProfile` into a coarse memory budget (no context length
or concurrency in this estimate, see "Explicitly deferred" below), and
`models/discovery.py` searches the HF Hub within a `--max-size` (GB)
ceiling defaulted from that budget, ranked by estimated-VRAM fit,
downloads, and recency. A candidate's param count and estimated VRAM
both come from the Hub's own safetensors metadata where available
(`ModelInfo.safetensors.total` for params - the same figure shown on
the model page as "Model size"; a direct sum of
`ModelInfo.safetensors.parameters`' per-dtype byte widths for VRAM, not
a per-quantization bytes-per-param guess, since a declared param count
alone doesn't reveal how a given quantization format packs its bits on
disk) - falling back to a unit-aware guess from the repo_id's naming
convention for params only (there's no repo_id-based fallback for VRAM)
when safetensors metadata is absent. `--min-params`/`--max-params` are
a second, independent restriction on top with no hardware-derived
default of their own. Either restriction excludes a candidate it can't
verify (no safetensors metadata at all) only when given explicitly, not
when merely defaulted - see `discovery.search_models`'s
`exclude_unknown_size` for why. `discovery.py` takes already-resolved
quantizations and a plain `max_size_gb` float as arguments - it has no
dependency on `HardwareProfile` or hardware scanning at all, which is
what lets `cli.py` bypass hardware entirely when `--quant`/`--max-size`
are both given explicitly (verified live: that path never calls
`scan_hardware()`, going straight to the Hub search). GGUF results are
excluded outright (`discovery._is_gguf`): GGUF-via-vLLM now needs a
separate out-of-tree plugin with no per-model compatibility guarantee,
on top of carrying no safetensors metadata for either estimate above to
work with. MLX isn't filtered - fllame has no MLX serving story at all,
unlike GGUF's (limited, unreliable) one, so it hasn't been evaluated on
its own merits. Adapted from the same admin-ui project's Hub-search
logic mentioned above, trimmed to fllame's scope: no training/LoRA
headroom.

## Explicitly deferred (implemented as an interface/hook, not a concrete answer)

- **Non-vLLM backends** (llama.cpp, MLX, ...) - `ServingBackend` exists
  as a seam precisely so one could be added later, but fllame is
  vLLM-only by design today. Don't add a second implementation without a
  decision from the project owner about which framework and why.
- **Sibling services alongside a recipe's generated compose folder**
  (Grafana, OpenWebUI, ...) - each recipe's folder
  (`fllame/config.py`'s `recipe_state_dir`) holds only that one
  fllame-managed service, and fully overwrites it on every run. A
  hand-written compose file that pulls one or more recipes' generated
  ones in via Compose's `include:` directive is the likely shape, but
  isn't built - don't have `serve`/`status`/`stop` silently merge into
  or preserve unrelated hand-added services until that's designed.
- **A container/Helm export target for multi-node scale** - compiling a
  recipe into a [production-stack](https://github.com/vllm-project/production-stack)
  Helm `values.yaml` is the intended graduation route beyond a single
  Docker host, but the compilation logic - and which of
  production-stack's routing/autoscaling options to default to - isn't
  designed yet.
- **Hardware-aware recipe selection** - recipes are looked up by handle
  alone today; picking between multiple recipes for the same handle
  based on detected GPU/VRAM is not implemented.
- **A real, recipe-level VRAM estimator** (params + quantization +
  `--max-model-len` + max concurrency, for weights and KV cache both) -
  likely surfacing as `fllame recipe vram-usage`. Deliberately not
  attempted yet; the admin-ui project's equivalent turned out to be a
  flat safety margin on on-disk model size, not a real estimate, and
  isn't worth copying. Don't confuse this with `models/sizing.py`'s
  ceiling, which is a much cruder, declared-params-only estimate built
  to narrow a Hub search, not to confirm a specific recipe fits - the
  two are intentionally separate and shouldn't gradually merge into each
  other without this being designed properly first.
- **A pre-flight OOM guard on `fllame serve`** - `fllame hardware scan`
  exists and reports what a box can run, but `serve` doesn't yet cross-
  check a recipe against it before launching. Blocked on the estimator
  above: a wrong "fits" verdict is worse than no verdict.
- **Injecting `--gpu-memory-utilization` at serve time from the
  hardware scan.** Decided, not yet built: this is deliberately *not* a
  `Recipe`/`serve_args` concern - a recipe (hand-written or from `recipe
  add`) is never required to set it, and nothing normalizes or defaults
  it into the recipe file. The intended design is for `fllame serve` to
  compute a safe value from `scan_hardware()` at launch time and add it
  to the generated compose service's command, the same invocation-time
  pattern `--offline` already uses for `HF_HUB_OFFLINE` in `cli.py` -
  not persisted, recomputed per machine. Until this exists, a recipe
  that omits `--gpu-memory-utilization` gets whatever vLLM's own default
  is; one that sets it explicitly (e.g. from a paste that already had
  it) is used as-is.
- **Any multi-user or remote-access concern** (auth, RBAC, a server
  process) - fllame is a local CLI for a trusted single operator by
  design, not a service. If that assumption ever needs to change, that's
  a new decision, not an extension of the current code.

## Merged so far

- CLI scaffold: `recipe list`/`show`/`add`/`edit`/`remove`, `hardware
  scan`, `model pull`/`list`/`scan`, `serve` (foreground, `--detach`,
  `--offline`), `status`, `stop` - `-h` works as a `--help` alias at
  every level (set via `context_settings` on each `Typer()` instance;
  Click only binds `--help` by default).
- `recipe add`: recipe creation from a `vllm serve` line, either as
  trailing CLI arguments (a quick one-liner, `recipes/parser.py` +
  `recipes/naming.py` + `RecipeStore.save`/`next_available_handle`) or,
  with no trailing arguments, a guided dialogue (image, then preinstall
  commands, then env vars, then the `vllm serve` command - each its own
  labeled step, not one undifferentiated stdin paste). `recipe edit`
  opens `$EDITOR` (`click.edit(filename=...)`, edits the file in place)
  and re-validates on save - `command`'s own indentation/trailing-`\`
  leniency (`RecipeStore.load`/`_extract_command_section`) handles the
  most common breakage on its own; a narrow whitespace autofix runs
  next for anything else (CRLF, tab indentation, trailing whitespace;
  `recipes/store.py`'s `autofix_whitespace`), and anything still invalid
  offers a choice to reopen `$EDITOR` or revert to the pre-edit version
  (kept in memory, not a backup file - the recipes directory is
  git-tracked already). Either way, a successful revalidation re-saves
  the file in fllame's own canonical rendering. `recipe remove` deletes
  with a confirmation prompt (`-y` to skip it).
- **Install with `pipx install .`, not `poetry install`, for everyday
  use.** `poetry install` only creates a project-local venv; the `fllame`
  command it produces isn't on `PATH` outside `poetry run`/`poetry
  shell`. `pipx install .` builds via the same `poetry-core` backend
  (`[tool.poetry.scripts]` needed no changes) but installs into its own
  isolated venv with a `PATH` shim - the standard way to install a
  Python CLI tool globally. `poetry install` stays the right command for
  developing fllame itself. `pipx install --editable .` covers both at
  once - same global `fllame` command, but the venv's `.pth` points
  straight at the checkout instead of a frozen copy, so source edits are
  picked up on the next invocation with no reinstall; verified this
  live, including that it isn't defeated by a stale `__pycache__` .pyc.
- `Recipe` domain type (`command` holds the whole `vllm serve ...`
  line, Docker image optional - falls back to `fllame config`'s
  configured default, `gpus: all|none`, `env` rejects `HF_HOME`,
  `preinstall` for a shell setup step before `vllm serve`) + YAML-
  directory-backed `RecipeStore`.
- `VllmServingBackend`, the sole `ServingBackend` implementation -
  compiles a `Recipe` into a docker-compose service definition, always
  a plain `docker-compose.yml` with no separate build step; when
  `preinstall` is set, wraps the container's own command in a shell
  (`sh -c "<preinstall> && exec vllm serve ..."`) instead of building a
  custom image.
- `fllame/compose/generator.py` - compiles one recipe into its own
  self-contained compose folder/project; `serve`/`status`/`stop` drive
  it via `docker compose up|ps|stop` instead of fllame tracking its own
  state.
- `fllame/models/puller.py` + `models/cache.py` - download and list via
  `huggingface_hub`; `pull_model(..., offline=True)` and `serve
  --offline` together guarantee a genuinely offline demo after an online
  `model pull`.
- `HardwareProfile` + live NVIDIA GPU/RAM scanning, unconnected to
  recipes or `serve` so far (see "Explicitly deferred").
- `fllame model scan` - `models/sizing.py` (a coarse hardware memory
  budget from a `HardwareProfile`) + `models/discovery.py` (Hub search,
  ranked by estimated-VRAM fit/downloads/recency) via `cli.py`.
  `--max-size` (GB) is the always-enforced size gate, defaulted from
  that budget unless given explicitly; `--quant` independently opts out
  of the hardware-detected quantization list; `--min-params`/
  `--max-params` are a separate, optional restriction with no
  hardware-derived default of their own. A candidate's params/VRAM come
  from the Hub's own safetensors metadata where available, not a
  per-quantization bytes-per-param guess (see the architecture
  paragraph above). GGUF results are excluded outright; MLX isn't.
