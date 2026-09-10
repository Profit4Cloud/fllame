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

## Setup vs. running - a core principle

fllame's commands split cleanly into two phases, and the boundary
between them is a hard rule, not a preference:

- **Setup** - `recipe add`/`edit`/`remove`/`build`, `model pull`/`scan`/
  `update`, `hardware scan`, `config`. Network access belongs here.
  `models/puller.py`'s `pull_model` (downloading a model) is called
  from `fllame model pull`, `recipe add HANDLE --pull` (also reachable
  via `recipe add --pull --build`), and `model update --apply` (which
  re-pulls a stale model through this same function - a no-op
  download-wise if nothing actually changed). `model update` is this
  boundary's whole reason to exist: `serve` can never notice a cached
  model has gone stale on the Hub, since it never checks the Hub at all
  (see "Running" below) - `model update` is the only place that ever
  can.
- **Running** - `serve`, `status`, `stop`. These must never touch the
  network, under any circumstance, no exceptions. `serve` in particular
  only ever *verifies* what's already on disk - `models/cache.py`'s
  `is_model_cached`/`local_estimate_vram_gb`, both pure filesystem scans
  (`huggingface_hub.scan_cache_dir`), never `models/puller.py`'s
  `pull_model`. If the model isn't fully cached, `serve` fails
  immediately with a message pointing at `fllame model pull <repo_id>`
  (`cli.py`'s `_require_model_cached`) - it never falls back to
  downloading it, not even a "local-files-only" download call. The
  distinction matters: parameterizing the download function not to use
  the network is still one accidental refactor away from a network call
  creeping back in; calling a function that has no networking code in it
  at all cannot regress that way.

Any future feature that needs the network for anything - a Hub lookup,
a version check, a richer VRAM estimate - belongs in a setup-phase
command. It must never be added to `serve`/`status`/`stop`'s own code
path, no matter how convenient inlining it there would be. This is what
makes "pull everything while online, then run entirely on a genuinely
offline machine" an actual guarantee rather than a best effort - see
the architecture paragraph below for how `HF_HUB_OFFLINE=1` on the
container extends the same guarantee to vLLM itself.

## Layering - a second, orthogonal core principle

Independent of the network boundary above, fllame has a strict
dependency direction between its two data concepts, and it runs one way
only:

- **`model`** (`fllame/models/*` - `puller.py`, `cache.py`,
  `discovery.py`, `sizing.py`, `updater.py`) is the lowest layer. It
  knows nothing above itself: no module under `models/` imports
  anything from `fllame.domain.recipe`, `fllame.recipes`, or `cli.py`.
  Every `model` CLI command (`model pull`/`list`/`scan`/`update`) takes
  a Hugging Face repo_id directly and does its job with no notion that
  a recipe exists at all.
- **`recipe`** (`fllame.domain.recipe`, `fllame.recipes.*`) sits above
  `model` conceptually - a recipe's whole job is finding a model
  (`Recipe.repo_id`, parsed from its `command`) and describing how to
  serve it - but the recipe *code itself* doesn't import from
  `fllame.models` either; it's a self-contained, lower-level building
  block. It's `cli.py` - the one layer above both - that wires the two
  together: `recipe add --pull` loads a `Recipe`, reads its `repo_id`,
  and only then calls `models/puller.py`'s `pull_model` with that
  repo_id, as two separate steps in sequence, not `recipes/store.py`
  reaching into `models/` itself.
- **`serve`** (also in `cli.py`) is a consumer of both, at the top:
  it needs the *recipe* folder structure to find/regenerate
  `compose.yaml` next to `recipe.yaml`, and it needs the *model* to be
  present in the cache before it can run - reaching `model` only
  indirectly, through the recipe's `repo_id`.

Concretely, this means `model pull`/`model update` must never call
`_load_or_exit`, touch `RecipeStore`, or import anything from
`fllame.domain.recipe`/`fllame.recipes` - not even to offer a
convenience shortcut. A `model` command's argument is always a
repo_id, never a recipe handle, full stop; resolving "which repo_id
does this recipe need" is `recipe show HANDLE`/`recipe add --pull`'s
job, one layer up, not something `model pull`/`model update` do on
their own behalf.

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
dropped) avoids that fragility entirely. Recipes are loaded from plain
YAML files (`fllame/recipes/store.py`), one `recipe.yaml` inside each
handle's own folder (`fllame/config.py`'s `recipe_dir(handle)`,
`$FLLAME_RECIPES_DIR/<handle>/` by default) - live in the *operator's*
own directory, not inside fllame, and meant to be hand-edited and
git-tracked the same way a Helm `values.yaml` or an Ollama Modelfile
is, not stored in a database. That same folder is also where this
handle's generated `compose.yaml` ends up (see below) - the two sit
side by side, one hand-edited and one fllame-owned, rather than in
separate trees. `RecipeStore` also writes them: `save()` (used by
`recipe add`), `remove()`, and
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
definition (image, entrypoint/command, ports, volumes, `gpus`/`ipc`) -
always a plain, self-contained `compose.yaml`, no separate build step.
When `preinstall` is non-empty, `VllmServingBackend` (the only
implementation, and the only one fllame ships) wraps the container's
own command in a shell instead: `entrypoint: [sh, -c]`, `command:
["<preinstall> && exec vllm serve ..."]` - the vLLM portion is still
built with `shlex.join` (so a repo_id/flag value containing shell
syntax stays a literal argument, not smuggled shell), while
`preinstall` entries are joined in as-is, since they're genuinely meant
to be shell text (see `Recipe.preinstall`'s docstring); `exec` on the
final command replaces the shell with vLLM itself, so it becomes PID 1
and receives `docker stop`'s SIGTERM directly. The trade-off, deliberate:
`preinstall` reruns on every container start (not cached the way a
Docker image layer would be) in exchange for there being nothing to
build or manage beyond the one compose file.
`fllame/compose/generator.py` compiles one recipe into its own
`compose.yaml`, written into that same handle's folder
(`fllame/config.py`'s `recipe_dir(handle)`, right next to its
`recipe.yaml` - see above) as its own compose project
(`compose_project_name(handle)`, `fllame-<handle>`), fully owned and
regenerated by fllame on every command that needs it, entirely
independent of every other recipe's; being self-contained, a folder can
also be copied elsewhere and driven with plain `docker compose up -d`,
no fllame CLI involved. `fllame serve`/`status`/`stop` shell out to
`docker compose` against HANDLE's own folder (`up`/`ps`/`stop`) -
`status` loops over every recipe, one invocation each - so container
lifecycle state is whatever Docker already tracks; fllame keeps none of
its own. `recipe add`/`edit`/`remove` never write `compose.yaml` on
their own - only `serve`/`status`/`stop` (and `recipe build HANDLE`,
which exists precisely to regenerate one recipe's `compose.yaml`
standalone, without starting anything) actually write it. `recipe
remove` deletes the whole handle folder - both `recipe.yaml` and
`compose.yaml` together (not a container still running under it, see
`RecipeStore.remove`). Before
`serve` ever calls `docker compose up`, it calls
`fllame/models/cache.py`'s `is_model_cached(repo_id)` - a pure
filesystem scan (`huggingface_hub.scan_cache_dir`), never
`fllame/models/puller.py`'s `pull_model` - to confirm the model is
already fully present in HF's own cache, and fails with a clear error
telling the operator to run `fllame model pull`/`recipe add --pull`
first if it isn't. `serve` (`cli.py`'s `_require_model_cached`) never
calls `pull_model` at all, so there is no code path by which it could
touch the network even by accident - see "Setup vs. running" above.
vLLM's own auto-download inside the container is never relied on
either, and that same host cache directory is bind-mounted into the
container. The bind
mount's host side (`VllmServingBackend`'s `_host_volume_source`) is
written as `${HOME}/...` rather than a literal absolute path whenever
that cache directory sits under the current user's home (the default
`HF_HOME`/`HF_HUB_CACHE` location), falling back to a literal path only
when it doesn't (a custom cache location outside the home directory
entirely, which has no portable `${HOME}`-relative form) - Docker
Compose interpolates `${HOME}` itself at `docker compose` invocation
time, so the generated `compose.yaml` stays correct after being copied
to a different machine or run under a different account, rather than
baking in the one home directory it happened to be generated under.
`fllame/models/cache.py` (`huggingface_hub.scan_cache_dir`) is the
read-only counterpart, backing `fllame model list`.

Every generated service also gets `HF_HUB_OFFLINE=1` unconditionally
(`VllmServingBackend.build_service`, not an invocation-time flag) - the
model is always already fully downloaded by the time the container
runs, so vLLM has no legitimate need to reach the Hub itself, and
letting it try is what fails silently and hangs rather than erroring.
`Recipe.from_dict` rejects a recipe that tries to set `HF_HUB_OFFLINE`
in `env` itself, the same way it already rejects `HF_HOME` - if a
specific model genuinely needs the network for something beyond its
own repo_id (e.g. a linked tokenizer/base-model repo), that's a
hand-edit-the-generated-compose-file situation (see the README's
"Advanced" section). `gpus: "all"` (the Compose Specification's own
shorthand for `docker run --gpus all`, only set when `recipe.gpus ==
"all"`, the default) and `ipc: "host"` (unconditional, vLLM's own
multiprocessing workers routinely need more shared memory than
Docker's tiny default `/dev/shm`) are the same kind of hard default for
the same reason - a sane baseline for a vLLM container specifically,
not a recipe-level knob - and the same hand-edit-the-compose-file
escape hatch applies to both (pinning specific device IDs instead of
`gpus: all`; swapping `ipc: host` for an explicit `shm_size:`).
`serve`'s `is_model_cached` check (see above) is what makes "pull while
online, `serve` later with no network at all" a real guarantee rather
than a hope: a pure filesystem scan means a fast, deterministic failure
if the model isn't fully cached, with no code path that could fall back
to a network call the way even `pull_model(offline=True)` - correct,
but still nominally the download function - would invite by association.
`HF_HUB_OFFLINE=1` on the container guarantees the other half, that
vLLM itself never tries either. There is no flag for any of this - it's
`serve`'s only mode, unconditionally; `model pull`/`recipe add --pull`
are the only things that ever call `pull_model`, since downloading a
model is deliberately their job alone, not `serve`'s.

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

`fllame serve`'s own pre-flight sanity check uses a sibling estimate,
not this one: `models/cache.py`'s `local_estimate_vram_gb(repo_id)`
sums the real on-disk size of a repo's cached `.safetensors` files (a
pure filesystem scan via `scan_cache_dir`, the same data source
`list_cached_models`/`is_model_cached` already use) rather than a Hub
lookup - `serve` only ever reaches this check once `is_model_cached`
has already confirmed the model is present, so there's no need to ask
the Hub about a model that's already sitting on disk. `cli.py`'s
`_warn_if_vram_likely_insufficient` compares that figure against
`hardware scan`'s usable-memory budget right after that confirmation -
a warn-only sanity check, not the blocking guard described under
"Explicitly deferred" below, and silently skipped whenever either side
of that comparison isn't known.

`fllame model update` is the answer to a question `serve` can
structurally never ask itself: is a cached model stale on the Hub?
`models/updater.py`'s `check_for_update(repo_id)` combines
`models/cache.py`'s `cached_revision_hash` (the local half - the
cached revision's commit hash, another pure filesystem read) with one
`huggingface_hub.model_info(repo_id).sha` call (the network half) and
compares the two. Takes a repo_id directly (never a recipe handle -
see "Layering"): with none given it checks every repo_id
`list_cached_models` finds; given one, it checks only that repo_id, no
recipe lookup involved at all. Unlike the VRAM sanity check, a Hub
failure here isn't swallowed into "unknown, skip" - `model update` is
a direct-purpose command the operator explicitly ran, so `cli.py` lets
`HfHubHTTPError`/`RequestException` surface as the same friendly
"Hugging Face Hub unreachable" failure `model scan` gives, rather than
silently reporting nothing wrong. Check-only by default (reports
up-to-date/stale per repo, downloads nothing); `--apply` re-pulls
anything stale via `pull_model` - the same function `model pull` uses,
so a model whose commit hash hasn't actually changed costs no network
transfer even under `--apply`, `snapshot_download` itself already only
fetches files that changed. A model reported "not cached" (asked about
directly by repo_id, but `cached_revision_hash` returns `None`) is left
alone either way - `model update` refreshes what's already there, it
doesn't do a first-time pull; that's `model pull`'s job.

Every one of `pull_model`'s three call sites (`model pull`, `model
update --apply`, `recipe add --pull`) wraps the call in a
`try`/`except (HfHubHTTPError, RequestException)` and reports through
`cli.py`'s `_friendly_download_error` rather than letting a transient
Hub failure mid-download (rate limiting, a dropped connection) surface
as a raw traceback - `LocalEntryNotFoundError`, the exception
`snapshot_download` itself raises after exhausting its own retries, is
already a subclass of `HfHubHTTPError` by way of `EntryNotFoundError`,
so no separate import/catch is needed for it. The message always
mentions that already-downloaded files stay in the cache, so re-running
the identical command resumes rather than restarting the download from
scratch (`snapshot_download` only ever fetches what's missing or
changed) - the one thing worth telling the operator that the raw
exception text doesn't say. `model pull`/`recipe add --pull` exit
non-zero on this; `model update --apply` instead reports that one
repo's row as "download interrupted - rerun to resume" and continues
checking the rest, so one bad re-pull doesn't blank out the whole
table's results. The same three call sites also catch `PermissionError`
separately, via `cli.py`'s `_friendly_permission_error` - the HF cache
is a directory shared with anything else that ever writes into it (most
plausibly a `serve` container that ran without `HF_HUB_OFFLINE` in
effect, e.g. via the hand-edited-compose-file escape hatch, while the
vLLM image ran as root - fllame itself never elevates privileges
anywhere in its own code), and a permission mismatch left behind by
that is fixed by reclaiming ownership, not by retrying - the message
names `config.hf_cache_dir()` and the exact `sudo chown -R $(id -u):
$(id -g) <cache dir>` command rather than just relaying the bare
`OSError` text.

Before `serve` regenerates a recipe's `compose.yaml` (see the
architecture paragraph below - it always does, on every invocation),
`cli.py`'s `_warn_if_cache_location_changed` compares the HF cache
directory it's about to mount against whatever host path is already
baked into that recipe's *existing* `compose.yaml`, if one exists
(`fllame/backends/vllm.py`'s `cache_volume_host_path`, reading the
volume list already written there rather than needing to know its
shape from scratch). A silent regeneration here would repoint the
container at a different cache without saying so, which may not have
this model in it even though the old location still does - so a
mismatch warns and asks to confirm (`-y`/`--yes` skips the prompt,
warning still printed), the same shape as the VRAM check above. Skipped
entirely on a recipe's first `serve` (nothing to compare against yet)
or when the existing file can't be parsed - never blocking the routine
case.

`fllame config set-default-image` only changes what an *unpinned*
recipe (`image: null`) resolves to on its next `serve`/`recipe build` -
that already happens for free, since both regenerate `compose.yaml`
from the recipe and the current config on every invocation (see
"Layering" above). What it doesn't do on its own is touch any
`compose.yaml` already sitting on disk from before the change. Since
`compose.yaml` is meant to be hand-editable (see the architecture
paragraph below and the README's "Advanced" section - gpu pinning,
`shm_size:`, a removed `HF_HUB_OFFLINE`), a blanket regeneration to
pick up the new image would also silently discard any of those edits -
so `config set-default-image` instead offers a narrower fix: `cli.py`'s
`_compose_files_using_image` finds every existing `compose.yaml` whose
service `image:` is a *literal, exact match* for the previous default
(never a recipe that pins its own image, and never a `compose.yaml`
already hand-edited to something else - both fail the exact-match
check on their own), lists them, and asks to confirm before
`_replace_image_in_compose_file` does a plain text substitution of just
that `image:` line's value - not a YAML round-trip, so nothing else in
the file (formatting, comments, an unrelated hand edit) is touched.

## Explicitly deferred (implemented as an interface/hook, not a concrete answer)

- **Non-vLLM backends** (llama.cpp, MLX, ...) - `ServingBackend` exists
  as a seam precisely so one could be added later, but fllame is
  vLLM-only by design today. Don't add a second implementation without a
  decision from the project owner about which framework and why.
- **Sibling services alongside a recipe's generated compose.yaml**
  (Grafana, OpenWebUI, ...) - each recipe's folder
  (`fllame/config.py`'s `recipe_dir`) holds only that one
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
- **A real, blocking pre-flight OOM guard on `fllame serve`** - still
  blocked on the recipe-level estimator above: a wrong "fits" verdict
  from a weights-only figure would be worse than no verdict at all, so
  `serve` doesn't (and shouldn't yet) refuse to launch a recipe based on
  one. What does exist, and is a deliberately narrower thing: a
  warn-only sanity check (`cli.py`'s `_warn_if_vram_likely_insufficient`,
  run right after `_require_model_cached` confirms the model is
  present) that compares `models/cache.py`'s `local_estimate_vram_gb`
  (the real on-disk size of the recipe's cached `.safetensors` files, a
  pure filesystem scan - no Hub lookup, since the model is already sitting
  in the local cache by the time this runs) against `hardware scan`'s
  usable-memory budget, prints a warning and asks to confirm
  (`-y`/`--yes` to skip the prompt) only when it has a confident number
  on both sides - silently skipped otherwise (no hardware signal, or no
  cached `.safetensors` files to measure), never treating "unknown" as
  "must be fine" or as "must not fit." This is a heads-up, not the
  verdict described above, and shouldn't be mistaken for it.
- **Injecting `--gpu-memory-utilization` at serve time from the
  hardware scan.** Decided, not yet built: this is deliberately *not* a
  `Recipe`/`serve_args` concern - a recipe (hand-written or from `recipe
  add`) is never required to set it, and nothing normalizes or defaults
  it into the recipe file. The intended design is for `fllame serve` to
  compute a safe value from `scan_hardware()` at launch time and add it
  to the generated compose service's command, the same invocation-time
  pattern `_warn_if_vram_likely_insufficient` already uses in `cli.py` -
  not persisted, recomputed per machine. Until this exists, a recipe
  that omits
  `--gpu-memory-utilization` gets whatever vLLM's own default
  is; one that sets it explicitly (e.g. from a paste that already had
  it) is used as-is.
- **Any multi-user or remote-access concern** (auth, RBAC, a server
  process) - fllame is a local CLI for a trusted single operator by
  design, not a service. If that assumption ever needs to change, that's
  a new decision, not an extension of the current code.

## Merged so far

- CLI scaffold: `recipe list`/`show`/`add`/`build`/`edit`/`remove`,
  `hardware scan`, `model pull`/`list`/`scan`, `serve` (foreground,
  `--detach`, `--yes`), `status`, `stop` - `-h` works as a `--help`
  alias at every level (set via `context_settings` on each `Typer()`
  instance; Click only binds `--help` by default).
- `recipe add`: recipe creation from a `vllm serve` line, either as
  trailing CLI arguments (a quick one-liner, `recipes/parser.py` +
  `recipes/naming.py` + `RecipeStore.save`/`next_available_handle`) or,
  with no trailing arguments, a guided dialogue (image, then preinstall
  commands, then env vars, then the `vllm serve` command - each its own
  labeled step, not one undifferentiated stdin paste). `--pull`/
  `--build` optionally download the model / regenerate the compose
  folder right after saving (`cli.py`'s `_build_or_exit`, shared with
  `recipe build HANDLE` below). `recipe edit`
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
- `recipe build HANDLE` regenerates just that recipe's `compose.yaml`,
  standalone - useful since `recipe add`/`edit`/etc. never write it
  themselves (only `serve`/`status`/`stop` do). Fails with the same
  cache-miss error `serve` itself gives if the model isn't fully
  downloaded yet, rather than writing a compose file that can't
  actually run.
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
  configured default, `gpus: all|none`, `env` rejects `HF_HOME` and
  `HF_HUB_OFFLINE`, `preinstall` for a shell setup step before `vllm
  serve`) + YAML-directory-backed `RecipeStore`, one `recipe.yaml` per
  handle's own folder.
- `VllmServingBackend`, the sole `ServingBackend` implementation -
  compiles a `Recipe` into a docker-compose service definition, always
  a plain `compose.yaml` with no separate build step, `HF_HUB_OFFLINE=1`/
  `gpus: "all"`/`ipc: "host"` all baked in as hard defaults (see the
  architecture paragraph above); when `preinstall` is set, wraps the
  container's own command in a shell (`sh -c "<preinstall> && exec vllm
  serve ..."`) instead of building a custom image.
- `fllame/compose/generator.py` - compiles one recipe into its own
  self-contained `compose.yaml`, written into that same recipe's own
  folder next to its `recipe.yaml`; `serve`/`status`/`stop` drive
  it via `docker compose up|ps|stop` instead of fllame tracking its own
  state.
- `fllame/models/puller.py` (download, via `pull_model`) + `models/
  cache.py` (list/verify, via pure filesystem scans - `list_cached_models`,
  `is_model_cached`, `local_estimate_vram_gb`) - kept as two separate
  modules on purpose, since `serve` is only ever allowed to import from
  the latter (see "Setup vs. running" above). This split is what makes
  "pull while online, `serve` later with no network at all" a real
  guarantee.
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
- Unified recipe/compose layout: `recipe.yaml` and its generated
  `compose.yaml` now live side by side in one folder per handle
  (`fllame/config.py`'s `recipe_dir`), replacing the earlier split
  between `$FLLAME_RECIPES_DIR` (flat `<handle>.yaml` files) and a
  separate `$FLLAME_STATE_DIR` tree - `FLLAME_STATE_DIR` no longer
  exists. `docker-compose.yml` renamed to `compose.yaml` throughout.
  `VllmServingBackend` also gained two more unconditional hard
  defaults alongside `HF_HUB_OFFLINE=1`: `gpus: "all"` (replacing the
  old `deploy.resources.reservations.devices` block with the Compose
  Specification's simpler shorthand) and `ipc: "host"` (vLLM's own
  multiprocessing workers need more shared memory than Docker's tiny
  default `/dev/shm`). `environment` renders as a `KEY=VALUE` list
  rather than a `KEY: VALUE` mapping (both are equivalent Compose
  syntax; the list form is the more familiar shell-like shape).
- The HF cache bind mount's host side is written as `${HOME}/...`
  rather than a literal absolute path whenever it sits under the
  current user's home directory (`VllmServingBackend`'s
  `_host_volume_source`), falling back to a literal path only when it
  doesn't - Docker Compose interpolates `${HOME}` itself, so a
  generated `compose.yaml` stays correct after being copied to a
  different machine or run under a different account, rather than
  baking in the one home directory it was generated under.
- `fllame serve`'s pre-flight VRAM sanity check (`cli.py`'s
  `_warn_if_vram_likely_insufficient`, run right after
  `_require_model_cached` confirms the model is present) -
  `models/cache.py`'s new `local_estimate_vram_gb(repo_id)` (the real
  on-disk size of the recipe's cached `.safetensors` files, a pure
  filesystem scan, no network) compared against `hardware scan`'s
  usable-memory budget. Warns and asks to confirm (`serve -y`/`--yes`
  to skip the prompt, warning still printed) only when both figures are
  actually known and the estimate exceeds the budget; silently skipped
  otherwise (no hardware signal, or no cached `.safetensors` files to
  measure) - reads the local cache only, consistent with `serve` never
  touching the network at all. Deliberately not the blocking,
  recipe-level guard tracked as still-deferred above - a coarse,
  weights-only heads-up, not a verdict.
- `fllame serve --offline` removed, and `serve` no longer calls
  `pull_model` at all (not even with `offline=True`) - it calls
  `models/cache.py`'s `is_model_cached(repo_id)` instead, via `cli.py`'s
  shared `_require_model_cached` (also used by `recipe build`/`recipe
  add --build`). Both are pure filesystem scans, so there is no code
  path inside `serve` that could touch the network even accidentally -
  a stronger guarantee than parameterizing the download function not to
  download. Fails with a clear error pointing at `fllame model pull
  <repo_id>` if the model isn't already fully cached, rather than
  downloading it - downloading is exclusively `fllame model pull`'s job
  (or `recipe add HANDLE --pull` right when a recipe is created). See
  "Setup vs. running" above.
- `fllame model update [REPO_ID] [--apply]` - `models/updater.py`'s
  `check_for_update` compares a cached repo's local commit hash
  (`models/cache.py`'s new `cached_revision_hash`, a pure filesystem
  read) against the Hub's current one (`huggingface_hub.model_info`).
  Check-only by default (a per-repo `up to date`/`stale`/`not cached`
  table via `cli.py`'s `_print_table`); `--apply` re-pulls anything
  stale through the same `pull_model` `model pull` already uses, so an
  unchanged model costs no transfer even under `--apply`. Takes a
  repo_id directly, never a recipe handle, same as `model pull` -
  neither command touches `RecipeStore` or anything under
  `fllame.domain.recipe`/`fllame.recipes` at all (see "Layering"). With no
  HANDLE, checks every repo `list_cached_models` finds; a Hub failure
  is a hard, friendly error (`model scan`'s own pattern), not a
  swallowed "unknown" the way the VRAM sanity check treats one - this
  is a direct-purpose command the operator explicitly ran. This is the
  only place fllame ever asks the Hub whether a cached model has gone
  stale, since `serve` structurally cannot (see "Setup vs. running").
- Friendly handling of a transient Hub failure mid-download - all three
  `pull_model` call sites (`model pull`, `model update --apply`,
  `recipe add --pull`) now catch `HfHubHTTPError`/`RequestException`
  and report through `cli.py`'s new `_friendly_download_error` instead
  of letting a raw traceback (e.g. a 429 rate-limit escalating to
  `LocalEntryNotFoundError`) reach the operator - see the architecture
  paragraph above for why no separate `LocalEntryNotFoundError` catch
  is needed.
- The same three `pull_model` call sites also catch `PermissionError`
  separately, via `cli.py`'s new `_friendly_permission_error` - a
  shared-cache ownership mismatch (see the architecture paragraph
  above) gets a message naming the cache directory and the exact
  `chown` command to fix it, instead of a bare `OSError` traceback.
- `fllame serve`'s new `_warn_if_cache_location_changed` (`cli.py`) -
  warns and asks to confirm, before regenerating a recipe's
  `compose.yaml`, if the HF cache directory it would now mount differs
  from what's already baked into that recipe's existing `compose.yaml`
  (`fllame/backends/vllm.py`'s new `cache_volume_host_path` reads the
  existing file's volume entry for the comparison). `-y`/`--yes` skips
  the confirmation, same as the VRAM check; silently skipped on a
  recipe's first `serve` or an unparseable existing file.
- `fllame config set-default-image` now offers to update every existing
  `compose.yaml` whose image is a literal, exact match for the previous
  default (`cli.py`'s `_compose_files_using_image`) - a plain text
  substitution of just the `image:` value
  (`_replace_image_in_compose_file`), not a regeneration, so any other
  hand edits already in those files survive untouched. Skipped
  entirely when there was no previous default to search for, or the new
  image is unchanged from it.
