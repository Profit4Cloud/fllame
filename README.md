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

Recipes live in `~/.config/fllame/recipes/<handle>/recipe.yaml` by
default (override the base directory with `FLLAME_RECIPES_DIR`). One
folder per model handle - the same folder later holds that handle's
generated `compose.yaml` too (see "Where compose.yaml lives" below) -
see `examples/recipes/llama-3-8b-instruct.yaml` for a full recipe
example:

```yaml
# ~/.config/fllame/recipes/llama-3-8b-instruct/recipe.yaml
image: vllm/vllm-openai:v0.27.1
description: Llama 3 8B Instruct, single-GPU default profile
command: |-
  vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --gpu-memory-utilization=0.9 \
  --max-model-len=8192
```

`command` is the whole `vllm serve <repo_id> <args...>` invocation -
not split into separate `repo_id`/`serve_args`/`port` keys - rendered
one flag per line, each ending in `\` (a plain single line when there
are no flags at all). This is exactly what `recipe show` prints and
exactly what you'd copy out to run by hand, on a box with vLLM
installed, with no reassembly - the same canonical rendering every
time regardless of how the command was originally pasted or typed.

Then:

```
fllame hardware scan                           # what this box can run
fllame model scan                              # models that plausibly fit, ranked
fllame model scan -q llama --min-params 7 --max-params 13   # narrower search
fllame model scan --quant gptq --max-size 40         # this box's quant list, a smaller budget
fllame config set-default-image vllm/vllm-openai:v0.27.1   # skip --image below from now on
fllame recipe add                              # guided dialogue: image, preinstall, env, command
fllame recipe add --image vllm/vllm-openai:v0.27.1 \
  vllm serve org/repo --tensor-parallel-size 1 --enable-auto-tool-choice   # or a quick one-liner
fllame recipe list
fllame recipe show llama-3-8b-instruct         # resolved recipe (image/env/preinstall/command), as YAML
fllame recipe edit llama-3-8b-instruct         # open the YAML file in $EDITOR, re-validated on save
fllame recipe remove llama-3-8b-instruct       # delete it (asks first, unless -y)
fllame model pull meta-llama/Meta-Llama-3-8B-Instruct  # download into the HF cache, standalone
fllame recipe build llama-3-8b-instruct        # write its compose.yaml - fails if not pulled yet
fllame recipe add --pull --build ...           # or do both right after creating the recipe
fllame model list                              # what's actually cached locally
fllame model update                            # check every cached model against the Hub for updates
fllame model update meta-llama/Meta-Llama-3-8B-Instruct --apply   # or just one, re-pulling it if stale
fllame serve llama-3-8b-instruct               # never touches the network - fails if not pulled yet
fllame serve llama-3-8b-instruct --detach      # same, but backgrounded
fllame serve llama-3-8b-instruct --yes         # skip the VRAM sanity check's confirmation prompt
fllame status                                  # docker compose ps
fllame stop llama-3-8b-instruct                # docker compose stop
```

`fllame model scan` searches the HF Hub, ranked by estimated-VRAM fit,
downloads, and recency (top 20 by default, `--limit` to change it).
`--max-size` (in GB, the same figure shown in the EST. VRAM column) is
the primary size gate and is always enforced: give it explicitly, or it
defaults to a coarse VRAM/RAM-based budget from this machine's
`hardware scan` - a starting point for "what can I run," not a
benchmarked guarantee (see CLAUDE.md for the difference between this
estimate and the stronger, still-unbuilt per-recipe one). `--quant`
independently overrides the hardware scan's detected quantization list -
useful for looking at a format your current box doesn't support, e.g.
because you're scanning from a laptop for a model you'll actually serve
elsewhere. `--min-params`/`--max-params` are a
separate, optional restriction on declared parameter count with no
hardware-derived default of their own - give neither and only
`--max-size` applies; give `--max-params` and ranking weighs closeness
to both equally alongside popularity/recency. `-q`/`--query` (free text)
narrows further regardless of any of the above. Alongside PARAMS and EST.
VRAM, results show DOWNLOADS (recent) and UPDATED (relative time) - both
of which also feed the ranking, same as size fit does; QUANT is only
shown when more than one quantization is being searched (see
`fllame model scan -h` for the full column breakdown).

`fllame recipe add`, given trailing arguments, treats them as a quick
one-liner: the whole `vllm serve <repo_id> <args...>` line, handy for
something you already have on your clipboard as a single command
(shell quoting/escaping applies as normal, e.g. wrap a value with
spaces in quotes) - the Docker image resolves the same way either way
(an explicit `--image` wins; otherwise the configured default from
`fllame config set-default-image`; only prompts when neither exists),
warning if the resolved tag looks unpinned (e.g. `:latest`). Env vars
and preinstall commands aren't spellable as trailing arguments at all.

With no trailing arguments, `recipe add` instead walks through a short
dialogue - the shape a recipe typically comes in from a model card or
vLLM's own docs, broken into labeled steps instead of one undifferentiated
paste:

1. **Docker image** - prompted with the configured default (or
   `vllm/vllm-openai:latest` if none is set) prefilled; press Enter to
   keep following that default (so a later `fllame config
   set-default-image` change still applies to this recipe), or type a
   different image to pin this recipe to it specifically.
2. **Preinstall commands** - paste one or more shell commands run
   before `vllm serve` (e.g. `pip install -U transformers`), one per
   line; a blank line or Ctrl-D moves on (immediately, to skip this
   step entirely).
3. **Environment variables** - paste one or more `KEY=VALUE` lines,
   same blank-line-or-Ctrl-D convention.
4. **The `vllm serve` command** - required; paste it verbatim, one flag
   per line works fine either with or without a trailing `\` line
   continuation - plenty of real examples show one flag per line with
   no continuation marker at all, relying on the code block's own line
   breaks rather than real shell syntax, and fllame doesn't require one
   either.

Preinstall commands are taken verbatim as shell text (`&&` and all,
same as a real shell command) since that's what they genuinely are;
env var values and the `vllm serve` command are parsed and sanitized,
not evaluated as shell - a shell metacharacter/substitution (`;`, `&`,
`|`, `` ` ``, `$(...)`) in either is a hard error and nothing gets
written. The handle is derived from the repo id (the part
after the last `/`, lowercased and slugified, e.g.
`meta-llama/Meta-Llama-3-8B-Instruct` -> `meta-llama-3-8b-instruct`); a
second recipe for a repo that already has one gets `_2`, `_3`, etc.
rather than overwriting - useful for keeping
more than one tuning of the same model around. `recipe edit` opens the
YAML file directly in `$EDITOR` and re-validates on save. `command`'s
formatting is read leniently - its own multi-line rendering never goes
through YAML's own (indentation-sensitive) parser at all, so deleting
what looks like meaningless leading whitespace, or a missing trailing
`\` on a continuation line (plenty of real examples don't use one),
doesn't break it. A stray tab or CRLF line ending elsewhere - the most
common way a text editor breaks YAML - is fixed automatically too;
anything still invalid (malformed YAML entirely, a missing `command`,
...) offers a choice: reopen `$EDITOR` to fix it, or revert to the
version from before this edit (kept in memory for the length of the
command - the recipes directory is meant to be git-tracked already,
which is the real backup, not a `.bak` file on disk). Either way, once
the file validates it's re-saved in fllame's own canonical rendering,
regardless of how it was actually formatted. `recipe remove` deletes a
recipe, asking first unless `-y`/`--yes`.

`fllame recipe add` only ever writes `recipe.yaml` - creating one
doesn't also write that handle's `compose.yaml` (see "Where
compose.yaml lives" below), so there's nothing to find there until you
actually run `serve`/`status`/`stop`, or one of these: `fllame recipe
build HANDLE` regenerates just that recipe's `compose.yaml` standalone,
without starting anything - handy for inspecting or copying the file
elsewhere. It fails with a clear error if the model isn't fully
downloaded yet (run `fllame model pull` first) rather than silently
writing a compose file that can't actually be run. `recipe add --pull`
downloads the model right after saving (a no-op if it's already
cached); `recipe add --build` regenerates `compose.yaml` right after
saving, with the same not-yet-cached failure as `recipe build` unless
combined with `--pull`, in which case the pull happens first so the
build always succeeds.

`fllame config` holds fllame's own persisted settings - today just
`default_image`, the Docker image a recipe falls back to when it doesn't
pin its own (`fllame config set-default-image ...` / `fllame config
show`), stored in `$FLLAME_CONFIG_FILE` (default
`~/.config/fllame/config.yaml`). Changing it applies to every recipe
that doesn't set its own `image` - nothing needs re-adding.

`fllame hardware scan` detects NVIDIA GPU(s) via `nvidia-smi` (name, count,
VRAM per GPU) and RAM via `/proc/meminfo`, and reports which vLLM
quantizations that hardware supports. It's a live scan, not a persisted
value - nothing to keep in sync.

`fllame serve` never touches the network itself, under any
circumstance - setting up a recipe (adding it, pulling its model) and
running it are strictly separate phases, and only the former is allowed
to reach the network. `serve` checks that the model is already fully
present in Hugging Face's own cache with a pure filesystem scan (no
network call of any kind, not even a "local files only" one) and fails
immediately with a clear error - telling you to run
`fllame model pull <repo_id>` first - if it isn't, rather than falling
back to a download of its own. Downloading is exclusively `model
pull`'s job (or `recipe add HANDLE --pull` right when the recipe is
created) -
`serve` only ever confirms, never fetches. That same cache directory
(wherever `HF_HOME`/`HF_HUB_CACHE` resolves to) is bind-mounted into
the container, so vLLM's own auto-download inside the container is
never relied on either. When that directory sits under the current
user's home (the default, out-of-the-box location), the bind mount's
host side is written as `${HOME}/...` rather than a literal absolute
path, so `compose.yaml` stays correct when copied to a different
machine or run under a different account - Docker Compose interpolates
`${HOME}` itself from whatever shell environment `docker compose` runs
in.

Once the model is confirmed cached, `fllame serve` compares a coarse,
weights-only VRAM estimate - the real on-disk size of that model's
cached `.safetensors` files, no network involved - against this same
hardware scan's usable-memory budget, and warns - asking to confirm,
unless `-y`/`--yes` - if it looks like it won't fit. This is a
heads-up, not a benchmarked guarantee either way (no KV cache/
activations/concurrency in the estimate - see CLAUDE.md for the
stronger, still-unbuilt recipe-level estimator this isn't), and it's
silently skipped whenever a confident comparison isn't possible: no
GPU/RAM figure from the hardware scan, or no cached `.safetensors`
files to measure.

Every generated compose file also sets `HF_HUB_OFFLINE=1` on the
container unconditionally, so vLLM itself never attempts a network call
either - the model is always already fully downloaded by the time it
starts. Together, this is what makes "pull while online, `serve` later
with no network at all" a real guarantee rather than a hope: run
`fllame model pull <repo_id>` (or `recipe add --pull`) while online,
then `fllame serve <handle>` later on a genuinely offline machine.

Because `serve` never touches the network, it also has no way to
notice a cached model has been updated upstream - `fllame model
update` is the other side of that trade-off. Like `model pull`, it
takes a repo_id directly, never a recipe handle - neither command has
any notion that recipes exist at all, since a recipe is a higher-level
concept built on top of a model, not the other way around. With no
argument it checks every model currently in the cache against the Hub;
given a repo_id, just that one. Check-only by default (reports up to
date/stale, downloads nothing); `--apply` re-downloads anything stale
through the same path `model pull` uses, so a model whose commit hash
hasn't actually changed costs no transfer even then.

A large download can hit a routine, transient Hub error partway through
(rate limiting, a dropped connection) - `model pull`, `model update
--apply`, and `recipe add --pull` all catch this and report a friendly
message rather than a raw traceback, noting that already-downloaded
files stay cached, so re-running the same command resumes rather than
starting over.

### Where compose.yaml lives

Each recipe compiles to its own self-contained `compose.yaml`, written
right next to that recipe's `recipe.yaml` -
`~/.config/fllame/recipes/<handle>/` (override the base directory with
`FLLAME_RECIPES_DIR`) holds both `recipe.yaml` (hand-edited) and
`compose.yaml` (generated), one folder per handle. Nothing else to
build or manage alongside it, even for a recipe with a `preinstall`
step (see below) - each is its own compose project (`fllame-<handle>`),
entirely independent of every other recipe's. `fllame serve`/`status`/
`stop` regenerate HANDLE's `compose.yaml` before running a `docker
compose` command against it (`status` loops over every recipe, one
`docker compose ps` each, under a `== <handle> ==` header).
`compose.yaml` is a generated artifact fllame fully owns and overwrites
on every one of those commands - normal changes belong in the recipe,
not the compose file, since they'd otherwise be silently discarded on
the next regeneration. See "Advanced" below for the cases where
hand-editing it is the right move. Being self-contained, the folder can
also be copied elsewhere and driven with plain `docker compose up -d`,
no fllame involved. `recipe remove` deletes a recipe's whole folder -
`recipe.yaml` and `compose.yaml` together (not a container already
running under it - `fllame stop` that first if it matters).

A recipe's `preinstall` step runs as part of the container's own
startup command, every time it starts, through a shell (`sh -c
"<preinstall> && exec vllm serve ..."`) - there's no separate image
build step, on purpose: one recipe stays one `compose.yaml`, with
nothing else to manage or go stale alongside it. The trade-off is that
the preinstall command(s) run again on every container start (a
restart included), not just the first one - if that install is slow,
that's the cost of keeping this compose-only.

## Recipe format

| Field         | Required | Meaning |
|---------------|----------|---------|
| `command`     | yes      | the whole `vllm serve <repo_id> <args...>` invocation - e.g. `vllm serve org/repo --max-model-len 8192`. Not split into separate keys, and rendered one flag per line (each ending in `\`) whenever there's more than one, so it's exactly what you'd copy out to run by hand. `repo_id` (downloaded via `fllame model pull`) and the host port mapping (`--port`, defaulting to vLLM's own `8000` if the command doesn't set one) are both derived from it, not separate fields |
| `image`       | no       | the Docker image to run, e.g. `vllm/vllm-openai:v0.27.1` - omit to use fllame's configured default (`fllame config`), falling back to `vllm/vllm-openai:latest` if none is configured |
| `backend`     | no       | must be `vllm` if set - the only backend fllame ships today |
| `description` | no       | free text, shown by `recipe show` |
| `gpus`        | no       | `all` (default) or `none` - whether the container gets a GPU reservation |
| `env`         | no       | environment variables set on the container; must not set `HF_HOME` or `HF_HUB_OFFLINE`, which fllame manages itself |
| `preinstall`  | no       | shell commands run, in order, before `vllm serve` (e.g. `pip install -U transformers`) - a preinstall step some recipes need on top of the base image |

## Advanced

`HF_HUB_OFFLINE=1` is unconditional on every generated `compose.yaml`
because letting vLLM's own download/update path touch the network is
unreliable - it can fail silently and leave you waiting indefinitely
instead of erroring, whereas fllame's own `model pull` step already
guarantees the model is fully cached before the container ever starts.
If a specific model genuinely needs network access (e.g. a linked
tokenizer or base-model repo), the way to work around this is to edit
that recipe's generated `compose.yaml` directly - fllame's job ends at
producing a working compose file, and an engineer is always free to
take it from there.

`gpus: all` is likewise a hard default on the generated service
whenever the recipe's own `gpus` is `all` (the default) - if you need
something more specific, like pinning particular device IDs instead of
reserving every GPU on the host, edit `compose.yaml`'s `gpus:` key
directly rather than looking for a finer-grained recipe field.

`ipc: host` is also always baked in, because vLLM's own multiprocessing
workers (tensor-parallel, NCCL) routinely need more shared memory than
Docker's tiny default `/dev/shm`, and running out shows up as an opaque
crash rather than a clear error. If `ipc: host` is undesirable for
isolation reasons on your host, edit `compose.yaml` to swap it for an
explicit `shm_size:` instead.

## Development

```
poetry install
poetry run pytest
poetry run ruff check .
```

See `CLAUDE.md` for the architecture this sits on and what's deliberately
not built yet.
