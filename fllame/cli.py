"""fllame - a headless CLI for vLLM serving.

`fllame serve <handle>` resolves a hand-edited recipe, verifies the
model is already fully downloaded via a pure filesystem check - never
touching the network itself; `model pull`/`recipe add --pull` are the
only things that ever download - and runs it as a Docker container via
`docker compose`. See README.md for the recipe file format and
CLAUDE.md for the architecture this sits on, including the setup/
running boundary this split enforces.
"""

from __future__ import annotations

import dataclasses
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
import typer
import yaml
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import RequestException

from fllame import config, config_file
from fllame.backends.vllm import VllmServingBackend, cache_volume_host_path
from fllame.compose.generator import generate_compose, write_compose_file
from fllame.domain.recipe import Recipe, RecipeError
from fllame.domain.vllm_command import (
    VllmCommandError,
    join_command_lines,
    parse_vllm_serve_command,
)
from fllame.hardware.scanner import scan_hardware
from fllame.models.cache import is_model_cached, list_cached_models, local_estimate_vram_gb
from fllame.models.discovery import search_models
from fllame.models.puller import pull_model
from fllame.models.sizing import memory_budget_gb, usable_memory_gb
from fllame.models.updater import check_for_update
from fllame.recipes.naming import derive_handle
from fllame.recipes.parser import RecipePasteError, parse_env_line, parse_pasted_recipe
from fllame.recipes.store import RecipeStore, autofix_whitespace

# `--help` is Click's default; `-h` is the standard Unix short form on
# top of it - wired in explicitly since Click doesn't bind it by default.
_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(no_args_is_help=True, add_completion=False, context_settings=_CONTEXT_SETTINGS)
recipe_app = typer.Typer(no_args_is_help=True, context_settings=_CONTEXT_SETTINGS)
app.add_typer(recipe_app, name="recipe", help="Inspect the recipe registry.")
hardware_app = typer.Typer(no_args_is_help=True, context_settings=_CONTEXT_SETTINGS)
app.add_typer(hardware_app, name="hardware", help="Detect this machine's GPU/RAM.")
model_app = typer.Typer(no_args_is_help=True, context_settings=_CONTEXT_SETTINGS)
app.add_typer(model_app, name="model", help="Search, download, and inspect Hugging Face models.")
config_app = typer.Typer(no_args_is_help=True, context_settings=_CONTEXT_SETTINGS)
app.add_typer(config_app, name="config", help="View and change fllame's persisted settings.")

BACKEND = VllmServingBackend()

# The image used when a recipe doesn't pin its own and no default has
# been configured via `fllame config set-default-image` - the last resort,
# not something an operator is expected to rely on long-term.
_FALLBACK_IMAGE = "vllm/vllm-openai:latest"


def _recipe_store() -> RecipeStore:
    return RecipeStore(config.recipes_dir())


def _resolve_image(recipe: Recipe) -> Recipe:
    """A recipe whose `image` is unset means "use fllame's configured
    default" - resolved here, at the point a Recipe is turned into a
    compose service, rather than baked into the recipe file itself.
    """
    if recipe.image is not None:
        return recipe
    return dataclasses.replace(recipe, image=config_file.get_default_image() or _FALLBACK_IMAGE)


def _load_or_exit(handle: str) -> Recipe:
    try:
        return _recipe_store().load(handle)
    except RecipeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e


def _write_recipe_compose(recipe: Recipe) -> None:
    """Regenerates HANDLE's `compose.yaml`, written right next to its
    `recipe.yaml` (`fllame/config.py`'s `recipe_dir`) - nothing else to
    build or manage alongside it (a stale `Dockerfile` from an older
    fllame version's preinstall handling is removed if found). Its
    `HF_HUB_OFFLINE=1` is baked in unconditionally by
    `VllmServingBackend` - not something this invocation controls.
    """
    resolved = _resolve_image(recipe)
    directory = config.recipe_dir(resolved.handle)
    compose = generate_compose(resolved, backend=BACKEND, hf_cache_dir=config.hf_cache_dir())

    directory.mkdir(parents=True, exist_ok=True)
    stale_dockerfile = directory / "Dockerfile"
    if stale_dockerfile.is_file():
        stale_dockerfile.unlink()

    write_compose_file(compose, directory / "compose.yaml")


def _require_model_cached(recipe: Recipe) -> None:
    """Fails with a clear, consistent error if `recipe`'s model isn't
    already fully present in the local HF cache - the shared "verify,
    never fetch" gate for `serve` and `recipe build`/`recipe add
    --build`. Deliberately never calls `models/puller.py`'s
    `pull_model` (which can touch the network): `models/cache.py`'s
    `is_model_cached` is a pure filesystem scan, so this can never be
    the thing that lets one of these commands cross the internet
    boundary. Downloading is exclusively `fllame model pull`/
    `recipe add --pull`'s job - see CLAUDE.md's setup/running boundary.
    """
    if is_model_cached(recipe.repo_id):
        return
    typer.echo(
        f"'{recipe.repo_id}' is not fully cached locally - run "
        f"`fllame model pull {recipe.repo_id}` first.",
        err=True,
    )
    raise typer.Exit(code=1)


def _build_or_exit(recipe: Recipe) -> None:
    """Regenerates `recipe`'s `compose.yaml` - failing with a clear
    error, same wording as `serve`'s cache-miss check, if the model
    isn't fully downloaded yet. Shared by `recipe build` and
    `recipe add --build`.
    """
    _require_model_cached(recipe)
    _write_recipe_compose(recipe)
    compose_path = config.recipe_dir(recipe.handle) / "compose.yaml"
    typer.echo(f"wrote {compose_path}")


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Left-aligned, space-padded columns - `docker ps`/`kubectl get`
    style, no border characters. The point is making a column (size,
    quantization, ...) comparable at a glance down the page; a border
    wouldn't add anything padding doesn't already give it.
    """
    all_rows = [headers, *rows]
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]
    for row in all_rows:
        padded = [cell.ljust(width) for cell, width in zip(row[:-1], widths[:-1], strict=False)]
        typer.echo("  ".join([*padded, row[-1]]))


def _format_count(n: int | None) -> str:
    """A compact form of a download count - "12.3k", "1.2M" - so the
    column stays narrow regardless of magnitude. Not locale-aware; this
    is a terminal table, not user-facing prose."""
    if n is None:
        return "unknown"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


# (label, seconds-per-unit, largest value still shown in this unit
# before rolling up to the next one - `None` for the last, open-ended
# unit). Mirrors huggingface_hub's own `CachedRepoInfo.last_modified_str`
# (used by `fllame model list`) for a consistent "3 days ago" feel
# across both tables, without depending on that library's private
# formatter.
_RELATIVE_TIME_UNITS = (
    ("second", 1, 59),
    ("minute", 60, 59),
    ("hour", 60 * 60, 23),
    ("day", 60 * 60 * 24, 6),
    ("week", 60 * 60 * 24 * 7, 3),
    ("month", 60 * 60 * 24 * 30, 11),
    ("year", 60 * 60 * 24 * 365, None),
)


def _format_relative_time(dt: datetime | None) -> str:
    if dt is None:
        return "unknown"
    delta_seconds = (datetime.now(UTC) - dt).total_seconds()
    if delta_seconds < 20:
        return "a few seconds ago"
    for label, divider, max_value in _RELATIVE_TIME_UNITS:  # noqa: B007 - used after the loop
        value = round(delta_seconds / divider)
        if max_value is not None and value <= max_value:
            break
    return f"{value} {label}{'s' if value != 1 else ''} ago"


def _run_compose(handle: str, *args: str) -> int:
    command = [
        "docker",
        "compose",
        "-f",
        str(config.recipe_dir(handle) / "compose.yaml"),
        "-p",
        config.compose_project_name(handle),
        *args,
    ]
    try:
        return subprocess.run(command).returncode
    except FileNotFoundError as e:
        typer.echo(
            "docker (or the compose plugin) was not found on PATH - fllame runs "
            "vLLM as a Docker container, install Docker to use this command.",
            err=True,
        )
        raise typer.Exit(code=1) from e


@recipe_app.command("list")
def recipe_list() -> None:
    """List model handles with a recipe on file."""
    handles = _recipe_store().list_handles()
    if not handles:
        typer.echo(f"No recipes found in {config.recipes_dir()}")
        raise typer.Exit(code=0)
    for handle in handles:
        typer.echo(handle)


@recipe_app.command("show")
def recipe_show(handle: str) -> None:
    """Print HANDLE's resolved recipe - the same shape as the recipe
    file, with `image` filled in from fllame's configured default when
    the recipe doesn't pin its own. `command` is the last line, ready to
    copy out and run by hand (`vllm serve ...` on a box with vLLM
    installed) without going through Docker at all.
    """
    recipe = _resolve_image(_load_or_exit(handle))
    typer.echo(recipe.to_yaml().rstrip())


def _read_block(prompt_text: str) -> list[str]:
    """Reads lines from stdin until a blank line or EOF (Ctrl-D) -
    works the same whether stdin is an interactive terminal or
    redirected/piped input, unlike `sys.stdin.read()` (which consumes
    to the *first* EOF and leaves nothing for a later step). `#` comment
    lines are skipped; everything else is collected verbatim, in order.
    """
    typer.echo(prompt_text)
    lines: list[str] = []
    while True:
        raw = sys.stdin.readline()
        if raw == "" or raw.strip() == "":
            break
        stripped = raw.rstrip("\n")
        if stripped.strip().startswith("#"):
            continue
        lines.append(stripped)
    return lines


def _read_command() -> str:
    lines = _read_block(
        "vllm serve command (required) - one flag per line works fine, "
        "with or without a trailing \\ - then a blank line or Ctrl-D:"
    )
    if not lines:
        typer.echo("no `vllm serve <repo_id> ...` command given", err=True)
        raise typer.Exit(code=1)
    joined = join_command_lines(lines)
    try:
        parse_vllm_serve_command(joined)
    except VllmCommandError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    return joined


_UV_PIP_INSTALL_PREFIX = "uv pip install"


def _replace_uv_pip_install(preinstall: list[str]) -> tuple[list[str], bool]:
    """`uv pip install ...` is a common preinstall step lifted straight
    from a model card, but the vllm/vllm-openai image's own Python
    environment isn't the uv-managed venv `uv pip install` expects -
    running it there can install into the wrong place rather than where
    `vllm serve` actually looks. Plain `pip install` targets the image's
    own site-packages directly, so it's swapped in automatically; the
    caller is expected to tell the operator when this fires, since it
    changes what they typed.
    """
    changed = False
    fixed = []
    for line in preinstall:
        stripped = line.lstrip()
        if stripped.startswith(_UV_PIP_INSTALL_PREFIX):
            leading_ws = line[: len(line) - len(stripped)]
            fixed.append(f"{leading_ws}pip install{stripped[len(_UV_PIP_INSTALL_PREFIX):]}")
            changed = True
        else:
            fixed.append(line)
    return fixed, changed


@recipe_app.command("add", context_settings={**_CONTEXT_SETTINGS, "ignore_unknown_options": True})
def recipe_add(
    vllm_serve_line: list[str] = typer.Argument(
        None,
        help="Optionally, the whole `vllm serve <repo_id> ...` line as trailing "
        "arguments instead of the guided dialogue - e.g. `fllame recipe add vllm "
        "serve org/repo --max-model-len 8192`. A quick one-liner only - env vars/"
        "preinstall commands need the dialogue (run with no trailing arguments).",
    ),
    image: str | None = typer.Option(
        None,
        "--image",
        show_default=False,
        help="Docker image to pin this recipe to, overriding fllame's configured "
        "default (`fllame config show`) - e.g. vllm/vllm-openai:v0.27.1. Omit to "
        "let this recipe follow the configured default, whatever it is later "
        "changed to.",
    ),
    gpus: str = typer.Option("all", "--gpus", help="GPU reservation: 'all' or 'none'."),
    pull: bool = typer.Option(
        False,
        "--pull",
        help="Also download the model into the HF cache right after saving "
        "(a no-op if it's already cached) - see `fllame model pull`.",
    ),
    build: bool = typer.Option(
        False,
        "--build",
        help="Also regenerate the compose folder right after saving - see "
        "`fllame recipe build`. Fails if the model isn't cached yet unless "
        "combined with --pull.",
    ),
) -> None:
    """Create a recipe.

    Given trailing arguments, treats them as a quick one-liner: the
    whole `vllm serve <repo_id> <args...>` line, same as today, no
    prompts beyond the Docker image if neither `--image` nor a
    configured default exists. With no trailing arguments, walks
    through a short dialogue instead - Docker image, then preinstall
    commands, then env vars, then the `vllm serve` command - the shape
    a recipe typically comes in from a model card or vLLM's own docs.
    """
    if vllm_serve_line:
        pasted = shlex.join(vllm_serve_line)
        try:
            parsed = parse_pasted_recipe(pasted)
        except RecipePasteError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1) from e
        command, env, preinstall = parsed.command, parsed.env, parsed.preinstall

        if image is None and config_file.get_default_image() is None:
            image = typer.prompt(
                "Docker image (e.g. vllm/vllm-openai:v0.27.1) - or set a default "
                "with `fllame config set-default-image` to skip this next time",
                default=_FALLBACK_IMAGE,
            )
    else:
        if image is None:
            configured_default = config_file.get_default_image()
            image = typer.prompt("Docker image", default=configured_default or _FALLBACK_IMAGE)
            if configured_default is not None and image == configured_default:
                image = None  # follow the configured default rather than pin it

        preinstall = _read_block(
            "Preinstall commands to run before `vllm serve`, one per line "
            "(e.g. `pip install -U transformers`) - blank line or Ctrl-D to skip:"
        )

        env = {}
        for line in _read_block(
            "Environment variables, one KEY=VALUE per line - blank line or Ctrl-D "
            "to skip:"
        ):
            try:
                key, value = parse_env_line(line)
            except RecipePasteError as e:
                typer.echo(str(e), err=True)
                raise typer.Exit(code=1) from e
            env[key] = value

        command = _read_command()

    preinstall, uv_pip_replaced = _replace_uv_pip_install(preinstall)
    if uv_pip_replaced:
        typer.echo(
            "note: replaced 'uv pip install' with 'pip install' in the preinstall "
            "step - the vllm/vllm-openai image's own Python environment isn't the "
            "uv-managed venv 'uv pip install' expects."
        )

    warn_image = image if image is not None else config_file.get_default_image()
    if warn_image is not None and (warn_image.endswith(":latest") or ":" not in warn_image):
        typer.echo(
            "warning: using an unpinned image tag - pin it to a specific "
            "version once you've confirmed this recipe works.",
            err=True,
        )

    repo_id, _ = parse_vllm_serve_command(command)
    store = _recipe_store()
    handle = store.next_available_handle(derive_handle(repo_id))
    try:
        recipe = Recipe.from_dict(
            handle,
            {
                "command": command,
                "image": image,
                "gpus": gpus,
                "env": env,
                "preinstall": preinstall,
            },
        )
    except RecipeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    store.save(recipe)
    typer.echo(f"saved recipe '{handle}' -> {config.recipe_dir(handle) / 'recipe.yaml'}")

    if pull:
        typer.echo(f"pulling '{recipe.repo_id}' into {config.hf_cache_dir()}")
        try:
            pull_model(recipe.repo_id)
        except (HfHubHTTPError, RequestException) as e:
            typer.echo(_friendly_download_error(e), err=True)
            raise typer.Exit(code=1) from e
        except PermissionError as e:
            typer.echo(_friendly_permission_error(e), err=True)
            raise typer.Exit(code=1) from e

    if build:
        _build_or_exit(recipe)


@recipe_app.command("build")
def recipe_build(handle: str) -> None:
    """Regenerate HANDLE's `compose.yaml`.

    Fails with a clear error if the model isn't fully downloaded yet -
    run `fllame model pull <repo_id>` (or `recipe add --pull`/`--build`)
    first.
    """
    _build_or_exit(_load_or_exit(handle))


def _validate_after_edit(handle: str, path: Path) -> Recipe | RecipeError:
    """Validates HANDLE's just-edited recipe file, trying a narrow
    whitespace autofix (see `autofix_whitespace`) once before giving up -
    catches a stray tab/CRLF from an editor without ever guessing at the
    file's intended structure (`RecipeStore.load`'s own leniency about a
    `command` block's indentation/trailing `\\` handles the far more
    common edit mistake before this is even needed). Returns the loaded
    Recipe on success.
    """
    try:
        return _recipe_store().load(handle)
    except RecipeError as first_error:
        raw = path.read_text()
        fixed = autofix_whitespace(raw)
        if fixed == raw:
            return first_error

    path.write_text(fixed)
    try:
        return _recipe_store().load(handle)
    except RecipeError as second_error:
        return second_error


@recipe_app.command("edit")
def recipe_edit(handle: str) -> None:
    """Open HANDLE's recipe file in $EDITOR, then re-validate it.

    Lenient about how the `command` block ends up formatted (missing
    indentation, a missing trailing `\\`, ...) and about a stray tab or
    CRLF line ending elsewhere - fixed automatically before anything is
    reported, and the file is re-saved in fllame's own canonical
    rendering once it validates, regardless of which of those kicked in.
    Anything else invalid offers a choice: reopen $EDITOR to fix it, or
    revert to the version from before this edit (kept in memory for the
    length of this command, not written to a backup file - the recipes
    directory is meant to be git-tracked already, which is the real
    backup).
    """
    path = config.recipe_dir(handle) / "recipe.yaml"
    if not path.is_file():
        typer.echo(f"no recipe found for '{handle}' (expected {path})", err=True)
        raise typer.Exit(code=1)

    original_text = path.read_text()
    click.edit(filename=str(path))

    while True:
        result = _validate_after_edit(handle, path)
        if isinstance(result, Recipe):
            _recipe_store().save(result)
            typer.echo(f"'{handle}' saved and valid.")
            return

        typer.echo(f"'{handle}' is no longer a valid recipe: {result}", err=True)
        if typer.confirm(
            "Reopen $EDITOR to fix it? (No reverts to the version from before this edit)",
            default=True,
        ):
            click.edit(filename=str(path))
            continue

        path.write_text(original_text)
        typer.echo(f"reverted '{handle}' to its previous version")
        raise typer.Exit(code=1)


@recipe_app.command("remove")
def recipe_remove(
    handle: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
) -> None:
    """Delete HANDLE's whole folder - its recipe file and its generated
    `compose.yaml` together (see `fllame/config.py`'s `recipe_dir`).

    Doesn't stop a container that's still running under it; if `fllame
    stop HANDLE` matters, run it first.
    """
    if not yes and not typer.confirm(f"Delete recipe '{handle}'?"):
        raise typer.Exit(code=0)
    try:
        _recipe_store().remove(handle)
    except RecipeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    typer.echo(f"removed '{handle}'")


@config_app.command("show")
def config_show() -> None:
    """Print fllame's current persisted settings."""
    default_image = config_file.get_default_image()
    if default_image is not None:
        typer.echo(f"default_image: {default_image}")
    else:
        typer.echo(f"default_image: (unset - falls back to '{_FALLBACK_IMAGE}')")


def _compose_files_using_image(image: str) -> list[Path]:
    """Every recipe's `compose.yaml` whose service currently has `image`
    set to exactly this string - the literal-match set `config
    set-default-image` offers to update in place. A recipe pinned to
    something else, or a compose.yaml already hand-edited to a
    different image, is never included.
    """
    matches = []
    for handle in _recipe_store().list_handles():
        compose_path = config.recipe_dir(handle) / "compose.yaml"
        if not compose_path.is_file():
            continue
        try:
            existing = yaml.safe_load(compose_path.read_text())
            current_image = existing["services"][handle]["image"]
        except (yaml.YAMLError, KeyError, TypeError):
            continue
        if current_image == image:
            matches.append(compose_path)
    return matches


def _replace_image_in_compose_file(path: Path, old_image: str, new_image: str) -> None:
    """A literal text substitution of just the `image:` value, not a
    full regeneration - any other hand edits already in `path` (gpu
    pinning, an `shm_size:`, a removed `HF_HUB_OFFLINE`, ...) survive
    untouched, since compose.yaml is expected to be hand-edited at
    times (see README's "Advanced" section).
    """
    text = path.read_text()
    updated = text.replace(f"image: {old_image}", f"image: {new_image}", 1)
    path.write_text(updated)


@config_app.command("set-default-image")
def config_set_default_image(image: str) -> None:
    """Set the Docker image recipes fall back to when they don't pin
    their own.

    Only affects recipes that don't pin their own `image` - those keep
    using whatever they're pinned to either way. Existing `compose.yaml`
    files aren't regenerated by this command; if any currently use the
    previous default image verbatim, offers to update just that one
    value in place, in every such file, without touching anything else
    already there.
    """
    if image.endswith(":latest") or ":" not in image:
        typer.echo(
            "warning: using an unpinned image tag - pin it to a specific "
            "version once you've confirmed recipes work with it.",
            err=True,
        )
    old_default = config_file.get_default_image()
    config_file.set_default_image(image)
    typer.echo(f"default image set to '{image}'")

    if old_default is None or old_default == image:
        return

    affected = _compose_files_using_image(old_default)
    if not affected:
        return

    typer.echo(f"{len(affected)} compose.yaml file(s) still use the previous default image:")
    for path in affected:
        typer.echo(f"  {path}")
    if not typer.confirm("Replace it with the new image in these files?", default=False):
        return

    for path in affected:
        _replace_image_in_compose_file(path, old_default, image)
    typer.echo(f"updated {len(affected)} file(s)")


@hardware_app.command("scan")
def hardware_scan() -> None:
    """Detect GPU(s), VRAM, and RAM on this machine."""
    profile = scan_hardware()
    if not profile.has_gpu:
        typer.echo("gpu:            none detected (no nvidia-smi on PATH)")
    else:
        typer.echo(f"gpu:            {profile.gpu_name} x{profile.gpu_count}")
        if profile.vram_gb_per_gpu is not None:
            typer.echo(f"vram_per_gpu:   {profile.vram_gb_per_gpu:.1f} GB")
        else:
            typer.echo("vram_per_gpu:   unknown (unified memory - see 'ram' below)")
    typer.echo(f"chip_family:    {profile.chip_family}")
    ram = f"{profile.ram_gb:.1f} GB" if profile.ram_gb is not None else "unknown"
    typer.echo(f"ram:            {ram}")
    quantizations = ", ".join(profile.supported_quantizations) or "none"
    typer.echo(f"quantizations:  {quantizations}")


def _friendly_download_error(e: Exception) -> str:
    """A large multi-file `snapshot_download` can fail partway through
    on a routine, transient Hub error (rate limiting, a connection
    blip) - `huggingface_hub`'s own retry logic already handles a lot
    of this internally, but not every case, and its own exception
    message doesn't mention the one thing that matters most to the
    operator: files already downloaded stay in the cache, so re-running
    the same command resumes rather than starting over.
    """
    return (
        f"{e}\n"
        "Already-downloaded files stay cached - running the same command "
        "again will resume rather than start over."
    )


def _friendly_permission_error(e: PermissionError) -> str:
    """The HF cache is a shared directory - anything else that has ever
    written into it as a different user (most commonly: a `fllame
    serve` container that ran without `HF_HUB_OFFLINE` in effect, e.g.
    via the hand-edited-compose-file escape hatch, while the vLLM image
    ran as root) can leave files there this operator's own account no
    longer has permission to touch. fllame itself never elevates
    privileges anywhere in `pull_model`'s call graph, so this is always
    an external ownership mismatch, not a bug in the download itself -
    the fix is reclaiming ownership of the cache, not retrying.
    """
    cache_dir = config.hf_cache_dir()
    return (
        f"{e}\n"
        f"The Hugging Face cache ({cache_dir}) has files this user can't write to - "
        "likely left behind by something else (e.g. a container) that wrote there as "
        "a different user. Fix with:\n"
        f"  sudo chown -R $(id -u):$(id -g) {cache_dir}"
    )


@model_app.command("pull")
def model_pull(repo_id: str) -> None:
    """Download REPO_ID into the Hugging Face cache.

    Takes a Hugging Face repo_id directly (e.g. `org/repo`), not a
    recipe handle - `model` commands never depend on recipes at all,
    since a recipe is a higher-level abstraction built on top of a
    model, not the other way around (see CLAUDE.md, "Layering"). To
    pull the model a specific recipe needs, either `recipe show
    HANDLE` first to see its repo_id, or use `recipe add --pull`/
    `recipe build HANDLE` instead.
    """
    typer.echo(f"pulling '{repo_id}' into {config.hf_cache_dir()}")
    try:
        path = pull_model(repo_id)
    except (HfHubHTTPError, RequestException) as e:
        typer.echo(_friendly_download_error(e), err=True)
        raise typer.Exit(code=1) from e
    except PermissionError as e:
        typer.echo(_friendly_permission_error(e), err=True)
        raise typer.Exit(code=1) from e
    typer.echo(f"done: {path}")


@model_app.command("update")
def model_update(
    repo_id: str | None = typer.Argument(
        None,
        help="Check only REPO_ID; omit to check every model currently in "
        "the local cache.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Re-pull any model found stale, via the same download path "
        "`model pull` uses. Check-only by default - reports status, "
        "downloads nothing.",
    ),
) -> None:
    """Check cached models against the Hub for a newer revision.

    A setup-phase command, like `model pull`/`model scan` - `fllame
    serve` never checks this itself (see CLAUDE.md, "Setup vs.
    running"), so this is the only place staleness is ever surfaced.
    Check-only by default; `--apply` re-pulls anything stale, a no-op
    download-wise if nothing has actually changed, since it goes
    through the same `pull_model` `model pull` already uses.

    Takes a repo_id directly, not a recipe handle - same reasoning as
    `model pull` above.
    """
    if repo_id is not None:
        repo_ids = [repo_id]
    else:
        repo_ids = [repo.repo_id for repo in list_cached_models()]

    if not repo_ids:
        typer.echo("No models cached.")
        raise typer.Exit(code=0)

    try:
        statuses = [check_for_update(repo_id) for repo_id in repo_ids]
    except (HfHubHTTPError, RequestException) as e:
        typer.echo(f"Hugging Face Hub unreachable: {e}", err=True)
        raise typer.Exit(code=1) from e

    rows = []
    for status in statuses:
        if status.cached_revision is None:
            rows.append([status.repo_id, "not cached - run `model pull` first"])
        elif not status.is_stale:
            rows.append([status.repo_id, "up to date"])
        elif apply:
            try:
                pull_model(status.repo_id)
            except (HfHubHTTPError, RequestException):
                rows.append([status.repo_id, "download interrupted - rerun to resume"])
                continue
            except PermissionError as e:
                typer.echo(_friendly_permission_error(e), err=True)
                rows.append([status.repo_id, "permission denied - see message above"])
                continue
            rows.append([status.repo_id, "updated"])
        else:
            rows.append([status.repo_id, "stale"])

    _print_table(["REPO_ID", "STATUS"], rows)


@model_app.command("list")
def model_list() -> None:
    """List models currently present in the local Hugging Face cache -
    a filesystem scan, no network involved."""
    models = list_cached_models()
    if not models:
        typer.echo(f"No models cached in {config.hf_cache_dir()}")
        raise typer.Exit(code=0)
    _print_table(
        ["REPO_ID", "SIZE", "LAST_MODIFIED"],
        [[repo.repo_id, repo.size_on_disk_str, repo.last_modified_str] for repo in models],
    )


@model_app.command("scan")
def model_scan(
    query: str | None = typer.Option(
        None,
        "--query",
        "-q",
        show_default=False,
        help=(
            "Free-text filter, e.g. a model family name. Place multi-word queries "
            'between quotes (-q "qwen 3.8").'
        ),
    ),
    quantization: str | None = typer.Option(
        None,
        "--quant",
        show_default=False,
        help="Search only this quantization, ignoring the hardware scan's supported list.",
    ),
    max_size: float | None = typer.Option(
        None,
        "--max-size",
        show_default=False,
        help=(
            "Maximum estimated VRAM usage, in GB (see the EST. VRAM column) - "
            "defaults to this machine's hardware scan budget when not given, "
            "but is enforced either way."
        ),
    ),
    min_params: float | None = typer.Option(
        None,
        "--min-params",
        show_default=False,
        help="Minimum size, in billions of parameters.",
    ),
    max_params: float | None = typer.Option(
        None,
        "--max-params",
        show_default=False,
        help=(
            "Maximum size, in billions of parameters. Independent of --max-size "
            "and not applied unless given - there's no hardware-based default for it."
        ),
    ),
    limit: int = typer.Option(20, "--limit", help="Number of ranked results to show."),
) -> None:
    """Search the Hugging Face Hub for models, ranked by fit and popularity.

    --max-size is the primary size gate and is always in effect: give it
    explicitly, or it defaults to a coarse VRAM/RAM-based budget from
    this machine's hardware scan - a starting point, not a benchmarked
    guarantee a result actually fits (see CLAUDE.md for why there's no
    stronger guarantee yet). A result with no Hub-reported size at all
    (no safetensors metadata - e.g. a GGUF-only export) is excluded only
    when --max-size was given explicitly; against the hardware-scan
    default it's left in, unpenalized, rather than judged against a
    number nobody asked it to satisfy.

    --min-params/--max-params are a separate, optional restriction on
    declared parameter count layered on top, with no hardware-derived
    default of their own and the same explicit-only exclusion rule for
    an unknown value. Give neither and only --max-size applies; give
    --max-params and both restrictions apply, and ranking then weighs
    closeness to each equally alongside popularity/recency.

    --quant searches only that quantization, still ignoring the hardware
    scan's supported list either way.

    Columns: PARAMS is the Hub's own reported parameter count where
    known (falling back to a guess from the repo_id otherwise); EST.
    VRAM is a separate, independent minimum weights-only VRAM estimate
    computed directly from the checkpoint's real on-disk byte layout -
    not derived from PARAMS, so the two can disagree for quantization
    formats that pack multiple values into one stored byte. DOWNLOADS is
    the Hub's recent (~30-day) download count, and UPDATED is how long
    ago the repo was last modified - both also feed the ranking, along
    with all-time downloads (not separately shown). QUANT is omitted
    when every result already shares one quantization.
    """
    profile = scan_hardware() if quantization is None or max_size is None else None

    quantizations = [quantization] if quantization is not None else profile.supported_quantizations
    if not quantizations:
        typer.echo(
            "No supported quantizations detected for this hardware - "
            "pass --quant explicitly to search anyway.",
            err=True,
        )
        raise typer.Exit(code=1)

    if max_size is not None:
        max_size_gb = max_size
    else:
        budget_gb = memory_budget_gb(profile)
        if budget_gb is None:
            typer.echo(
                "Could not determine available VRAM/RAM for this hardware - "
                "pass --max-size explicitly to search anyway.",
                err=True,
            )
            raise typer.Exit(code=1)
        unified = profile.chip_family == "grace_blackwell"
        max_size_gb = usable_memory_gb(budget_gb=budget_gb, unified_memory=unified)

    try:
        candidates = search_models(
            quantizations=quantizations,
            max_size_gb=max_size_gb,
            exclude_unknown_size=max_size is not None,
            min_params_billion=min_params,
            max_params_billion=max_params,
            query=query,
            max_results=limit,
        )
    except (HfHubHTTPError, RequestException) as e:
        typer.echo(f"Hugging Face Hub unreachable: {e}", err=True)
        raise typer.Exit(code=1) from e

    if not candidates:
        typer.echo("No matching models found.")
        raise typer.Exit(code=0)

    show_quant_column = len(quantizations) > 1
    headers = ["REPO_ID", *(["QUANT"] if show_quant_column else []), "PARAMS", "EST. VRAM"]
    headers += ["DOWNLOADS", "UPDATED"]
    rows = [
        [
            c.repo_id,
            *([c.quantization] if show_quant_column else []),
            f"{c.params_billion:.1f}B" if c.params_billion is not None else "unknown",
            f"{c.estimated_vram_gb:.1f} GB" if c.estimated_vram_gb is not None else "unknown",
            _format_count(c.downloads),
            _format_relative_time(c.last_modified),
        ]
        for c in candidates
    ]
    _print_table(headers, rows)


def _warn_if_vram_likely_insufficient(recipe: Recipe, *, assume_yes: bool) -> None:
    """A weights-only, best-effort heads-up before `serve` launches a
    container - never a blocking verdict, and silently skipped whenever
    a confident comparison isn't possible (no GPU/RAM figure from the
    hardware scan, or the model has no cached `.safetensors` files to
    measure): a wrong "won't fit" warning is worse than none, so this
    only ever speaks up when it has a real number on both sides. Not
    the recipe-level VRAM verdict (weights + KV cache + `--max-model-
    len` + concurrency) still tracked as deferred in CLAUDE.md.

    Runs after `_require_model_cached` has already confirmed the model
    is fully cached (see `serve` below), and reads that same local
    cache (`models/cache.py`'s `local_estimate_vram_gb` - a pure
    filesystem scan, same data source as `model list`) rather than
    making a network call of its own - consistent with `serve` never
    touching the network at all.
    """
    profile = scan_hardware()
    budget_gb = memory_budget_gb(profile)
    if budget_gb is None:
        return
    usable_gb = usable_memory_gb(
        budget_gb=budget_gb, unified_memory=profile.chip_family == "grace_blackwell"
    )

    estimated_gb = local_estimate_vram_gb(recipe.repo_id)
    if estimated_gb is None or estimated_gb <= usable_gb:
        return

    typer.echo(
        f"warning: '{recipe.repo_id}' is estimated at {estimated_gb:.1f} GB of "
        f"weights alone - this machine's usable budget is {usable_gb:.1f} GB. This "
        "is a coarse, weights-only estimate (no KV cache/activations/concurrency), "
        "not a benchmarked verdict - it may still fit, or may not even with this "
        "margin.",
        err=True,
    )
    if not assume_yes and not typer.confirm("Continue anyway?", default=False):
        raise typer.Exit(code=1)


def _warn_if_cache_location_changed(recipe: Recipe, *, assume_yes: bool) -> None:
    """Warns and asks to confirm, before `serve` regenerates
    `compose.yaml`, if the HF cache directory it would now mount (see
    `config.hf_cache_dir`, which honors `HF_HOME`/`HF_HUB_CACHE`)
    differs from what's already baked into this recipe's existing
    `compose.yaml` - a silent switch here would point the container at
    a different cache than it last ran against, which may not have this
    model in it even though the old location still does. Silently
    skipped when there's no existing `compose.yaml` to compare against
    (first `serve`), or it can't be read/parsed - nothing to warn about
    either way, and letting the routine regeneration proceed is safe.
    """
    compose_path = config.recipe_dir(recipe.handle) / "compose.yaml"
    if not compose_path.is_file():
        return
    try:
        existing = yaml.safe_load(compose_path.read_text())
        old_host = cache_volume_host_path(existing["services"][recipe.handle])
    except (yaml.YAMLError, KeyError, TypeError, AttributeError):
        return

    new_service = generate_compose(recipe, backend=BACKEND, hf_cache_dir=config.hf_cache_dir())
    new_host = cache_volume_host_path(new_service["services"][recipe.handle])
    if old_host is None or old_host == new_host:
        return

    typer.echo(
        f"warning: the Hugging Face cache location has changed since this recipe's "
        f"compose.yaml was last generated:\n"
        f"  was: {old_host}\n"
        f"  now: {new_host}\n"
        "Continuing will point the container at the new location - make sure the "
        "model is cached there too (`fllame model pull` again if not).",
        err=True,
    )
    if not assume_yes and not typer.confirm("Continue and update compose.yaml?", default=False):
        raise typer.Exit(code=1)


@app.command()
def serve(
    handle: str,
    detach: bool = typer.Option(False, "--detach", "-d", help="Run in the background."),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the pre-serve VRAM/cache-location sanity checks' confirmation "
        "prompts - warnings (if any) are still printed.",
    ),
) -> None:
    """Launch the recipe for HANDLE as a Docker container via `docker
    compose`, from HANDLE's own self-contained compose folder
    (`fllame config` aside, entirely independent of every other
    recipe's).

    Never touches the network, full stop: the model must already be
    fully present in the HF cache - `fllame model pull <repo_id>`, or
    `recipe add HANDLE --pull` when the recipe was created, does that
    separately - and this only ever verifies that via a filesystem
    check, failing with a clear error if it isn't there rather than
    falling back to a download of its own. vLLM's own auto-download
    inside the container is never relied on either. A recipe with a
    preinstall step runs it as part of the container's own startup,
    every time - there's no separate image build step.

    Once the model is confirmed cached, compares a coarse, weights-only
    VRAM estimate (from the cached files themselves, no network) against
    this machine's hardware scan and warns (asking to confirm, unless
    `-y`) if it looks like it won't fit. Not a benchmarked guarantee
    either way, and silently skipped whenever a confident comparison
    isn't possible - see `fllame hardware scan`/`fllame model scan` for
    the same underlying estimate.

    Also warns (asking to confirm, unless `-y`) if the HF cache location
    this recipe's `compose.yaml` would now mount has changed since it was
    last generated - a silent switch there could point the container at
    a cache that doesn't have this model in it.
    """
    recipe = _load_or_exit(handle)
    _require_model_cached(recipe)
    _warn_if_vram_likely_insufficient(recipe, assume_yes=yes)
    _warn_if_cache_location_changed(recipe, assume_yes=yes)

    _write_recipe_compose(recipe)
    args = ["up", "-d", handle] if detach else ["up", handle]
    raise typer.Exit(code=_run_compose(handle, *args))


@app.command()
def status() -> None:
    """Show the state of every recipe's container via `docker compose
    ps`, one recipe at a time (each is its own compose project)."""
    store = _recipe_store()
    handles = store.list_handles()
    if not handles:
        typer.echo(f"No recipes found in {config.recipes_dir()}")
        raise typer.Exit(code=0)

    exit_code = 0
    for handle in handles:
        try:
            recipe = store.load(handle)
        except RecipeError as e:
            typer.echo(f"warning: {e}", err=True)
            continue
        _write_recipe_compose(recipe)
        typer.echo(f"== {handle} ==")
        code = _run_compose(handle, "ps")
        if code != 0:
            exit_code = code
    raise typer.Exit(code=exit_code)


@app.command()
def stop(handle: str) -> None:
    """Stop HANDLE's container via `docker compose stop`."""
    recipe = _load_or_exit(handle)
    _write_recipe_compose(recipe)
    raise typer.Exit(code=_run_compose(handle, "stop", handle))


if __name__ == "__main__":
    app()
