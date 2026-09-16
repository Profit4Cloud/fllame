"""fllame - a headless CLI for vLLM serving."""

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
from fllame.backends.vllm import (
    VllmServingBackend,
    cache_volume_host_path,
    generate_dockerfile,
    max_model_len_shortfall,
    tensor_parallel_size_mismatch_warning,
)
from fllame.compose.generator import generate_compose, write_compose_file
from fllame.domain.hardware import HardwareProfile
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
from fllame.models.sizing import SizingConfig, memory_budget_gb, usable_memory_gb
from fllame.models.updater import check_for_update
from fllame.recipes import build_state
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

_FALLBACK_IMAGE = "vllm/vllm-openai:latest"


def _recipe_store() -> RecipeStore:
    return RecipeStore(config.recipes_dir())


def _resolve_image(recipe: Recipe) -> Recipe:
    if recipe.image is not None:
        return recipe
    return dataclasses.replace(recipe, image=config_file.get_default_image() or _FALLBACK_IMAGE)


def _resolve_sizing_config() -> SizingConfig:
    """`SizingConfig`'s own field defaults are what a fresh install
    without `fllame config` settings gets - only override the fields
    that are actually persisted."""
    overrides = {}
    min_len = config_file.get_min_usable_max_model_len()
    if min_len is not None:
        overrides["min_usable_max_model_len"] = min_len
    overhead_gb = config_file.get_activation_overhead_gb()
    if overhead_gb is not None:
        overrides["activation_overhead_gb"] = overhead_gb
    return SizingConfig(**overrides)


def _load_or_exit(handle: str) -> Recipe:
    try:
        return _recipe_store().load(handle)
    except RecipeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e


def _write_recipe_compose(
    recipe: Recipe, *, hardware: HardwareProfile, sizing_config: SizingConfig
) -> str | None:
    """Writes `compose.yaml` and, for a recipe with `preinstall`, the
    `Dockerfile` it builds from - `compose.yaml`'s `image:` then points
    at the local tag `recipe build` is about to build, not the base
    image. Returns the Dockerfile content written, or `None` when this
    recipe has no `preinstall` (nothing to build - `Dockerfile` is left
    untouched either way, since it's a real, hand-editable artifact now,
    not cleanup-on-sight cruft from an older fllame version).
    """
    resolved = _resolve_image(recipe)
    directory = config.recipe_dir(resolved.handle)
    directory.mkdir(parents=True, exist_ok=True)

    dockerfile_content = generate_dockerfile(resolved)
    compose = generate_compose(
        resolved,
        backend=BACKEND,
        hf_cache_dir=config.hf_cache_dir(),
        hardware=hardware,
        sizing_config=sizing_config,
    )
    service = compose["services"][resolved.handle]
    # A stable, predictable name (matching the compose project name)
    # instead of docker compose's own auto-generated
    # `<project>-<service>-1` - lets `serve` print the right `docker
    # logs` command without an extra `docker compose ps` round-trip.
    service["container_name"] = config.compose_project_name(resolved.handle)
    if dockerfile_content is not None:
        (directory / "Dockerfile").write_text(dockerfile_content)
        service["image"] = config.local_image_tag(resolved.handle)

    write_compose_file(compose, directory / "compose.yaml")
    return dockerfile_content


def _require_model_cached(recipe: Recipe) -> None:
    if is_model_cached(recipe.repo_id):
        return
    typer.echo(
        f"'{recipe.repo_id}' is not fully cached locally - run "
        f"`fllame model pull {recipe.repo_id}` first.",
        err=True,
    )
    raise typer.Exit(code=1)


def _require_compose_built(recipe: Recipe) -> None:
    if (config.recipe_dir(recipe.handle) / "compose.yaml").is_file():
        return
    typer.echo(
        f"'{recipe.handle}' has no compose.yaml yet - run "
        f"`fllame recipe build {recipe.handle}` first.",
        err=True,
    )
    raise typer.Exit(code=1)


def _friendly_docker_build_error(handle: str, *, dockerfile: bool) -> str:
    rebuild = f"fllame recipe build {handle}"
    if dockerfile:
        return (
            f"`recipe build` couldn't build the Docker image for '{handle}' - see the "
            "`docker build` output above for the underlying error (a failing "
            "`preinstall` command is the usual cause). Fix the recipe's `preinstall` "
            f"list (or the generated Dockerfile directly), then re-run `{rebuild}`."
        )
    return (
        f"`recipe build` couldn't pull the image configured for '{handle}' - see the "
        "`docker compose pull` output above for the underlying error (a bad image tag, "
        "an unreachable registry, or missing credentials are the usual causes). Fix the "
        f"recipe's (or the configured default) image, then re-run `{rebuild}`."
    )


def _build_or_exit(recipe: Recipe, *, assume_yes: bool = False) -> None:
    """Shared by `recipe build` and `recipe add --build` - the only two
    commands that ever write `compose.yaml`/`Dockerfile`.

    Always validates the result with a real `docker build` (a recipe
    with `preinstall`) or `docker compose pull` (one without) - setup
    phase, so the network access and repeat-run cost are both fine
    (Docker's layer cache makes an unchanged rebuild cheap), and this is
    the only way to catch a broken preinstall command or an unresolvable
    image before `serve` ever tries to use it.
    """
    _require_model_cached(recipe)
    # Scanned once and reused below - a hardware change mid-command
    # would be a strange thing to chase, and scanning is cheap either
    # way (see `HardwareProfile`'s own docstring).
    hardware = scan_hardware()
    sizing_config = _resolve_sizing_config()
    _warn_if_cache_location_changed(
        recipe, assume_yes=assume_yes, hardware=hardware, sizing_config=sizing_config
    )
    tp_warning = tensor_parallel_size_mismatch_warning(recipe, hardware)
    if tp_warning is not None:
        typer.echo(tp_warning, err=True)
    dockerfile_content = _write_recipe_compose(
        recipe, hardware=hardware, sizing_config=sizing_config
    )
    directory = config.recipe_dir(recipe.handle)
    compose_path = directory / "compose.yaml"

    # Checked after compose.yaml is written (build_service's own
    # shortfall detection already left it without --max-model-len in
    # this case) and before the expensive Docker step below - fail fast
    # on the cheap check first.
    shortfall = max_model_len_shortfall(recipe, hardware, sizing_config)
    if shortfall is not None:
        typer.echo(
            f"{shortfall} This is advanced territory - hand-edit {compose_path}'s command "
            "directly (a smaller --max-model-len, or your own --gpu-memory-utilization) at "
            "your own risk; `serve` will flag that edit the same way it flags any other "
            "hand edit.",
            err=True,
        )
        raise typer.Exit(code=1)

    if dockerfile_content is not None:
        tag = config.local_image_tag(recipe.handle)
        typer.echo(f"building '{tag}' from {directory / 'Dockerfile'} ...")
        code = _run_docker("build", "-t", tag, str(directory))
    else:
        typer.echo("validating the configured image with `docker compose pull` ...")
        code = _run_compose(recipe.handle, "pull")

    if code != 0:
        typer.echo(
            _friendly_docker_build_error(recipe.handle, dockerfile=dockerfile_content is not None),
            err=True,
        )
        raise typer.Exit(code=1)

    build_state.save(directory, build_state.for_current_files(directory))
    typer.echo(f"wrote {compose_path}")


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Left-aligned, space-padded columns - `docker ps` style, no
    border characters."""
    all_rows = [headers, *rows]
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]
    for row in all_rows:
        padded = [cell.ljust(width) for cell, width in zip(row[:-1], widths[:-1], strict=False)]
        typer.echo("  ".join([*padded, row[-1]]))


def _format_count(n: int | None) -> str:
    """e.g. "12.3k", "1.2M"."""
    if n is None:
        return "unknown"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


# (label, seconds-per-unit, largest value still shown in this unit
# before rolling up to the next - None for the last, open-ended unit).
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


def _run_docker(*args: str) -> int:
    try:
        return subprocess.run(["docker", *args]).returncode
    except FileNotFoundError as e:
        typer.echo(
            "docker (or the compose plugin) was not found on PATH - fllame runs "
            "vLLM as a Docker container, install Docker to use this command.",
            err=True,
        )
        raise typer.Exit(code=1) from e


def _run_compose(handle: str, *args: str) -> int:
    return _run_docker(
        "compose",
        "-f",
        str(config.recipe_dir(handle) / "compose.yaml"),
        "-p",
        config.compose_project_name(handle),
        *args,
    )


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
    """Reads until a blank line or EOF - unlike `sys.stdin.read()`,
    which consumes to the *first* EOF and leaves nothing for a later
    call. `#` comment lines are skipped."""
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
    """The vllm/vllm-openai image's own Python environment isn't the
    uv-managed venv `uv pip install` expects; `pip install` targets its
    site-packages directly. Caller must tell the operator when the
    returned bool is True, since it changes what they typed.
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
def recipe_build(
    handle: str,
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the cache-location-changed confirmation prompt - the "
        "warning (if any) is still printed.",
    ),
) -> None:
    """Write (or overwrite) HANDLE's `compose.yaml` - the only command,
    besides `recipe add --build`, that ever does.

    Fails with a clear error if the model isn't fully downloaded yet -
    run `fllame model pull <repo_id>` (or `recipe add --pull`/`--build`)
    first. `serve`/`status`/`stop` never write or regenerate
    `compose.yaml` themselves - it's meant to be hand-edited, and only
    ever touched again by explicitly re-running this command.

    Also validates the result with real Docker: a recipe with
    `preinstall` gets a `Dockerfile` (same hand-edit contract as
    `compose.yaml`) built into a local image `compose.yaml` then points
    at; one without gets its configured image checked with `docker
    compose pull`. Both catch a broken preinstall command or an
    unresolvable image now, in this setup-phase command, rather than
    later at `serve` time.

    Computes `--max-model-len` the same way, unless the recipe sets its
    own: if even fllame's minimum usable context doesn't fit this
    model's cached weights and this hardware's memory budget, this
    command hard-fails before the Docker step, since serving an
    unusably short context by default would be worse than an explicit
    error. Also warns (never blocks) when the recipe's
    `--tensor-parallel-size` doesn't match the number of GPUs detected.
    """
    _build_or_exit(_load_or_exit(handle), assume_yes=yes)


def _validate_after_edit(handle: str, path: Path) -> Recipe | RecipeError:
    """Tries `autofix_whitespace` once before giving up."""
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
    length of this command, not written to a backup file).
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
    `compose.yaml`/`Dockerfile` together (see `fllame/config.py`'s
    `recipe_dir`).

    Doesn't stop a container that's still running under it; if `fllame
    stop HANDLE` matters, run it first. Also doesn't remove any Docker
    image built or pulled for it - Docker holds that state, not fllame,
    regardless of whether this recipe had a `Dockerfile`; `docker image
    prune`/`docker rmi` is the way to reclaim that space.
    """
    if not yes and not typer.confirm(f"Delete recipe '{handle}'?"):
        raise typer.Exit(code=0)
    try:
        _recipe_store().remove(handle)
    except RecipeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    typer.echo(f"removed '{handle}'")
    typer.echo(
        "note: this did not remove any Docker image built or pulled for it - "
        "run `docker image prune`/`docker rmi` yourself if you want that space back."
    )


@config_app.command("show")
def config_show() -> None:
    """Print fllame's current persisted settings."""
    default_image = config_file.get_default_image()
    if default_image is not None:
        typer.echo(f"default_image: {default_image}")
    else:
        typer.echo(f"default_image: (unset - falls back to '{_FALLBACK_IMAGE}')")

    defaults = SizingConfig()
    min_len = config_file.get_min_usable_max_model_len()
    if min_len is not None:
        typer.echo(f"min_usable_max_model_len: {min_len}")
    else:
        typer.echo(
            f"min_usable_max_model_len: (unset - falls back to {defaults.min_usable_max_model_len})"
        )

    overhead_gb = config_file.get_activation_overhead_gb()
    if overhead_gb is not None:
        typer.echo(f"activation_overhead_gb: {overhead_gb}")
    else:
        typer.echo(
            f"activation_overhead_gb: (unset - falls back to {defaults.activation_overhead_gb})"
        )


def _compose_files_using_image(image: str) -> list[Path]:
    """Recipes whose `compose.yaml` service `image` is an exact match -
    never one pinned to something else or already hand-edited."""
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
    """A text substitution, not a regeneration - other hand edits in
    `path` survive untouched."""
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

    # fllame's own edit, not a hand-edit - recalculate compose_hash right
    # away so `serve`'s hand-edit check (point 5a) never mistakes this
    # for one, and mark the image as not yet re-validated (point 5c)
    # until the shared pull below confirms - or fails to confirm - it.
    for path in affected:
        directory = path.parent
        state = build_state.load(directory)
        build_state.save(
            directory,
            dataclasses.replace(
                state,
                compose_hash=build_state.hash_text(path.read_text()),
                image_synced_via_config=True,
            ),
        )

    typer.echo(f"pulling '{image}' to confirm it resolves ...")
    if _run_docker("pull", image) == 0:
        for path in affected:
            directory = path.parent
            state = build_state.load(directory)
            build_state.save(directory, dataclasses.replace(state, image_synced_via_config=False))
    else:
        typer.echo(
            f"warning: couldn't pull '{image}' - the config change and the file "
            "update above are kept either way, but `fllame serve` will keep noting "
            "that this image hasn't been re-validated until it does.",
            err=True,
        )


@config_app.command("set-min-context-length")
def config_set_min_context_length(tokens: int) -> None:
    """Set the minimum usable `--max-model-len`, in tokens.

    `recipe build` computes a default `--max-model-len` (see `fllame
    recipe build -h`) sized to fit this hardware; below this many
    tokens, it's judged not worth serving, and `recipe build` fails
    instead of injecting a too-short value. Only affects future `recipe
    build` runs, not compose.yaml files already written.
    """
    if tokens <= 0:
        typer.echo("minimum usable context length must be a positive number of tokens", err=True)
        raise typer.Exit(code=1)
    config_file.set_min_usable_max_model_len(tokens)
    typer.echo(f"minimum usable context length set to {tokens} tokens")


@config_app.command("set-activation-overhead")
def config_set_activation_overhead(gb: float) -> None:
    """Set the fixed GB reserved for activation memory/other overhead
    beyond weights and KV cache, when `recipe build` computes a default
    `--max-model-len`. A coarse allowance, not modeled per-architecture
    - raise it if models are still running out of memory at their
    computed default, lower it to allow a longer default context.
    """
    if gb < 0:
        typer.echo("activation overhead must not be negative", err=True)
        raise typer.Exit(code=1)
    config_file.set_activation_overhead_gb(gb)
    typer.echo(f"activation/overhead allowance set to {gb} GB")


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
    return (
        f"{e}\n"
        "Already-downloaded files stay cached - running the same command "
        "again will resume rather than start over."
    )


def _friendly_permission_error(e: PermissionError) -> str:
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
    """Silently skipped whenever a confident comparison isn't possible -
    a wrong "won't fit" warning is worse than none."""
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


def _warn_if_cache_location_changed(
    recipe: Recipe, *, assume_yes: bool, hardware: HardwareProfile, sizing_config: SizingConfig
) -> None:
    """Overwriting silently would point the container at a cache that
    may not have this model in it. Skipped on a recipe's first build,
    or an unparseable existing file."""
    compose_path = config.recipe_dir(recipe.handle) / "compose.yaml"
    if not compose_path.is_file():
        return
    try:
        existing = yaml.safe_load(compose_path.read_text())
        old_host = cache_volume_host_path(existing["services"][recipe.handle])
    except (yaml.YAMLError, KeyError, TypeError, AttributeError):
        return

    new_service = generate_compose(
        recipe,
        backend=BACKEND,
        hf_cache_dir=config.hf_cache_dir(),
        hardware=hardware,
        sizing_config=sizing_config,
    )
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


def _warn_if_compose_hand_edited(recipe: Recipe, *, assume_yes: bool) -> None:
    """Case (a): `compose.yaml` no longer matches what the last
    successful `recipe build` wrote - skipped when there's no recorded
    hash to compare against at all (an old build-state-less compose.yaml,
    or a compose.yaml written some other way)."""
    directory = config.recipe_dir(recipe.handle)
    state = build_state.load(directory)
    if state.compose_hash is None:
        return
    current_hash = build_state.hash_text((directory / "compose.yaml").read_text())
    if current_hash == state.compose_hash:
        return

    typer.echo(
        f"warning: compose.yaml no longer matches the last build for '{recipe.handle}' - "
        "it looks hand-edited since `fllame recipe build` last wrote it.",
        err=True,
    )
    if not assume_yes and not typer.confirm(
        "Continue serving the current compose.yaml?", default=False
    ):
        raise typer.Exit(code=1)


def _warn_if_dockerfile_hand_edited(recipe: Recipe, *, assume_yes: bool) -> None:
    """Case (b): same idea as above for `Dockerfile`, when this recipe
    has one - the message names the exact command to run, since a stale
    image is otherwise invisible (the container keeps running whatever
    was last actually built)."""
    directory = config.recipe_dir(recipe.handle)
    dockerfile_path = directory / "Dockerfile"
    if not dockerfile_path.is_file():
        return
    state = build_state.load(directory)
    if state.dockerfile_hash is None:
        return
    if build_state.hash_text(dockerfile_path.read_text()) == state.dockerfile_hash:
        return

    typer.echo(
        f"warning: Dockerfile no longer matches the last build for '{recipe.handle}' - "
        "it looks hand-edited since. The running image won't reflect that change until "
        f"it's rebuilt - run `fllame recipe build {recipe.handle}` (or `docker build -t "
        f"{config.local_image_tag(recipe.handle)} {directory}`).",
        err=True,
    )
    if not assume_yes and not typer.confirm(
        "Continue serving the current image anyway?", default=False
    ):
        raise typer.Exit(code=1)


def _note_if_image_not_yet_revalidated(recipe: Recipe) -> None:
    """Case (c): `config set-default-image` already edited this
    recipe's compose.yaml itself, so it's not a hand-edit and doesn't go
    through the two checks above at all - it's flagged separately (see
    `image_synced_via_config`) and only ever informational, never a
    confirmation prompt, since it was already validated once via a
    `docker pull` at the time (or the warning from that failed pull
    already told the operator about it)."""
    directory = config.recipe_dir(recipe.handle)
    if not build_state.load(directory).image_synced_via_config:
        return
    try:
        compose = yaml.safe_load((directory / "compose.yaml").read_text())
        image = compose["services"][recipe.handle]["image"]
    except (yaml.YAMLError, KeyError, TypeError):
        image = "the configured default image"
    typer.echo(
        f"note: default image changed to '{image}' since this was last built - not "
        "re-validated with a full `fllame recipe build`."
    )


@app.command()
def serve(
    handle: str,
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip this command's confirmation prompts (compose.yaml/Dockerfile "
        "hand-edit and VRAM sanity checks) - any warning is still printed.",
    ),
) -> None:
    """Launch the recipe for HANDLE as a Docker container via `docker
    compose up -d`, from HANDLE's own self-contained compose folder
    (`fllame config` aside, entirely independent of every other
    recipe's). Always detached - there's no foreground mode - since
    that's what gives this command a single, immediate "did it start OK"
    signal to act on below.

    Never touches the network, full stop: the model must already be
    fully present in the HF cache - `fllame model pull <repo_id>`, or
    `recipe add HANDLE --pull` when the recipe was created, does that
    separately - and this only ever verifies that via a filesystem
    check, failing with a clear error if it isn't there rather than
    falling back to a download of its own. vLLM's own auto-download
    inside the container is never relied on either.

    Never writes or regenerates `compose.yaml` either - that's `recipe
    build`'s job alone, so a hand edit to it survives every `serve`.
    Fails with a clear error pointing at `fllame recipe build HANDLE`
    if it doesn't exist yet.

    Once the model is confirmed cached, compares a coarse, weights-only
    VRAM estimate (from the cached files themselves, no network) against
    this machine's hardware scan and warns (asking to confirm, unless
    `-y`) if it looks like it won't fit. Not a benchmarked guarantee
    either way, and silently skipped whenever a confident comparison
    isn't possible - see `fllame hardware scan`/`fllame model scan` for
    the same underlying estimate.

    Also compares `compose.yaml`/`Dockerfile` against the hashes
    recorded by the last successful `recipe build`: a hand-edit to
    either warns and asks to confirm (unless `-y`); an image changed
    only by `fllame config set-default-image` since then is just noted,
    never blocked on, since that edit was already validated with its own
    `docker pull` at the time.

    A successful start (`docker compose up -d` itself exits 0 - a
    compose-level check, not a deeper vLLM health check, which is out of
    scope) refreshes those recorded hashes and clears the
    not-yet-re-validated note, so none of the above ever nags about
    something that's since gone away, and prints the exact `docker logs`
    command to follow the container's own startup - model loading can
    take several minutes, so a quiet log right after this returns is
    expected, not itself a problem.
    """
    recipe = _load_or_exit(handle)
    _require_model_cached(recipe)
    _require_compose_built(recipe)
    _warn_if_compose_hand_edited(recipe, assume_yes=yes)
    _warn_if_dockerfile_hand_edited(recipe, assume_yes=yes)
    _note_if_image_not_yet_revalidated(recipe)
    _warn_if_vram_likely_insufficient(recipe, assume_yes=yes)

    code = _run_compose(handle, "up", "-d", handle)
    if code == 0:
        directory = config.recipe_dir(handle)
        build_state.save(directory, build_state.for_current_files(directory))
        container_name = config.compose_project_name(handle)
        typer.echo(f"'{handle}' started - follow its logs with: docker logs -f {container_name}")
        typer.echo(
            "model loading can take several minutes - an empty or quiet log right "
            "after this returns is expected, not a problem."
        )
    raise typer.Exit(code=code)


@app.command()
def status() -> None:
    """Show the state of every recipe's container via `docker compose
    ps`, one recipe at a time (each is its own compose project). Reads
    whatever `compose.yaml` is already on disk - never regenerates it."""
    store = _recipe_store()
    handles = store.list_handles()
    if not handles:
        typer.echo(f"No recipes found in {config.recipes_dir()}")
        raise typer.Exit(code=0)

    exit_code = 0
    for handle in handles:
        try:
            store.load(handle)
        except RecipeError as e:
            typer.echo(f"warning: {e}", err=True)
            continue
        typer.echo(f"== {handle} ==")
        if not (config.recipe_dir(handle) / "compose.yaml").is_file():
            typer.echo(f"not built - run `fllame recipe build {handle}`")
            continue
        code = _run_compose(handle, "ps")
        if code != 0:
            exit_code = code
    raise typer.Exit(code=exit_code)


@app.command()
def stop(handle: str) -> None:
    """Stop HANDLE's container via `docker compose stop`. Reads
    whatever `compose.yaml` is already on disk - never regenerates it."""
    recipe = _load_or_exit(handle)
    _require_compose_built(recipe)
    raise typer.Exit(code=_run_compose(handle, "stop", handle))


if __name__ == "__main__":
    app()
