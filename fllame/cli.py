"""fllame - a headless CLI for vLLM serving.

`fllame serve <handle>` resolves a hand-edited recipe, guarantees the
model is fully downloaded, and runs it as a Docker container via
`docker compose`. See README.md for the recipe file format and CLAUDE.md
for the architecture this sits on.
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
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from requests.exceptions import RequestException

from fllame import config, config_file
from fllame.backends.vllm import VllmServingBackend
from fllame.compose.generator import generate_compose, write_compose_file
from fllame.domain.recipe import Recipe, RecipeError
from fllame.domain.vllm_command import (
    VllmCommandError,
    join_line_continuations,
    parse_vllm_serve_command,
)
from fllame.hardware.scanner import scan_hardware
from fllame.models.cache import list_cached_models
from fllame.models.discovery import search_models
from fllame.models.puller import pull_model
from fllame.models.sizing import memory_budget_gb, usable_memory_gb
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


def _write_compose_file(*, offline_handle: str | None = None) -> None:
    """Regenerates the compose file from every valid recipe on file. An
    invalid recipe is skipped with a warning rather than blocking every
    other recipe from being served.

    `offline_handle`, when set, forces that one service to run with
    HF_HUB_OFFLINE=1 - an invocation-time concern (`fllame serve
    --offline`), not a property of the recipe itself, so it's applied
    here rather than threaded through `Recipe`/`ServingBackend`.
    """
    store = _recipe_store()
    recipes = []
    for handle in store.list_handles():
        try:
            recipes.append(_resolve_image(store.load(handle)))
        except RecipeError as e:
            typer.echo(f"warning: {e}", err=True)
    compose = generate_compose(recipes, backend=BACKEND, hf_cache_dir=config.hf_cache_dir())
    if offline_handle is not None and offline_handle in compose["services"]:
        compose["services"][offline_handle]["environment"]["HF_HUB_OFFLINE"] = "1"
    write_compose_file(compose, config.compose_file_path())


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


def _run_compose(*args: str) -> int:
    command = [
        "docker",
        "compose",
        "-f",
        str(config.compose_file_path()),
        "-p",
        config.COMPOSE_PROJECT_NAME,
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
        "vllm serve command (required) - paste it, ending a continued line "
        "with \\ if needed, then a blank line or Ctrl-D:"
    )
    if not lines:
        typer.echo("no `vllm serve <repo_id> ...` command given", err=True)
        raise typer.Exit(code=1)
    joined = join_line_continuations("\n".join(lines)).strip()
    if "\n" in joined:
        typer.echo(
            "expected a single `vllm serve` command - continue a long one "
            "with a trailing \\ instead of a new, unrelated line",
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        parse_vllm_serve_command(joined)
    except VllmCommandError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    return joined


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
    typer.echo(f"saved recipe '{handle}' -> {config.recipes_dir() / f'{handle}.yaml'}")


def _validate_after_edit(handle: str, path: Path) -> RecipeError | None:
    """Validates HANDLE's just-edited recipe file, trying a narrow
    whitespace autofix (see `autofix_whitespace`) once before giving up -
    catches a stray tab/CRLF from an editor without ever guessing at the
    file's intended structure. `None` means valid.
    """
    try:
        _recipe_store().load(handle)
        return None
    except RecipeError as first_error:
        raw = path.read_text()
        fixed = autofix_whitespace(raw)
        if fixed == raw:
            return first_error

    path.write_text(fixed)
    try:
        _recipe_store().load(handle)
        return None
    except RecipeError as second_error:
        return second_error


@recipe_app.command("edit")
def recipe_edit(handle: str) -> None:
    """Open HANDLE's recipe file in $EDITOR, then re-validate it.

    A stray tab or CRLF line ending is fixed automatically before
    anything is reported. Anything else invalid offers a choice: reopen
    $EDITOR to fix it, or revert to the version from before this edit
    (kept in memory for the length of this command, not written to a
    backup file - the recipes directory is meant to be git-tracked
    already, which is the real backup).
    """
    path = config.recipes_dir() / f"{handle}.yaml"
    if not path.is_file():
        typer.echo(f"no recipe found for '{handle}' (expected {path})", err=True)
        raise typer.Exit(code=1)

    original_text = path.read_text()
    click.edit(filename=str(path))

    while True:
        error = _validate_after_edit(handle, path)
        if error is None:
            typer.echo(f"'{handle}' saved and valid.")
            return

        typer.echo(f"'{handle}' is no longer a valid recipe: {error}", err=True)
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
    """Delete HANDLE's recipe file."""
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


@config_app.command("set-default-image")
def config_set_default_image(image: str) -> None:
    """Set the Docker image recipes fall back to when they don't pin their own."""
    if image.endswith(":latest") or ":" not in image:
        typer.echo(
            "warning: using an unpinned image tag - pin it to a specific "
            "version once you've confirmed recipes work with it.",
            err=True,
        )
    config_file.set_default_image(image)
    typer.echo(f"default image set to '{image}'")


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


@model_app.command("pull")
def model_pull(handle: str) -> None:
    """Download HANDLE's model into the Hugging Face cache."""
    recipe = _load_or_exit(handle)
    typer.echo(f"pulling '{recipe.repo_id}' into {config.hf_cache_dir()}")
    path = pull_model(recipe.repo_id)
    typer.echo(f"done: {path}")


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


@app.command()
def serve(
    handle: str,
    detach: bool = typer.Option(False, "--detach", "-d", help="Run in the background."),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Never touch the network - fail if the model isn't already fully cached "
        "(pull it first with `fllame model pull`).",
    ),
) -> None:
    """Launch the recipe for HANDLE as a Docker container via `docker compose`.

    Always downloads the model first (see `model pull`) - vLLM's own
    auto-download inside the container is never relied on.
    """
    recipe = _load_or_exit(handle)
    if offline:
        typer.echo(f"resolving '{recipe.repo_id}' from the local cache only (--offline)")
    else:
        typer.echo(f"pulling '{recipe.repo_id}' into {config.hf_cache_dir()}")
    try:
        pull_model(recipe.repo_id, offline=offline)
    except LocalEntryNotFoundError as e:
        typer.echo(
            f"'{recipe.repo_id}' is not fully cached locally - run "
            f"`fllame model pull {handle}` while online first.",
            err=True,
        )
        raise typer.Exit(code=1) from e

    _write_compose_file(offline_handle=handle if offline else None)
    args = ["up", "-d", handle] if detach else ["up", handle]
    raise typer.Exit(code=_run_compose(*args))


@app.command()
def status() -> None:
    """Show the state of fllame-managed containers via `docker compose ps`."""
    _write_compose_file()
    raise typer.Exit(code=_run_compose("ps"))


@app.command()
def stop(handle: str) -> None:
    """Stop HANDLE's container via `docker compose stop`."""
    _load_or_exit(handle)
    _write_compose_file()
    raise typer.Exit(code=_run_compose("stop", handle))


if __name__ == "__main__":
    app()
