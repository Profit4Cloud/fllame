"""fllame - a headless CLI for vLLM serving."""

from __future__ import annotations

import dataclasses
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import click
import typer
import yaml
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import RequestException
from typer.core import TyperGroup

from fllame import config, config_file
from fllame.backends.vllm import (
    VllmServingBackend,
    cache_volume_host_path,
    generate_dockerfile,
    tensor_parallel_size_mismatch_warning,
    validate_gpu_memory_utilization,
)
from fllame.bench.sweep import (
    DEFAULT_CONCURRENCY,
    DEFAULT_INPUT_LEN,
    DEFAULT_OUTPUT_LEN,
    Level,
    SweepError,
    bench_command,
    build_levels,
    column_widths,
    columns_help,
    format_row,
    header_row,
    parse_int_list,
    result_cells,
)
from fllame.compose.generator import generate_compose, write_compose_file
from fllame.domain.recipe import Recipe, RecipeError
from fllame.domain.vllm_command import (
    VllmCommandError,
    extract_flag_value,
    join_command_lines,
    parse_vllm_serve_command,
)
from fllame.hardware.scanner import scan_hardware
from fllame.models.cache import is_model_cached, list_cached_models, local_estimate_vram_gb
from fllame.models.discovery import search_models
from fllame.models.puller import pull_model
from fllame.models.sizing import memory_budget_gb, usable_memory_gb
from fllame.models.updater import check_for_update
from fllame.models.vram import VramEstimateError, estimate_vram
from fllame.recipes.naming import derive_handle
from fllame.recipes.parser import RecipePasteError, parse_env_line, parse_pasted_recipe
from fllame.recipes.store import RecipeStore, autofix_whitespace

# `--help` is Click's default; `-h` is the standard Unix short form on
# top of it - wired in explicitly since Click doesn't bind it by default.
_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

# Typer always lists plain @app.command()s before add_typer() sub-apps,
# regardless of registration order, so the top-level `--help` listing
# can't otherwise follow the happy path (see CLAUDE.md).
_TOP_LEVEL_COMMAND_ORDER = (
    "config",
    "hardware",
    "model",
    "recipe",
    "serve",
    "bench",
    "status",
    "stop",
)


class _TopLevelGroup(TyperGroup):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(_TOP_LEVEL_COMMAND_ORDER)


app = typer.Typer(
    cls=_TopLevelGroup,
    no_args_is_help=True,
    add_completion=False,
    context_settings=_CONTEXT_SETTINGS,
)
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


def _load_or_exit(handle: str) -> Recipe:
    try:
        return _recipe_store().load(handle)
    except RecipeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e


def _write_recipe_compose(recipe: Recipe, *, default_gpu_memory_utilization: float) -> str | None:
    """Writes `compose.yaml` for a recipe. If the recipe has preinstall lines,
    also a `Dockerfile` is generated, which is used by `compose.yaml`.
    Returns the Dockerfile content written, or `None` when this
    recipe has no `preinstall`.
    """
    resolved = _resolve_image(recipe)
    directory = config.recipe_dir(resolved.handle)
    directory.mkdir(parents=True, exist_ok=True)

    dockerfile_content = generate_dockerfile(resolved)
    compose = generate_compose(
        resolved,
        backend=BACKEND,
        hf_cache_dir=config.hf_cache_dir(),
        default_gpu_memory_utilization=default_gpu_memory_utilization,
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
    """Builds the recipe by creating a `compose.yaml` (and possibly a `Dockerfile`) in its folder.
    Validates the result with `docker build` (a recipe with `preinstall`) or `docker compose
    pull` (one without). This ensures that network access is needed during build time only,
    and not during serve time.
    """
    _require_model_cached(recipe)
    # Hardware scanned for validing the tensor-parallel-size later.
    hardware = scan_hardware()
    default_gpu_memory_utilization = config_file.get_default_gpu_memory_utilization()
    _warn_if_cache_location_changed(
        recipe, assume_yes=assume_yes, default_gpu_memory_utilization=default_gpu_memory_utilization
    )
    tp_warning = tensor_parallel_size_mismatch_warning(recipe, hardware)
    if tp_warning is not None:
        typer.echo(tp_warning, err=True)
    dockerfile_content = _write_recipe_compose(
        recipe, default_gpu_memory_utilization=default_gpu_memory_utilization
    )
    directory = config.recipe_dir(recipe.handle)
    compose_path = directory / "compose.yaml"

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


_DOCKER_NOT_FOUND_MESSAGE = (
    "docker (or the compose plugin) was not found on PATH - fllame runs "
    "vLLM as a Docker container, install Docker to use this command."
)


def _run_docker(*args: str) -> int:
    try:
        return subprocess.run(["docker", *args]).returncode
    except FileNotFoundError as e:
        typer.echo(_DOCKER_NOT_FOUND_MESSAGE, err=True)
        raise typer.Exit(code=1) from e


def _compose_args(handle: str, *args: str) -> list[str]:
    return [
        "compose",
        "-f",
        str(config.recipe_dir(handle) / "compose.yaml"),
        "-p",
        config.compose_project_name(handle),
        *args,
    ]


def _run_compose(handle: str, *args: str) -> int:
    return _run_docker(*_compose_args(handle, *args))


def _parse_compose_ps_json(output: str) -> list[dict]:
    """`docker compose ps --format json` - a single JSON array on some
    docker versions, one JSON object per line on others."""
    output = output.strip()
    if not output:
        return []
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return [json.loads(line) for line in output.splitlines() if line.strip()]
    return data if isinstance(data, list) else [data]


def _format_ports(publishers: list[dict]) -> str:
    """Reconstructs `docker compose ps`'s own PORTS column (e.g.
    `0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp`) from its JSON
    `Publishers` field - an IPv6 bind address gets bracketed, same as
    docker's own display."""
    parts = []
    for publisher in publishers:
        published = publisher.get("PublishedPort")
        if not published:
            continue
        url = publisher.get("URL") or ""
        host = f"[{url}]" if ":" in url else url
        protocol = publisher.get("Protocol") or "tcp"
        parts.append(f"{host}:{published}->{publisher.get('TargetPort')}/{protocol}")
    return ", ".join(parts)


def _compose_ps_json(handle: str) -> tuple[list[dict], int]:
    """HANDLE's containers via `docker compose ps --all --format json`
    (`--all` so a stopped-but-not-removed container still shows up, not
    just a running one) - an empty list and the failing exit code if
    the command itself failed (its stderr is echoed either way)."""
    try:
        result = subprocess.run(
            ["docker", *_compose_args(handle, "ps", "--all", "--format", "json")],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        typer.echo(_DOCKER_NOT_FOUND_MESSAGE, err=True)
        raise typer.Exit(code=1) from e
    if result.returncode != 0:
        if result.stderr:
            typer.echo(result.stderr.rstrip(), err=True)
        return [], result.returncode
    return _parse_compose_ps_json(result.stdout), 0


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
def recipe_show(handle: str = typer.Argument(..., show_default=False)) -> None:
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
        show_default=False,
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
            "Environment variables, one KEY=VALUE per line - blank line or Ctrl-D " "to skip:"
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
    handle: str = typer.Argument(..., show_default=False),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the cache-location-changed confirmation prompt - the "
        "warning (if any) is still printed.",
    ),
) -> None:
    """Builds the recipe by creating a compose.yaml (and possibly a Dockerfile) in its folder.
    Fails with a clear error if the model isn't fully downloaded yet.
    Validates the result with docker build (a recipe with preinstall) or docker compose pull
    (one without). This ensures that network access is needed during build time only, and not
    during serve time.
    """
    _build_or_exit(_load_or_exit(handle), assume_yes=yes)


_TOKEN_COUNT_SUFFIXES = {"": 1, "k": 1000, "K": 1024, "m": 1000**2, "M": 1024**2}


def _parse_token_count(flag: str, value: str) -> int:
    """vLLM's own convention for `--max-model-len`: lowercase suffixes
    are decimal, uppercase binary (32k = 32000, 32K = 32768)."""
    match = re.fullmatch(r"(\d+)([kKmM]?)", value.strip())
    if match is None:
        typer.echo(f"{flag} must be a whole number of tokens, got {value!r}.", err=True)
        raise typer.Exit(code=1)
    return int(match.group(1)) * _TOKEN_COUNT_SUFFIXES[match.group(2)]


@recipe_app.command("vram")
def recipe_vram(
    handle: str = typer.Argument(..., show_default=False),
    max_model_len: str | None = typer.Option(
        None,
        "--max-model-len",
        show_default=False,
        help="Context length per request, overriding the recipe's own value.",
    ),
    max_num_seqs: int | None = typer.Option(
        None,
        "--max-num-seqs",
        show_default=False,
        help="Concurrent requests, overriding the recipe's own value.",
    ),
    details: bool = typer.Option(False, "--details", help="Show how the estimate is calculated."),
) -> None:
    """Estimate the VRAM HANDLE's model needs as served by its recipe, from the pulled
    model's config.json and weight files.

    Use it to pick --gpu-memory-utilization: vLLM claims that share of GPU memory whatever
    the model needs, and fills what the weights leave with KV cache.
    """
    recipe = _load_or_exit(handle)
    args = recipe.serve_args

    raw_max_model_len = max_model_len or extract_flag_value(args, "--max-model-len")
    raw_max_num_seqs = (
        str(max_num_seqs)
        if max_num_seqs is not None
        else extract_flag_value(args, "--max-num-seqs")
    )
    missing = [
        flag
        for flag, value in (
            ("--max-model-len", raw_max_model_len),
            ("--max-num-seqs", raw_max_num_seqs),
        )
        if value is None
    ]
    if missing:
        flags = " and ".join(missing)
        them = "them" if len(missing) > 1 else "it"
        example = " ".join(f"{flag} N" for flag in missing)
        typer.echo(
            f"{flags} not found in recipe '{handle}'. Add {them} to this command with "
            f"a value: fllame recipe vram {handle} {example}",
            err=True,
        )
        raise typer.Exit(code=1)

    tensor_parallel_size = extract_flag_value(args, "--tensor-parallel-size") or extract_flag_value(
        args, "-tp"
    )
    try:
        estimate = estimate_vram(
            recipe.repo_id,
            max_model_len=_parse_token_count("--max-model-len", raw_max_model_len),
            max_num_seqs=_parse_token_count("--max-num-seqs", raw_max_num_seqs),
            kv_cache_dtype=extract_flag_value(args, "--kv-cache-dtype") or "auto",
            tensor_parallel_size=int(tensor_parallel_size or 1),
        )
    except (VramEstimateError, ValueError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    if not details:
        typer.echo(f"{estimate.total_gb:.1f} GB")
        return
    for part in estimate.parts:
        typer.echo(f"{part.label + ':':<24}{part.gb:>7.1f} GB")
        typer.echo(f"    {part.formula}")
    typer.echo(f"{'Total:':<24}{estimate.total_gb:>7.1f} GB")
    for note in estimate.notes:
        typer.echo(note)


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
def recipe_edit(handle: str = typer.Argument(..., show_default=False)) -> None:
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
        try:
            reopen = typer.confirm(
                "Reopen $EDITOR to fix it? (No reverts to the version from before this edit)",
                default=True,
            )
        except click.exceptions.Abort:
            reopen = False

        if reopen:
            click.edit(filename=str(path))
            continue

        path.write_text(original_text)
        typer.echo(f"reverted '{handle}' to its previous version")
        raise typer.Exit(code=1)


@recipe_app.command("remove")
def recipe_remove(
    handle: str = typer.Argument(..., show_default=False),
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

    typer.echo(
        f"default_gpu_memory_utilization: {config_file.get_default_gpu_memory_utilization()}"
    )


def _build_artifacts_using_image(image: str) -> list[Path]:
    """One path per affected recipe: a preinstall recipe's `Dockerfile`
    (its `compose.yaml` points at the local build tag, never the base
    image, so that's the file that actually needs the new value) or,
    for every other recipe, `compose.yaml` itself. No check against
    what the recipe pins - a recipe that happens to already pin exactly
    `image` gets offered the same update as one relying on the
    default; harmless either way, and not worth telling apart."""
    matches = []
    for handle in _recipe_store().list_handles():
        directory = config.recipe_dir(handle)
        dockerfile_path = directory / "Dockerfile"
        if dockerfile_path.is_file():
            first_line = dockerfile_path.read_text().split("\n", 1)[0]
            if first_line == f"FROM {image}":
                matches.append(dockerfile_path)
            continue
        compose_path = directory / "compose.yaml"
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


def _replace_image_in_file(path: Path, old_image: str, new_image: str) -> None:
    """A text substitution, not a regeneration - anything else in
    `path` survives untouched, hand-edited or not."""
    text = path.read_text()
    anchor = f"FROM {old_image}" if path.name == "Dockerfile" else f"image: {old_image}"
    replacement = f"FROM {new_image}" if path.name == "Dockerfile" else f"image: {new_image}"
    path.write_text(text.replace(anchor, replacement, 1))


@config_app.command("set-default-image")
def config_set_default_image(image: str = typer.Argument(..., show_default=False)) -> None:
    """Set the vLLM Docker image for recipes that do not define one.

    Pulls IMAGE first to confirm it actually resolves - the default is
    left unchanged if the pull fails. Only affects recipes that don't
    pin their own `image` - those keep using whatever they're pinned to
    either way. Existing `compose.yaml`/`Dockerfile` files aren't
    regenerated by this command; if any currently use the previous
    default image verbatim, offers to update just that one value in
    place, without touching anything else already there. A patched
    Dockerfile's local image is then rebuilt automatically (a plain
    `docker build`, not `recipe build`, so compose.yaml is never
    touched by this).
    """
    if image.endswith(":latest") or ":" not in image:
        typer.echo(
            "warning: using an unpinned image tag - pin it to a specific "
            "version once you've confirmed recipes work with it.",
            err=True,
        )

    typer.echo(f"pulling '{image}' to confirm it resolves ...")
    if _run_docker("pull", image) != 0:
        typer.echo("Failed to pull image. The default image was not changed.", err=True)
        raise typer.Exit(code=1)

    old_default = config_file.get_default_image()
    config_file.set_default_image(image)
    typer.echo(f"default image set to '{image}'")

    if old_default is None or old_default == image:
        return

    affected = _build_artifacts_using_image(old_default)
    if not affected:
        return

    typer.echo(f"{len(affected)} file(s) still use the previous default image:")
    for path in affected:
        typer.echo(f"  {path}")
    if not typer.confirm(
        "Replace it with the new image in these files? (a text replace - hand edits "
        "are kept as-is; an edited Dockerfile is then rebuilt via docker build)",
        default=False,
    ):
        return

    for path in affected:
        _replace_image_in_file(path, old_default, image)
    typer.echo(f"updated {len(affected)} file(s)")

    # A Dockerfile edit alone doesn't touch the local image it produces -
    # `docker build` alone, never `recipe build`, since that would also
    # regenerate compose.yaml and lose any hand edits there.
    for path in sorted(p for p in affected if p.name == "Dockerfile"):
        handle = path.parent.name
        tag = config.local_image_tag(handle)
        typer.echo(f"building '{tag}' from {path} ...")
        if _run_docker("build", "-t", tag, str(path.parent)) != 0:
            typer.echo(
                f"Failed to build '{tag}'. Run `fllame recipe build {handle}` to retry.",
                err=True,
            )


def _compose_files_using_gpu_memory_utilization(value: float) -> list[Path]:
    """compose.yaml files whose `--gpu-memory-utilization` bakes in
    `value` exactly - no check against what the recipe's own command
    sets; see `_build_artifacts_using_image` for why that's fine."""
    formatted = f"{value:.2f}"
    matches = []
    for handle in _recipe_store().list_handles():
        compose_path = config.recipe_dir(handle) / "compose.yaml"
        if not compose_path.is_file():
            continue
        try:
            existing = yaml.safe_load(compose_path.read_text())
            command = [str(token) for token in existing["services"][handle]["command"]]
        except (yaml.YAMLError, KeyError, TypeError):
            continue
        if extract_flag_value(command, "--gpu-memory-utilization") == formatted:
            matches.append(compose_path)
    return matches


def _replace_gpu_memory_utilization_in_compose_file(
    path: Path, old_value: float, new_value: float
) -> None:
    """A text substitution, not a regeneration - anything else in
    `path` survives untouched, hand-edited or not."""
    text = path.read_text()
    # `write_compose_file` always nests `command` two levels deep
    # (services -> handle -> command), and PyYAML's block sequences sit
    # at their key's own indentation rather than one level deeper - so
    # each item is a fixed 4 spaces in, not flush left.
    anchor = f"    - --gpu-memory-utilization\n    - '{old_value:.2f}'"
    replacement = f"    - --gpu-memory-utilization\n    - '{new_value:.2f}'"
    path.write_text(text.replace(anchor, replacement, 1))


@config_app.command("set-default-gpu-memory-utilization")
def config_set_default_gpu_memory_utilization(
    value: float = typer.Argument(..., show_default=False),
) -> None:
    """Set the `--gpu-memory-utilization` value for recipes that do not define one.

    An explicit `--gpu-memory-utilization` in a recipe's own command
    always wins over this default - fllame never overrides it. `serve`
    separately refuses to run any compose.yaml (regardless of where its
    value came from) whose --gpu-memory-utilization is missing or
    obviously invalid (<= 0 or > 1), since leaving vLLM unbounded can
    let it claim a unified-memory machine's entire memory pool.

    Existing `compose.yaml` files aren't regenerated by this command;
    if any currently bake in the previous default value verbatim,
    offers to update just that one value in place, without touching
    anything else already there.
    """
    if value <= 0 or value > 1.0:
        typer.echo(
            "default GPU memory utilization must be greater than 0 and no more than 1", err=True
        )
        raise typer.Exit(code=1)
    old_default = config_file.get_default_gpu_memory_utilization()
    config_file.set_default_gpu_memory_utilization(value)
    typer.echo(f"default GPU memory utilization set to {value}")

    if old_default == value:
        return

    affected = _compose_files_using_gpu_memory_utilization(old_default)
    if not affected:
        return

    typer.echo(f"{len(affected)} compose.yaml file(s) still use the previous default value:")
    for path in affected:
        typer.echo(f"  {path}")
    if not typer.confirm(
        "Replace it with the new value in these files? (a text replace - hand "
        "edits are kept as-is)",
        default=False,
    ):
        return

    for path in affected:
        _replace_gpu_memory_utilization_in_compose_file(path, old_default, value)
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
def model_pull(repo_id: str = typer.Argument(..., show_default=False)) -> None:
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
        show_default=False,
        help="Check only REPO_ID; omit to check every model currently in " "the local cache.",
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
        help=(
            "Search only this quantization. Without this option, this machine's "
            "supported quantizations are used."
        ),
    ),
    max_size: float | None = typer.Option(
        None,
        "--max-size",
        show_default=False,
        help=(
            "Maximum estimated VRAM usage, in GB (see the estimated VRAM column). "
            "Defaults to this machine's hardware scan budget."
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
            "and not applied unless given."
        ),
    ),
    limit: int = typer.Option(15, "--limit", help="Number of ranked results to show."),
) -> None:
    """Search the Hugging Face Hub for models, ranked by parameter count,
    total downloads and 30-day downloads.

    Parameter counts come from the repo name (e.g. 27B), falling back to the
    Hub's own count.

    Estimated VRAM assumes one request at a time with --max-model-len 32768:
    weights estimated from the parameter count and quantization, plus 8 KB of
    KV cache per token per billion parameters, plus 2 GB runtime overhead.

    Real VRAM use depends on the recipe, --max-model-len above all: if it isn't
    set, vLLM uses the model's maximum context length, which can need far more.

    Each concurrent request needs its own KV cache, adding about 0.25 GB per
    billion parameters at 32K context.
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
    headers += ["DL TOTAL", "DL 30D", "UPDATED"]
    rows = [
        [
            c.repo_id,
            *([c.quantization] if show_quant_column else []),
            f"{c.params_billion:.1f}B" if c.params_billion is not None else "unknown",
            f"{c.estimated_vram_gb:.1f} GB" if c.estimated_vram_gb is not None else "unknown",
            _format_count(c.downloads_all_time),
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
    recipe: Recipe, *, assume_yes: bool, default_gpu_memory_utilization: float
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
        default_gpu_memory_utilization=default_gpu_memory_utilization,
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


def _require_safe_gpu_memory_utilization(recipe: Recipe) -> None:
    compose_path = config.recipe_dir(recipe.handle) / "compose.yaml"
    try:
        compose = yaml.safe_load(compose_path.read_text())
        service = compose["services"][recipe.handle]
    except (yaml.YAMLError, KeyError, TypeError, AttributeError, OSError):
        # compose.yaml exists (_require_compose_built already checked)
        # but isn't in a shape this can even inspect - let the actual
        # `docker compose up` call surface whatever is wrong with it.
        return

    error = validate_gpu_memory_utilization(service)
    if error is not None:
        typer.echo(error, err=True)
        raise typer.Exit(code=1)


@app.command()
def serve(
    handle: str = typer.Argument(..., show_default=False),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip this command's confirmation prompts (the VRAM sanity check) - "
        "any warning is still printed.",
    ),
) -> None:
    """Launch HANDLE's recipe via `docker compose up -d`. Reads whatever
    `compose.yaml` is already on disk - never regenerates it - and never
    touches the network."""
    recipe = _load_or_exit(handle)
    _require_model_cached(recipe)
    _require_compose_built(recipe)
    _require_safe_gpu_memory_utilization(recipe)
    _warn_if_vram_likely_insufficient(recipe, assume_yes=yes)

    code = _run_compose(handle, "up", "-d", handle)
    if code == 0:
        container_name = config.compose_project_name(handle)
        typer.echo(f"'{handle}' started - follow its logs with: docker logs -f {container_name}")
        typer.echo(
            "model loading can take several minutes - an empty or quiet log right "
            "after this returns is expected, not a problem."
        )
    raise typer.Exit(code=code)


def _exec_in_container(handle: str, *command: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["docker", *_compose_args(handle, "exec", "-T", handle, *command)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        typer.echo(_DOCKER_NOT_FOUND_MESSAGE, err=True)
        raise typer.Exit(code=1) from e


# Run with the container's own python: the vLLM image is not guaranteed to ship curl.
_PROBE_SCRIPT = """
import sys, urllib.request
base = sys.argv[1]
urllib.request.urlopen(base + "/health", timeout=5)
print(urllib.request.urlopen(base + "/v1/models", timeout=5).read().decode())
"""


def _require_running(handle: str) -> None:
    containers, code = _compose_ps_json(handle)
    if code != 0:
        raise typer.Exit(code=code)
    if any(c.get("State") == "running" for c in containers if c.get("Service", handle) == handle):
        return
    typer.echo(
        f"'{handle}' is not running - start it first with `fllame serve {handle}`.", err=True
    )
    raise typer.Exit(code=1)


def _served_command(handle: str) -> list[str]:
    """What the container actually runs, from compose.yaml - it may have
    been hand-edited away from recipe.yaml."""
    try:
        compose = yaml.safe_load((config.recipe_dir(handle) / "compose.yaml").read_text())
        return [str(token) for token in compose["services"][handle].get("command") or []]
    except (yaml.YAMLError, KeyError, TypeError, AttributeError, OSError):
        return []


def _probe_served_model(handle: str, base_url: str) -> dict:
    result = _exec_in_container(handle, "python3", "-c", _PROBE_SCRIPT, base_url)
    try:
        return json.loads(result.stdout)["data"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass
    typer.echo(
        f"'{handle}' is running, but vLLM isn't answering at {base_url} yet - the model is "
        f"probably still loading. Follow it with: docker logs -f "
        f"{config.compose_project_name(handle)}",
        err=True,
    )
    raise typer.Exit(code=1)


_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _run_with_spinner(label: str, run) -> subprocess.CompletedProcess:
    spin = sys.stdout.isatty()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run)
        frame = 0
        while not future.done():
            if spin:
                elapsed = int(time.monotonic() - started)
                sys.stdout.write(
                    f"\r{_SPINNER_FRAMES[frame % len(_SPINNER_FRAMES)]} {label} "
                    f"{elapsed // 60}m{elapsed % 60:02d}s"
                )
                sys.stdout.flush()
                frame += 1
            time.sleep(0.1)
    if spin:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
    return future.result()


def _save_bench_context(
    run_dir: Path, recipe: Recipe, params: dict, *, vllm_version: str | None
) -> None:
    recipe_dir = config.recipe_dir(recipe.handle)
    for name in ("recipe.yaml", "compose.yaml", "Dockerfile"):
        if (recipe_dir / name).is_file():
            shutil.copy2(recipe_dir / name, run_dir / name)
    settings = {
        "default_image": config_file.get_default_image() or _FALLBACK_IMAGE,
        "default_gpu_memory_utilization": config_file.get_default_gpu_memory_utilization(),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    (run_dir / "params.yaml").write_text(
        yaml.safe_dump({**params, "vllm_version": vllm_version}, sort_keys=False)
    )


@app.command(
    help=(
        "Benchmark HANDLE's running container with `vllm bench serve` at several "
        "concurrency levels, on random prompts - a speed test, not a quality test. "
        "HANDLE must already be up via `fllame serve`.\n\n"
        "Every run is saved to HANDLE's bench/<timestamp>/ folder, with the result JSON "
        "per level plus the recipe, compose.yaml, Dockerfile, config and parameters used.\n\n"
        "Columns:\n\n\b\n" + columns_help()
    )
)
def bench(
    handle: str = typer.Argument(..., show_default=False),
    concurrency: str = typer.Option(
        ",".join(str(c) for c in DEFAULT_CONCURRENCY),
        "--concurrency",
        help="Comma-separated concurrency levels.",
    ),
    num_prompts: str | None = typer.Option(
        None,
        "--num-prompts",
        show_default=False,
        help="Comma-separated requests per level, each a multiple of its concurrency "
        "[default: at least 10 requests and 2 full waves].",
    ),
    input_len: int = typer.Option(DEFAULT_INPUT_LEN, "--input-len", help="Prompt tokens."),
    output_len: int = typer.Option(DEFAULT_OUTPUT_LEN, "--output-len", help="Generated tokens."),
) -> None:
    recipe = _load_or_exit(handle)
    _require_compose_built(recipe)
    try:
        levels = build_levels(
            parse_int_list("--concurrency", concurrency),
            parse_int_list("--num-prompts", num_prompts) if num_prompts is not None else None,
        )
    except SweepError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    _require_running(handle)
    port = extract_flag_value(_served_command(handle), "--port") or "8000"
    base_url = f"http://localhost:{port}"
    served = _probe_served_model(handle, base_url)
    model = served["id"]
    # A --served-model-name isn't a repo id, so bench can't load the tokenizer by it.
    root = served.get("root")
    tokenizer = root if root and root != model else None

    max_model_len = served.get("max_model_len")
    if isinstance(max_model_len, int) and input_len + output_len > max_model_len:
        typer.echo(
            f"--input-len {input_len} + --output-len {output_len} exceeds '{handle}''s "
            f"max model length of {max_model_len} tokens - lower either.",
            err=True,
        )
        raise typer.Exit(code=1)

    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    run_dir = config.recipe_dir(handle) / "bench" / stamp
    run_dir.mkdir(parents=True)
    container_dir = f"/tmp/fllame-bench-{stamp}"

    def command_for(level: Level) -> list[str]:
        return bench_command(
            model=model,
            tokenizer=tokenizer,
            base_url=base_url,
            level=level,
            input_len=input_len,
            output_len=output_len,
            result_dir=container_dir,
            result_filename=f"c{level.concurrency}.json",
        )

    version = _exec_in_container(handle, "vllm", "--version")
    vllm_version = version.stdout.strip() if version.returncode == 0 else ""
    params = {
        "handle": handle,
        "started_at": started.isoformat(),
        "model": model,
        "base_url": base_url,
        "input_len": input_len,
        "output_len": output_len,
        "levels": [dataclasses.asdict(level) for level in levels],
        "commands": [shlex.join(command_for(level)) for level in levels],
    }
    _save_bench_context(
        run_dir,
        recipe,
        params,
        vllm_version=vllm_version or None,
    )

    widths = column_widths()
    lines = [header_row(widths)]
    typer.echo(lines[0])
    try:
        for level in levels:
            result = _run_with_spinner(
                f"concurrency {level.concurrency}, {level.num_prompts} requests",
                lambda level=level: _exec_in_container(handle, *command_for(level)),
            )
            if result.returncode != 0:
                typer.echo((result.stdout + result.stderr).rstrip(), err=True)
                typer.echo(
                    f"`vllm bench serve` failed at concurrency {level.concurrency} - see "
                    f"its output above. Levels finished so far are in {run_dir}",
                    err=True,
                )
                raise typer.Exit(code=1)
            saved = _exec_in_container(handle, "cat", f"{container_dir}/c{level.concurrency}.json")
            try:
                data = json.loads(saved.stdout)
            except json.JSONDecodeError:
                data = {}
            (run_dir / f"c{level.concurrency}.json").write_text(json.dumps(data, indent=2))
            lines.append(format_row(result_cells(level, data), widths))
            typer.echo(lines[-1])
    finally:
        _exec_in_container(handle, "rm", "-rf", container_dir)
        (run_dir / "results.txt").write_text("\n".join(lines) + "\n")

    typer.echo(f"\nsaved to {run_dir}")


def _status_row_from_recipe(handle: str, recipe: Recipe) -> list[str]:
    """A recipe with no `compose.yaml` yet - there's nothing built to
    read, so every column but STATUS comes straight from recipe.yaml."""
    return [
        handle,
        config.compose_project_name(handle),
        _resolve_image(recipe).image,
        "Not built",
        f"{recipe.port}:{recipe.port}",
    ]


def _status_row_from_compose(
    handle: str, recipe: Recipe, compose_path: Path, *, status: str = "Never started"
) -> list[str]:
    """A built recipe with no confirmed container (never `serve`d, or
    `docker compose ps` itself failed) - compose.yaml's own `image`
    (the local build tag for a preinstall recipe, same as `serve`
    would actually run) stands in for docker's."""
    try:
        existing = yaml.safe_load(compose_path.read_text())
        image = existing["services"][handle]["image"]
    except (yaml.YAMLError, KeyError, TypeError):
        image = _resolve_image(recipe).image
    return [
        handle,
        config.compose_project_name(handle),
        image,
        status,
        f"{recipe.port}:{recipe.port}",
    ]


@app.command()
def status() -> None:
    """Show every recipe in one table - built or not, running or not.
    A container's own row comes from `docker compose ps --all --format
    json` (each recipe is still its own compose project); everything
    else is read straight from recipe.yaml/compose.yaml. Never
    regenerates compose.yaml."""
    store = _recipe_store()
    handles = store.list_handles()
    if not handles:
        typer.echo(f"No recipes found in {config.recipes_dir()}")
        raise typer.Exit(code=0)

    rows = []
    exit_code = 0
    for handle in handles:
        try:
            recipe = store.load(handle)
        except RecipeError as e:
            typer.echo(f"warning: {e}", err=True)
            continue

        compose_path = config.recipe_dir(handle) / "compose.yaml"
        if not compose_path.is_file():
            rows.append(_status_row_from_recipe(handle, recipe))
            continue

        containers, code = _compose_ps_json(handle)
        if code != 0:
            exit_code = code
            rows.append(_status_row_from_compose(handle, recipe, compose_path, status="Unknown"))
            continue
        if not containers:
            rows.append(_status_row_from_compose(handle, recipe, compose_path))
            continue
        for container in containers:
            # A stopped container reports no live port bindings at all
            # - fall back to the configured port rather than leaving
            # this blank.
            ports = _format_ports(container.get("Publishers") or [])
            rows.append(
                [
                    handle,
                    container.get("Name", ""),
                    container.get("Image", ""),
                    container.get("Status", ""),
                    ports or f"{recipe.port}:{recipe.port}",
                ]
            )

    if rows:
        _print_table(["RECIPE", "NAME", "IMAGE", "STATUS", "PORTS"], rows)
    raise typer.Exit(code=exit_code)


@app.command()
def stop(handle: str = typer.Argument(..., show_default=False)) -> None:
    """Stop HANDLE's container via `docker compose stop`. Reads
    whatever `compose.yaml` is already on disk - never regenerates it."""
    recipe = _load_or_exit(handle)
    _require_compose_built(recipe)
    raise typer.Exit(code=_run_compose(handle, "stop", handle))


if __name__ == "__main__":
    app()
