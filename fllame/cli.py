"""fllame - a headless CLI for vLLM serving.

`fllame serve <handle>` resolves a hand-edited recipe, guarantees the
model is fully downloaded, and runs it as a Docker container via
`docker compose`. See README.md for the recipe file format and CLAUDE.md
for the architecture this sits on.
"""

from __future__ import annotations

import subprocess

import typer
import yaml
from huggingface_hub.errors import LocalEntryNotFoundError

from fllame import config
from fllame.backends.vllm import VllmServingBackend
from fllame.compose.generator import generate_compose, write_compose_file
from fllame.domain.recipe import Recipe, RecipeError
from fllame.hardware.scanner import scan_hardware
from fllame.models.cache import list_cached_models
from fllame.models.puller import pull_model
from fllame.recipes.store import RecipeStore

app = typer.Typer(no_args_is_help=True, add_completion=False)
recipe_app = typer.Typer(no_args_is_help=True)
app.add_typer(recipe_app, name="recipe", help="Inspect the recipe registry.")
hardware_app = typer.Typer(no_args_is_help=True)
app.add_typer(hardware_app, name="hardware", help="Detect this machine's GPU/RAM.")
model_app = typer.Typer(no_args_is_help=True)
app.add_typer(model_app, name="model", help="Download and inspect the local HF model cache.")

BACKEND = VllmServingBackend()


def _recipe_store() -> RecipeStore:
    return RecipeStore(config.recipes_dir())


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
            recipes.append(store.load(handle))
        except RecipeError as e:
            typer.echo(f"warning: {e}", err=True)
    compose = generate_compose(recipes, backend=BACKEND, hf_cache_dir=config.hf_cache_dir())
    if offline_handle is not None and offline_handle in compose["services"]:
        compose["services"][offline_handle]["environment"]["HF_HUB_OFFLINE"] = "1"
    write_compose_file(compose, config.compose_file_path())


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
    """Print the resolved docker-compose service for HANDLE."""
    recipe = _load_or_exit(handle)
    service = BACKEND.build_service(recipe, hf_cache_dir=config.hf_cache_dir())
    typer.echo(yaml.safe_dump({recipe.handle: service}, sort_keys=False).rstrip())


@hardware_app.command("scan")
def hardware_scan() -> None:
    """Detect GPU(s), VRAM, and RAM on this machine."""
    profile = scan_hardware()
    if not profile.has_gpu:
        typer.echo("gpu:            none detected (no nvidia-smi on PATH)")
    else:
        typer.echo(f"gpu:            {profile.gpu_name} x{profile.gpu_count}")
        typer.echo(f"vram_per_gpu:   {profile.vram_gb_per_gpu:.1f} GB")
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
    for repo in models:
        typer.echo(f"{repo.repo_id}\t{repo.size_on_disk_str}\t{repo.last_modified_str}")


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
