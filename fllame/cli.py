"""fllame - a headless CLI for vLLM serving.

`fllame serve <handle>` resolves a hand-edited recipe (model repo id +
vLLM args) and launches `vllm serve`. See README.md for the recipe file
format and CLAUDE.md for the architecture this sits on.
"""

from __future__ import annotations

import os
import signal
import subprocess

import typer

from fllame import config
from fllame.backends.vllm import VllmServingBackend
from fllame.domain.recipe import RecipeError
from fllame.hardware.scanner import scan_hardware
from fllame.recipes.store import RecipeStore
from fllame.state.store import StateStore

app = typer.Typer(no_args_is_help=True, add_completion=False)
recipe_app = typer.Typer(no_args_is_help=True)
app.add_typer(recipe_app, name="recipe", help="Inspect the recipe registry.")
hardware_app = typer.Typer(no_args_is_help=True)
app.add_typer(hardware_app, name="hardware", help="Detect this machine's GPU/RAM.")

BACKEND = VllmServingBackend()


def _recipe_store() -> RecipeStore:
    return RecipeStore(config.recipes_dir())


def _state_store() -> StateStore:
    return StateStore(config.state_db_path())


def _load_or_exit(handle: str):
    try:
        return _recipe_store().load(handle)
    except RecipeError as e:
        typer.echo(str(e), err=True)
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
    """Print the resolved recipe for HANDLE."""
    recipe = _load_or_exit(handle)
    argv = BACKEND.build_argv(recipe, port=recipe.port)
    typer.echo(f"handle:       {recipe.handle}")
    typer.echo(f"backend:      {recipe.backend}")
    typer.echo(f"repo_id:      {recipe.repo_id}")
    typer.echo(f"port:         {recipe.port}")
    typer.echo(f"env:          {recipe.env or '{}'}")
    typer.echo(f"command:      {' '.join(argv)}")


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


@app.command()
def serve(
    handle: str,
    port: int | None = typer.Option(None, help="Override the recipe's default port."),
    detach: bool = typer.Option(False, "--detach", "-d", help="Run in the background."),
) -> None:
    """Launch the recipe for HANDLE with vLLM."""
    recipe = _load_or_exit(handle)

    resolved_port = port if port is not None else recipe.port
    argv = BACKEND.build_argv(recipe, port=resolved_port)
    env = {**os.environ, **recipe.env}

    if not detach:
        os.execvpe(argv[0], argv, env)  # replaces this process; never returns

    process = subprocess.Popen(argv, env=env, start_new_session=True)
    _state_store().record_started(handle, process.pid, resolved_port, argv)
    typer.echo(f"started '{handle}' (pid {process.pid}, port {resolved_port})")


@app.command()
def status() -> None:
    """List servers fllame started in the background."""
    servers = _state_store().list_all()
    if not servers:
        typer.echo("No servers tracked.")
        raise typer.Exit(code=0)
    for s in servers:
        state = "running" if s.is_alive() else "not running (stale)"
        typer.echo(f"{s.handle}\tpid={s.pid}\tport={s.port}\t{state}\tstarted={s.started_at}")


@app.command()
def stop(handle: str) -> None:
    """Stop a server fllame started in the background."""
    store = _state_store()
    server = store.get(handle)
    if server is None:
        typer.echo(f"'{handle}' is not tracked as running.", err=True)
        raise typer.Exit(code=1)
    if server.is_alive():
        os.kill(server.pid, signal.SIGTERM)
        typer.echo(f"sent SIGTERM to '{handle}' (pid {server.pid})")
    store.remove(handle)


if __name__ == "__main__":
    app()
