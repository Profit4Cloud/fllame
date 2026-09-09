"""Filesystem locations fllame reads and writes, overridable via env vars."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import constants as hf_constants


def recipes_dir() -> Path:
    override = os.environ.get("FLLAME_RECIPES_DIR")
    if override:
        return Path(override)
    return Path.home() / ".config" / "fllame" / "recipes"


def recipe_dir(handle: str) -> Path:
    """HANDLE's own folder under `recipes_dir()` - holds both its
    hand-edited `recipe.yaml` (`RecipeStore`) and its generated,
    fllame-owned `compose.yaml` (written by `serve`/`status`/`stop`/
    `recipe build`), side by side. Keeping both in one folder means the
    compose file sits exactly where an operator would look for it - next
    to the recipe it came from, in the same git-tracked directory - and
    can be copied/driven elsewhere with plain `docker compose up -d`, no
    fllame CLI involved.
    """
    return recipes_dir() / handle


def compose_project_name(handle: str) -> str:
    """The docker-compose project name for HANDLE's recipe - each
    recipe is its own independent compose project (see `recipe_dir`),
    not one project shared across every recipe.
    """
    return f"fllame-{handle}"


def config_file_path() -> Path:
    """A small persisted settings file for cross-invocation CLI defaults
    that aren't per-recipe data (`recipes_dir`) - today just the
    fallback Docker image (see `fllame/config_file.py`, `fllame
    config`). Read/written directly by that module, not hand-edited.
    """
    override = os.environ.get("FLLAME_CONFIG_FILE")
    if override:
        return Path(override)
    return Path.home() / ".config" / "fllame" / "config.yaml"


def hf_cache_dir() -> Path:
    """Wherever huggingface_hub itself resolves its cache to (honoring
    HF_HOME/HF_HUB_CACHE) - fllame deliberately has no opinion of its own
    about where models live on disk.
    """
    return Path(hf_constants.HF_HUB_CACHE)
