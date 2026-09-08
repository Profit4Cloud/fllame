"""Filesystem locations fllame reads and writes, overridable via env vars."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import constants as hf_constants

COMPOSE_PROJECT_NAME = "fllame"


def recipes_dir() -> Path:
    override = os.environ.get("FLLAME_RECIPES_DIR")
    if override:
        return Path(override)
    return Path.home() / ".config" / "fllame" / "recipes"


def state_dir() -> Path:
    override = os.environ.get("FLLAME_STATE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".local" / "state" / "fllame"


def compose_file_path() -> Path:
    """The docker-compose file fllame generates and owns - see
    `fllame/compose/generator.py`. Not something an operator hand-edits.
    """
    return state_dir() / "docker-compose.yml"


def config_file_path() -> Path:
    """A small persisted settings file for cross-invocation CLI defaults
    that aren't per-recipe data (`recipes_dir`) or generated container
    state (`state_dir`) - today just the fallback Docker image (see
    `fllame/config_file.py`, `fllame config`). Read/written directly by
    that module, not hand-edited.
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
