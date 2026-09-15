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
    """HANDLE's folder - holds both `recipe.yaml` and its generated
    `compose.yaml`, side by side."""
    return recipes_dir() / handle


def compose_project_name(handle: str) -> str:
    return f"fllame-{handle}"


def local_image_tag(handle: str) -> str:
    """The local image tag `recipe build` builds and tags a preinstall
    recipe's Dockerfile under - retagged fresh on every build, not
    content-hash-derived; that level of image/version management is out
    of scope for now."""
    return f"fllame-{handle}:latest"


def config_file_path() -> Path:
    override = os.environ.get("FLLAME_CONFIG_FILE")
    if override:
        return Path(override)
    return Path.home() / ".config" / "fllame" / "config.yaml"


def hf_cache_dir() -> Path:
    """Wherever huggingface_hub itself resolves HF_HOME/HF_HUB_CACHE
    to - fllame has no opinion of its own."""
    return Path(hf_constants.HF_HUB_CACHE)
