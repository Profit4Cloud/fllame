"""Filesystem locations fllame reads and writes, overridable via env vars."""

from __future__ import annotations

import os
from pathlib import Path


def recipes_dir() -> Path:
    override = os.environ.get("FLLAME_RECIPES_DIR")
    if override:
        return Path(override)
    return Path.home() / ".config" / "fllame" / "recipes"


def state_db_path() -> Path:
    override = os.environ.get("FLLAME_STATE_DB")
    if override:
        return Path(override)
    return Path.home() / ".local" / "state" / "fllame" / "state.db"
