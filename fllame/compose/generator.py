"""Compiles the recipe registry into a single docker-compose file - a
generated artifact fllame fully owns and overwrites on every command that
needs it, not something an operator hand-edits. It currently holds only
fllame-managed services; a hand-written compose file that adds sibling
services (Grafana, OpenWebUI) alongside it via Compose's `include:` is a
natural next step, not built yet - see CLAUDE.md, "Explicitly deferred".
"""

from __future__ import annotations

from pathlib import Path

import yaml

from fllame.backends.base import ServingBackend
from fllame.domain.recipe import Recipe


def generate_compose(recipes: list[Recipe], *, backend: ServingBackend, hf_cache_dir: Path) -> dict:
    return {
        "services": {
            recipe.handle: backend.build_service(recipe, hf_cache_dir=hf_cache_dir)
            for recipe in recipes
        }
    }


def write_compose_file(compose: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(compose, sort_keys=False))
