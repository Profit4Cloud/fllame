"""Not unique on its own - `RecipeStore.next_available_handle` handles
collisions."""

from __future__ import annotations

import re

_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")


def derive_handle(repo_id: str) -> str:
    name = repo_id.rsplit("/", 1)[-1]
    slug = _NON_SLUG_CHARS.sub("-", name.lower()).strip("-")
    return slug or "model"
