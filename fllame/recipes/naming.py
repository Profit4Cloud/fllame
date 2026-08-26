"""Derives a recipe handle from a repo_id: the part after the last `/`
(the model name, not the org/provider), lowercased and slugified.
Purely a starting point - `RecipeStore.next_available_handle` is what
actually guarantees uniqueness against what's on file.
"""

from __future__ import annotations

import re

_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")


def derive_handle(repo_id: str) -> str:
    name = repo_id.rsplit("/", 1)[-1]
    slug = _NON_SLUG_CHARS.sub("-", name.lower()).strip("-")
    return slug or "model"
