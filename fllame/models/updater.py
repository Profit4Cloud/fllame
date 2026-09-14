"""Checks whether a cached model has a newer revision on the Hub."""

from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import model_info

from fllame.models.cache import cached_revision_hash


@dataclass(frozen=True)
class UpdateStatus:
    repo_id: str
    # None: never pulled - not the same as "up to date."
    cached_revision: str | None
    latest_revision: str

    @property
    def is_stale(self) -> bool:
        return self.cached_revision != self.latest_revision


def check_for_update(repo_id: str) -> UpdateStatus:
    """A Hub failure here is left to raise, not swallowed - `model
    update` is a direct-purpose command a caller explicitly ran."""
    return UpdateStatus(
        repo_id=repo_id,
        cached_revision=cached_revision_hash(repo_id),
        latest_revision=model_info(repo_id).sha,
    )
