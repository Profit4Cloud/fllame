"""Checks whether a cached model has a newer revision on the Hub -
`fllame model update`'s data source, and the *only* place in fllame
that ever asks "is my cache stale," since `serve` itself never touches
the network at all (see CLAUDE.md, "Setup vs. running").

`check_for_update` combines `models/cache.py`'s local, network-free
`cached_revision_hash` with one Hub call (`huggingface_hub.model_info`)
for the same repo_id's current commit hash. Deliberately two separate
reads rather than a single richer call: the local half already exists
for other reasons (`is_model_cached`), and keeping the network half
isolated here means `models/cache.py` stays entirely network-free, true
to its own docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import model_info

from fllame.models.cache import cached_revision_hash


@dataclass(frozen=True)
class UpdateStatus:
    repo_id: str
    # `None` means "never pulled" - there's nothing to compare against,
    # not the same as "up to date."
    cached_revision: str | None
    latest_revision: str

    @property
    def is_stale(self) -> bool:
        return self.cached_revision != self.latest_revision


def check_for_update(repo_id: str) -> UpdateStatus:
    """`latest_revision` comes straight from `huggingface_hub.model_info`
    - a real network call, left to raise (`HfHubHTTPError`/
    `RequestException`) rather than swallowed, since `model update` is a
    direct-purpose command a caller explicitly ran, not a best-effort
    background check where "unknown" should stay silent (contrast
    `models/discovery.py`'s Hub lookups, or `cli.py`'s VRAM sanity
    check).
    """
    return UpdateStatus(
        repo_id=repo_id,
        cached_revision=cached_revision_hash(repo_id),
        latest_revision=model_info(repo_id).sha,
    )
