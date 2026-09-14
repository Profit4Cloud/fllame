import pytest

from fllame.models import updater
from fllame.models.updater import UpdateStatus, check_for_update


class _FakeModelInfo:
    def __init__(self, sha: str):
        self.sha = sha


def test_check_for_update_detects_stale(monkeypatch):
    monkeypatch.setattr(updater, "cached_revision_hash", lambda repo_id: "abc123")
    monkeypatch.setattr(updater, "model_info", lambda repo_id: _FakeModelInfo("def456"))

    status = check_for_update("org/demo")

    assert status == UpdateStatus(
        repo_id="org/demo", cached_revision="abc123", latest_revision="def456"
    )
    assert status.is_stale is True


def test_check_for_update_up_to_date_when_hashes_match(monkeypatch):
    monkeypatch.setattr(updater, "cached_revision_hash", lambda repo_id: "abc123")
    monkeypatch.setattr(updater, "model_info", lambda repo_id: _FakeModelInfo("abc123"))

    status = check_for_update("org/demo")

    assert status.is_stale is False


def test_check_for_update_never_pulled_counts_as_stale(monkeypatch):
    """`cached_revision=None` (never pulled) still compares unequal to
    any real latest_revision - `is_stale` doesn't special-case it, since
    the caller (cli.py's `model update`) checks `cached_revision is
    None` separately to report "not cached" rather than "stale"."""
    monkeypatch.setattr(updater, "cached_revision_hash", lambda repo_id: None)
    monkeypatch.setattr(updater, "model_info", lambda repo_id: _FakeModelInfo("def456"))

    status = check_for_update("org/never-pulled")

    assert status.cached_revision is None
    assert status.is_stale is True


def test_check_for_update_propagates_hub_errors(monkeypatch):
    def raise_error(repo_id):
        raise RuntimeError("boom")

    monkeypatch.setattr(updater, "cached_revision_hash", lambda repo_id: "abc123")
    monkeypatch.setattr(updater, "model_info", raise_error)

    with pytest.raises(RuntimeError):
        check_for_update("org/demo")
