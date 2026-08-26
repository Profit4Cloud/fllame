import os
from pathlib import Path

from fllame.state.store import StateStore


def test_record_and_list(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")

    store.record_started("demo", pid=os.getpid(), port=8000, argv=["vllm", "serve"])
    servers = store.list_all()

    assert len(servers) == 1
    assert servers[0].handle == "demo"
    assert servers[0].is_alive() is True


def test_remove(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    store.record_started("demo", pid=os.getpid(), port=8000, argv=["vllm"])

    store.remove("demo")

    assert store.get("demo") is None


def test_dead_pid_reported_not_alive(tmp_path: Path, monkeypatch):
    store = StateStore(tmp_path / "state.db")
    store.record_started("demo", pid=12345, port=8000, argv=["vllm"])

    def fake_kill(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(os, "kill", fake_kill)

    assert store.get("demo").is_alive() is False
