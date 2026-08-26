from fllame.models import puller


def test_pull_model_calls_snapshot_download(monkeypatch):
    captured = {}

    def fake_snapshot_download(repo_id, local_files_only=False):
        captured["repo_id"] = repo_id
        captured["local_files_only"] = local_files_only
        return "/cache/models--org--demo/snapshots/abc"

    monkeypatch.setattr(puller, "snapshot_download", fake_snapshot_download)

    path = puller.pull_model("org/demo")

    assert captured["repo_id"] == "org/demo"
    assert captured["local_files_only"] is False
    assert path == "/cache/models--org--demo/snapshots/abc"


def test_pull_model_offline_forces_local_files_only(monkeypatch):
    captured = {}

    def fake_snapshot_download(repo_id, local_files_only=False):
        captured["local_files_only"] = local_files_only
        return "/cache/models--org--demo/snapshots/abc"

    monkeypatch.setattr(puller, "snapshot_download", fake_snapshot_download)

    puller.pull_model("org/demo", offline=True)

    assert captured["local_files_only"] is True
