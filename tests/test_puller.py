from fllame.models import puller


def test_pull_model_calls_snapshot_download(monkeypatch):
    captured = {}

    def fake_snapshot_download(repo_id):
        captured["repo_id"] = repo_id
        return "/cache/models--org--demo/snapshots/abc"

    monkeypatch.setattr(puller, "snapshot_download", fake_snapshot_download)

    path = puller.pull_model("org/demo")

    assert captured["repo_id"] == "org/demo"
    assert path == "/cache/models--org--demo/snapshots/abc"
