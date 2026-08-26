from huggingface_hub.errors import CacheNotFound

from fllame.models import cache


class _FakeRepo:
    def __init__(self, repo_id: str, repo_type: str):
        self.repo_id = repo_id
        self.repo_type = repo_type


class _FakeCacheInfo:
    def __init__(self, repos):
        self.repos = repos


def test_list_cached_models_filters_to_models_only_and_sorts(monkeypatch):
    fake_info = _FakeCacheInfo(
        [
            _FakeRepo("org/b-model", "model"),
            _FakeRepo("org/a-dataset", "dataset"),
            _FakeRepo("org/a-model", "model"),
        ]
    )
    monkeypatch.setattr(cache, "scan_cache_dir", lambda: fake_info)

    models = cache.list_cached_models()

    assert [m.repo_id for m in models] == ["org/a-model", "org/b-model"]


def test_list_cached_models_empty(monkeypatch):
    monkeypatch.setattr(cache, "scan_cache_dir", lambda: _FakeCacheInfo([]))

    assert cache.list_cached_models() == []


def test_list_cached_models_cache_dir_never_created(monkeypatch):
    def raise_not_found():
        raise CacheNotFound("no cache yet", cache_dir="/root/.cache/huggingface/hub")

    monkeypatch.setattr(cache, "scan_cache_dir", raise_not_found)

    assert cache.list_cached_models() == []
