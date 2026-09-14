from huggingface_hub.errors import CacheNotFound

from fllame.models import cache


class _FakeRepo:
    def __init__(self, repo_id: str, repo_type: str, revisions=None):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revisions = revisions if revisions is not None else []


class _FakeCacheInfo:
    def __init__(self, repos):
        self.repos = repos


class _FakeFile:
    def __init__(self, file_name: str, size_on_disk: int):
        self.file_name = file_name
        self.size_on_disk = size_on_disk


class _FakeRevision:
    def __init__(self, files, last_modified: float = 0, commit_hash: str = "deadbeef"):
        self.files = files
        self.last_modified = last_modified
        self.commit_hash = commit_hash


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


def test_is_model_cached_true_when_repo_has_a_revision(monkeypatch):
    revision = _FakeRevision(files=[_FakeFile("model.safetensors", 1024)])
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "model", revisions=[revision])])
    monkeypatch.setattr(cache, "scan_cache_dir", lambda: fake_info)

    assert cache.is_model_cached("org/demo") is True


def test_is_model_cached_false_when_repo_never_pulled(monkeypatch):
    monkeypatch.setattr(cache, "scan_cache_dir", lambda: _FakeCacheInfo([]))

    assert cache.is_model_cached("org/never-pulled") is False


def test_is_model_cached_false_when_repo_present_but_no_revisions(monkeypatch):
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "model", revisions=[])])
    monkeypatch.setattr(cache, "scan_cache_dir", lambda: fake_info)

    assert cache.is_model_cached("org/demo") is False


def test_is_model_cached_false_when_cache_dir_never_created(monkeypatch):
    def raise_not_found():
        raise CacheNotFound("no cache yet", cache_dir="/root/.cache/huggingface/hub")

    monkeypatch.setattr(cache, "scan_cache_dir", raise_not_found)

    assert cache.is_model_cached("org/demo") is False


def test_is_model_cached_ignores_non_model_repo_type(monkeypatch):
    revision = _FakeRevision(files=[_FakeFile("data.parquet", 1024)])
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "dataset", revisions=[revision])])
    monkeypatch.setattr(cache, "scan_cache_dir", lambda: fake_info)

    assert cache.is_model_cached("org/demo") is False


def test_cached_revision_hash_returns_most_recent_revisions_hash(monkeypatch):
    stale = _FakeRevision(files=[], last_modified=1, commit_hash="stale-hash")
    current = _FakeRevision(files=[], last_modified=2, commit_hash="current-hash")
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "model", revisions=[stale, current])])
    monkeypatch.setattr(cache, "scan_cache_dir", lambda: fake_info)

    assert cache.cached_revision_hash("org/demo") == "current-hash"


def test_cached_revision_hash_none_when_repo_not_cached(monkeypatch):
    monkeypatch.setattr(cache, "scan_cache_dir", lambda: _FakeCacheInfo([]))

    assert cache.cached_revision_hash("org/never-pulled") is None


def test_cached_revision_hash_none_when_cache_dir_never_created(monkeypatch):
    def raise_not_found():
        raise CacheNotFound("no cache yet", cache_dir="/root/.cache/huggingface/hub")

    monkeypatch.setattr(cache, "scan_cache_dir", raise_not_found)

    assert cache.cached_revision_hash("org/demo") is None


def test_local_estimate_vram_gb_sums_only_safetensors_files(monkeypatch):
    revision = _FakeRevision(
        files=[
            _FakeFile("model-00001-of-00002.safetensors", round(3 * 1024**3)),
            _FakeFile("model-00002-of-00002.safetensors", round(2 * 1024**3)),
            _FakeFile("config.json", 2_000),
            _FakeFile("tokenizer.json", 500_000),
        ]
    )
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "model", revisions=[revision])])
    monkeypatch.setattr(cache, "scan_cache_dir", lambda: fake_info)

    assert cache.local_estimate_vram_gb("org/demo") == 5.0


def test_local_estimate_vram_gb_uses_most_recently_modified_revision(monkeypatch):
    stale = _FakeRevision(
        files=[_FakeFile("model.safetensors", round(1 * 1024**3))], last_modified=1
    )
    current = _FakeRevision(
        files=[_FakeFile("model.safetensors", round(9 * 1024**3))], last_modified=2
    )
    fake_info = _FakeCacheInfo(
        [_FakeRepo("org/demo", "model", revisions=[stale, current])]
    )
    monkeypatch.setattr(cache, "scan_cache_dir", lambda: fake_info)

    assert cache.local_estimate_vram_gb("org/demo") == 9.0


def test_local_estimate_vram_gb_none_when_repo_not_cached(monkeypatch):
    monkeypatch.setattr(cache, "scan_cache_dir", lambda: _FakeCacheInfo([]))

    assert cache.local_estimate_vram_gb("org/never-pulled") is None


def test_local_estimate_vram_gb_none_when_no_safetensors_files(monkeypatch):
    revision = _FakeRevision(files=[_FakeFile("model.gguf", round(4 * 1024**3))])
    fake_info = _FakeCacheInfo([_FakeRepo("org/gguf-only", "model", revisions=[revision])])
    monkeypatch.setattr(cache, "scan_cache_dir", lambda: fake_info)

    assert cache.local_estimate_vram_gb("org/gguf-only") is None


def test_local_estimate_vram_gb_none_when_cache_dir_never_created(monkeypatch):
    def raise_not_found():
        raise CacheNotFound("no cache yet", cache_dir="/root/.cache/huggingface/hub")

    monkeypatch.setattr(cache, "scan_cache_dir", raise_not_found)

    assert cache.local_estimate_vram_gb("org/demo") is None
