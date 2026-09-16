import json

from huggingface_hub.errors import CacheNotFound

from fllame.models import architecture


class _FakeRepo:
    def __init__(self, repo_id: str, repo_type: str, revisions=None):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revisions = revisions if revisions is not None else []


class _FakeCacheInfo:
    def __init__(self, repos):
        self.repos = repos


class _FakeRevision:
    def __init__(self, snapshot_path, last_modified: float = 0):
        self.snapshot_path = snapshot_path
        self.files = []
        self.last_modified = last_modified


def _write_config(tmp_path, config: dict):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return tmp_path


_LLAMA_STYLE_CONFIG = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 8192,
}


def test_read_architecture_parses_well_formed_llama_style_config(tmp_path, monkeypatch):
    snapshot = _write_config(tmp_path, _LLAMA_STYLE_CONFIG)
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "model", [_FakeRevision(snapshot)])])
    monkeypatch.setattr(architecture, "scan_cache_dir", lambda: fake_info)

    arch = architecture.read_architecture("org/demo")

    assert arch is not None
    assert arch.num_layers == 32
    assert arch.num_kv_heads == 8
    assert arch.head_dim == 128
    assert arch.max_context_length == 8192


def test_read_architecture_num_kv_heads_falls_back_to_num_attention_heads(tmp_path, monkeypatch):
    config = dict(_LLAMA_STYLE_CONFIG)
    del config["num_key_value_heads"]
    snapshot = _write_config(tmp_path, config)
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "model", [_FakeRevision(snapshot)])])
    monkeypatch.setattr(architecture, "scan_cache_dir", lambda: fake_info)

    arch = architecture.read_architecture("org/demo")

    assert arch is not None
    assert arch.num_kv_heads == 32


def test_read_architecture_head_dim_derived_when_absent(tmp_path, monkeypatch):
    config = dict(_LLAMA_STYLE_CONFIG)
    snapshot = _write_config(tmp_path, config)
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "model", [_FakeRevision(snapshot)])])
    monkeypatch.setattr(architecture, "scan_cache_dir", lambda: fake_info)

    arch = architecture.read_architecture("org/demo")

    assert arch is not None
    assert arch.head_dim == 4096 // 32


def test_read_architecture_head_dim_used_directly_when_present(tmp_path, monkeypatch):
    config = dict(_LLAMA_STYLE_CONFIG, hidden_size=4097, head_dim=128)
    snapshot = _write_config(tmp_path, config)
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "model", [_FakeRevision(snapshot)])])
    monkeypatch.setattr(architecture, "scan_cache_dir", lambda: fake_info)

    arch = architecture.read_architecture("org/demo")

    assert arch is not None
    assert arch.head_dim == 128


def test_read_architecture_none_when_num_hidden_layers_missing(tmp_path, monkeypatch):
    config = dict(_LLAMA_STYLE_CONFIG)
    del config["num_hidden_layers"]
    snapshot = _write_config(tmp_path, config)
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "model", [_FakeRevision(snapshot)])])
    monkeypatch.setattr(architecture, "scan_cache_dir", lambda: fake_info)

    assert architecture.read_architecture("org/demo") is None


def test_read_architecture_none_when_max_position_embeddings_missing(tmp_path, monkeypatch):
    config = dict(_LLAMA_STYLE_CONFIG)
    del config["max_position_embeddings"]
    snapshot = _write_config(tmp_path, config)
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "model", [_FakeRevision(snapshot)])])
    monkeypatch.setattr(architecture, "scan_cache_dir", lambda: fake_info)

    assert architecture.read_architecture("org/demo") is None


def test_read_architecture_none_when_repo_not_cached(monkeypatch):
    monkeypatch.setattr(architecture, "scan_cache_dir", lambda: _FakeCacheInfo([]))

    assert architecture.read_architecture("org/never-pulled") is None


def test_read_architecture_none_when_cache_dir_never_created(monkeypatch):
    def raise_not_found():
        raise CacheNotFound("no cache yet", cache_dir="/root/.cache/huggingface/hub")

    monkeypatch.setattr(architecture, "scan_cache_dir", raise_not_found)

    assert architecture.read_architecture("org/demo") is None


def test_read_architecture_none_when_config_json_missing(tmp_path, monkeypatch):
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "model", [_FakeRevision(tmp_path)])])
    monkeypatch.setattr(architecture, "scan_cache_dir", lambda: fake_info)

    assert architecture.read_architecture("org/demo") is None


def test_read_architecture_none_when_config_json_malformed(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{not valid json")
    fake_info = _FakeCacheInfo([_FakeRepo("org/demo", "model", [_FakeRevision(tmp_path)])])
    monkeypatch.setattr(architecture, "scan_cache_dir", lambda: fake_info)

    assert architecture.read_architecture("org/demo") is None
