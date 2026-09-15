from pathlib import Path

from fllame.recipes import build_state


def test_load_missing_file_returns_defaults(tmp_path: Path):
    state = build_state.load(tmp_path)

    assert state.compose_hash is None
    assert state.dockerfile_hash is None
    assert state.image_synced_via_config is False


def test_save_then_load_round_trips(tmp_path: Path):
    state = build_state.BuildState(
        compose_hash="abc", dockerfile_hash="def", image_synced_via_config=True
    )

    build_state.save(tmp_path, state)
    loaded = build_state.load(tmp_path)

    assert loaded == state


def test_save_omits_unset_fields_from_the_file(tmp_path: Path):
    build_state.save(tmp_path, build_state.BuildState(compose_hash="abc"))

    text = build_state.path_for(tmp_path).read_text()

    assert "dockerfile_hash" not in text
    assert "image_synced_via_config" not in text


def test_hash_text_is_stable_and_content_sensitive():
    assert build_state.hash_text("a") == build_state.hash_text("a")
    assert build_state.hash_text("a") != build_state.hash_text("b")


def test_for_current_files_reads_compose_and_dockerfile_from_disk(tmp_path: Path):
    (tmp_path / "compose.yaml").write_text("services: {}\n")
    (tmp_path / "Dockerfile").write_text("FROM img\n")

    state = build_state.for_current_files(tmp_path)

    assert state.compose_hash == build_state.hash_text("services: {}\n")
    assert state.dockerfile_hash == build_state.hash_text("FROM img\n")
    assert state.image_synced_via_config is False


def test_for_current_files_dockerfile_hash_none_when_absent(tmp_path: Path):
    (tmp_path / "compose.yaml").write_text("services: {}\n")

    state = build_state.for_current_files(tmp_path)

    assert state.dockerfile_hash is None


def test_for_current_files_passes_through_image_synced_flag(tmp_path: Path):
    (tmp_path / "compose.yaml").write_text("services: {}\n")

    state = build_state.for_current_files(tmp_path, image_synced_via_config=True)

    assert state.image_synced_via_config is True
