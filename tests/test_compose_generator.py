from pathlib import Path

from fllame.backends.vllm import VllmServingBackend
from fllame.compose.generator import generate_compose, write_compose_file
from fllame.domain.recipe import Recipe


def test_generate_compose_single_service_keyed_by_handle():
    recipe = Recipe(handle="a", command="vllm serve org/a --port 8001", image="img-a")

    compose = generate_compose(recipe, backend=VllmServingBackend(), hf_cache_dir=Path("/cache"))

    assert set(compose["services"].keys()) == {"a"}
    assert compose["services"]["a"]["image"] == "img-a"
    assert compose["services"]["a"]["ports"] == ["8001:8001"]


def test_write_compose_file_creates_parent_dirs(tmp_path: Path):
    compose = {"services": {"a": {"image": "img-a"}}}
    path = tmp_path / "nested" / "docker-compose.yml"

    write_compose_file(compose, path)

    assert path.is_file()
    assert "img-a" in path.read_text()
