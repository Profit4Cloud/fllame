from pathlib import Path

from fllame.backends.vllm import VllmServingBackend
from fllame.compose.generator import generate_compose, write_compose_file
from fllame.domain.recipe import Recipe


def test_generate_compose_one_service_per_recipe():
    recipes = [
        Recipe(handle="a", command="vllm serve org/a", image="img-a"),
        Recipe(handle="b", command="vllm serve org/b --port 8001", image="img-b"),
    ]

    compose = generate_compose(recipes, backend=VllmServingBackend(), hf_cache_dir=Path("/cache"))

    assert set(compose["services"].keys()) == {"a", "b"}
    assert compose["services"]["a"]["image"] == "img-a"
    assert compose["services"]["b"]["ports"] == ["8001:8001"]


def test_write_compose_file_creates_parent_dirs(tmp_path: Path):
    compose = {"services": {"a": {"image": "img-a"}}}
    path = tmp_path / "nested" / "docker-compose.yml"

    write_compose_file(compose, path)

    assert path.is_file()
    assert "img-a" in path.read_text()
