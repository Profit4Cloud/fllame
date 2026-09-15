from pathlib import Path

from fllame.backends.vllm import VllmServingBackend
from fllame.compose.generator import generate_compose, write_compose_file
from fllame.domain.hardware import HardwareProfile
from fllame.domain.recipe import Recipe

_NO_GPU = HardwareProfile(
    gpu_name=None, gpu_count=0, vram_gb_per_gpu=None, ram_gb=None, chip_family="none"
)


def test_generate_compose_single_service_keyed_by_handle():
    recipe = Recipe(handle="a", command="vllm serve org/a --port 8001", image="img-a")

    compose = generate_compose(
        recipe, backend=VllmServingBackend(), hf_cache_dir=Path("/cache"), hardware=_NO_GPU
    )

    assert set(compose["services"].keys()) == {"a"}
    assert compose["services"]["a"]["image"] == "img-a"
    assert compose["services"]["a"]["ports"] == ["8001:8001"]


def test_write_compose_file_creates_parent_dirs(tmp_path: Path):
    compose = {"services": {"a": {"image": "img-a"}}}
    path = tmp_path / "nested" / "compose.yaml"

    write_compose_file(compose, path)

    assert path.is_file()
    assert "img-a" in path.read_text()
