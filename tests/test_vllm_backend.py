from pathlib import Path

from fllame.backends.vllm import (
    VllmServingBackend,
    cache_volume_host_path,
    generate_dockerfile,
    tensor_parallel_size_mismatch_warning,
    validate_gpu_memory_utilization,
)
from fllame.domain.hardware import HardwareProfile
from fllame.domain.recipe import Recipe

_NO_GPU = HardwareProfile(
    gpu_name=None, gpu_count=0, vram_gb_per_gpu=None, ram_gb=None, chip_family="none"
)

_DISCRETE_GPU = HardwareProfile(
    gpu_name="NVIDIA A100 80GB PCIe",
    gpu_count=1,
    vram_gb_per_gpu=80.0,
    ram_gb=256.0,
    chip_family="nvidia",
)


def test_cache_volume_host_path_extracts_host_side():
    service = {"volumes": ["/home/alice/.cache/huggingface:/root/.cache/huggingface"]}

    assert cache_volume_host_path(service) == "/home/alice/.cache/huggingface"


def test_cache_volume_host_path_none_when_no_volumes():
    assert cache_volume_host_path({}) is None


def test_cache_volume_host_path_none_when_no_matching_mount():
    service = {"volumes": ["/some/other/path:/some/other/container/path"]}

    assert cache_volume_host_path(service) is None


def test_build_service_with_gpus(monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent-home"))
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo --port 9000 --max-model-len 8192",
        image="vllm/vllm-openai:v0.27.1",
        env={"FOO": "bar"},
    )

    service = backend.build_service(
        recipe,
        hf_cache_dir=Path("/home/user/.cache/huggingface"),
        default_gpu_memory_utilization=0.92,
    )

    assert service["image"] == "vllm/vllm-openai:v0.27.1"
    assert service["entrypoint"] == ["vllm", "serve"]
    assert service["command"] == [
        "org/demo",
        "--port",
        "9000",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.92",
    ]
    assert service["ports"] == ["9000:9000"]
    assert service["environment"] == [
        "HF_HOME=/root/.cache/huggingface",
        "HF_HUB_CACHE=/root/.cache/huggingface",
        "HF_HUB_OFFLINE=1",
        "FOO=bar",
    ]
    assert service["volumes"] == ["/home/user/.cache/huggingface:/root/.cache/huggingface"]
    assert service["ipc"] == "host"
    assert service["gpus"] == "all"


def test_build_service_volume_uses_home_variable_when_cache_is_under_home(monkeypatch):
    """The host side of the HF cache bind mount is written as
    `${HOME}/...` rather than a literal absolute path when it sits
    under the current user's home directory - the default,
    out-of-the-box location - so the generated compose.yaml stays
    correct after being copied to a different machine or account,
    rather than baking in the one home directory it was generated
    under."""
    monkeypatch.setattr(Path, "home", lambda: Path("/home/alice"))
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(
        recipe,
        hf_cache_dir=Path("/home/alice/.cache/huggingface/hub"),
        default_gpu_memory_utilization=0.92,
    )

    assert service["volumes"] == ["${HOME}/.cache/huggingface/hub:/root/.cache/huggingface"]


def test_build_service_volume_uses_bare_home_variable_when_cache_is_home_itself(monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: Path("/home/alice"))
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(
        recipe, hf_cache_dir=Path("/home/alice"), default_gpu_memory_utilization=0.92
    )

    assert service["volumes"] == ["${HOME}:/root/.cache/huggingface"]


def test_build_service_volume_falls_back_to_literal_path_outside_home(monkeypatch):
    """A custom HF_HOME/HF_HUB_CACHE pointed somewhere other than the
    home directory (a separate data volume, a network share) has no
    portable `${HOME}`-relative form - the literal path is the correct
    fallback."""
    monkeypatch.setattr(Path, "home", lambda: Path("/home/alice"))
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(
        recipe, hf_cache_dir=Path("/mnt/models/hf-cache"), default_gpu_memory_utilization=0.92
    )

    assert service["volumes"] == ["/mnt/models/hf-cache:/root/.cache/huggingface"]


def test_build_service_command_has_no_duplicate_port():
    """recipe.port is only for the host mapping - not re-inserted into
    the container's argv on top of whatever --port the command already
    has (or vLLM's own default when it has none)."""
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo --port 9000", image="img")

    service = backend.build_service(
        recipe, hf_cache_dir=Path("/cache"), default_gpu_memory_utilization=0.92
    )

    assert service["command"].count("--port") == 1


def test_build_service_without_explicit_port_defaults_to_8000():
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(
        recipe, hf_cache_dir=Path("/cache"), default_gpu_memory_utilization=0.92
    )

    assert service["command"] == ["org/demo", "--gpu-memory-utilization", "0.92"]
    assert service["ports"] == ["8000:8000"]


def test_build_service_gpus_all_is_unconditional():
    """`gpus: "all"` is a hard default, not a recipe-level knob - pin
    specific device IDs by hand-editing the generated compose.yaml."""
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(
        recipe, hf_cache_dir=Path("/cache"), default_gpu_memory_utilization=0.92
    )

    assert service["gpus"] == "all"


def test_build_service_ipc_host_is_unconditional():
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(
        recipe, hf_cache_dir=Path("/cache"), default_gpu_memory_utilization=0.92
    )

    assert service["ipc"] == "host"


def test_build_service_preinstall_never_changes_entrypoint_or_command():
    """`preinstall` is now a Dockerfile-build-time concern (see
    `generate_dockerfile`) - the generated service always runs the
    plain `vllm serve` form, whether or not the recipe has one."""
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo --tensor-parallel-size 1",
        image="vllm/vllm-openai:v0.27.1",
        preinstall=["pip install -U transformers"],
    )

    service = backend.build_service(
        recipe, hf_cache_dir=Path("/cache"), default_gpu_memory_utilization=0.92
    )

    assert "build" not in service
    assert service["entrypoint"] == ["vllm", "serve"]
    assert service["command"] == [
        "org/demo",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.92",
    ]


def test_generate_dockerfile_none_without_preinstall():
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    assert generate_dockerfile(recipe) is None


def test_generate_dockerfile_one_run_line_per_preinstall_entry_in_order():
    """One `RUN` per entry, not joined with `&&` into a single layer -
    keeps Docker's layer cache reusing an unchanged earlier step even
    when a later one changes."""
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo",
        image="vllm/vllm-openai:v0.27.1",
        preinstall=["pip install -U transformers", "pip install foo"],
    )

    dockerfile = generate_dockerfile(recipe)

    assert dockerfile == (
        "FROM vllm/vllm-openai:v0.27.1\n"
        "RUN pip install -U transformers\n"
        "RUN pip install foo\n"
    )


def test_generate_dockerfile_single_preinstall_entry():
    recipe = Recipe(
        handle="demo", command="vllm serve org/demo", image="img", preinstall=["pip install foo"]
    )

    assert generate_dockerfile(recipe) == "FROM img\nRUN pip install foo\n"


def test_build_service_never_overrides_an_explicit_gpu_memory_utilization():
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo --gpu-memory-utilization 0.75",
        image="img",
    )

    service = backend.build_service(
        recipe, hf_cache_dir=Path("/cache"), default_gpu_memory_utilization=0.92
    )

    assert service["command"].count("--gpu-memory-utilization") == 1
    assert "0.75" in service["command"]
    assert "0.92" not in service["command"]


def test_build_service_respects_gpu_memory_utilization_equals_form():
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo --gpu-memory-utilization=0.75",
        image="img",
    )

    service = backend.build_service(
        recipe, hf_cache_dir=Path("/cache"), default_gpu_memory_utilization=0.92
    )

    assert service["command"].count("--gpu-memory-utilization=0.75") == 1
    assert "--gpu-memory-utilization" not in service["command"]


def test_build_service_never_validates_an_explicit_value():
    """The recipe wins outright, however implausible the value - no
    range-checking at build time, only `validate_gpu_memory_utilization`
    at `serve` catches an obviously-wrong value."""
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo", command="vllm serve org/demo --gpu-memory-utilization 0.99", image="img"
    )

    service = backend.build_service(
        recipe, hf_cache_dir=Path("/cache"), default_gpu_memory_utilization=0.92
    )

    assert service["command"].count("--gpu-memory-utilization") == 1
    assert "0.99" in service["command"]


def test_validate_gpu_memory_utilization_none_for_a_safe_value():
    service = {"command": ["org/demo", "--gpu-memory-utilization", "0.8"]}

    assert validate_gpu_memory_utilization(service) is None


def test_validate_gpu_memory_utilization_none_for_a_value_at_the_mathematical_ceiling():
    service = {"command": ["org/demo", "--gpu-memory-utilization", "1.0"]}

    assert validate_gpu_memory_utilization(service) is None


def test_validate_gpu_memory_utilization_refuses_when_missing():
    service = {"command": ["org/demo"]}

    error = validate_gpu_memory_utilization(service)

    assert error is not None
    assert "no --gpu-memory-utilization" in error


def test_validate_gpu_memory_utilization_refuses_when_above_one():
    service = {"command": ["org/demo", "--gpu-memory-utilization", "1.5"]}

    error = validate_gpu_memory_utilization(service)

    assert error is not None
    assert "1.5" in error


def test_validate_gpu_memory_utilization_refuses_when_non_positive():
    service = {"command": ["org/demo", "--gpu-memory-utilization", "0"]}

    assert validate_gpu_memory_utilization(service) is not None


def test_validate_gpu_memory_utilization_refuses_when_unparseable():
    service = {"command": ["org/demo", "--gpu-memory-utilization", "garbage"]}

    error = validate_gpu_memory_utilization(service)

    assert error is not None
    assert "isn't a number" in error


def test_validate_gpu_memory_utilization_no_command_at_all_refuses():
    assert validate_gpu_memory_utilization({}) is not None


def test_tensor_parallel_size_mismatch_warning_none_when_matching():
    recipe = Recipe(
        handle="demo", command="vllm serve org/demo --tensor-parallel-size 1", image="img"
    )

    assert tensor_parallel_size_mismatch_warning(recipe, _DISCRETE_GPU) is None


def test_tensor_parallel_size_mismatch_warning_none_when_no_gpu_detected():
    recipe = Recipe(
        handle="demo", command="vllm serve org/demo --tensor-parallel-size 2", image="img"
    )

    assert tensor_parallel_size_mismatch_warning(recipe, _NO_GPU) is None


def test_tensor_parallel_size_mismatch_warning_too_many_gpus_requested():
    recipe = Recipe(
        handle="demo", command="vllm serve org/demo --tensor-parallel-size 4", image="img"
    )

    message = tensor_parallel_size_mismatch_warning(recipe, _DISCRETE_GPU)

    assert message is not None
    assert "4" in message
    assert "1" in message


def test_tensor_parallel_size_mismatch_warning_gpus_left_unused():
    two_gpus = HardwareProfile(
        gpu_name="NVIDIA A100 80GB PCIe",
        gpu_count=2,
        vram_gb_per_gpu=80.0,
        ram_gb=256.0,
        chip_family="nvidia",
    )
    recipe = Recipe(
        handle="demo", command="vllm serve org/demo --tensor-parallel-size 1", image="img"
    )

    message = tensor_parallel_size_mismatch_warning(recipe, two_gpus)

    assert message is not None
    assert "unused" in message
    assert "2" in message


def test_tensor_parallel_size_mismatch_warning_defaults_to_one_when_unset():
    two_gpus = HardwareProfile(
        gpu_name="NVIDIA A100 80GB PCIe",
        gpu_count=2,
        vram_gb_per_gpu=80.0,
        ram_gb=256.0,
        chip_family="nvidia",
    )
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    message = tensor_parallel_size_mismatch_warning(recipe, two_gpus)

    assert message is not None
    assert "unused" in message
