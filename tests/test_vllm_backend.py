from pathlib import Path

from fllame.backends import vllm as vllm_backend
from fllame.backends.vllm import (
    VllmServingBackend,
    cache_volume_host_path,
    default_gpu_memory_utilization,
    generate_dockerfile,
    gpu_memory_utilization_error,
    max_model_len_shortfall,
    tensor_parallel_size_mismatch_warning,
    validate_gpu_memory_utilization,
)
from fllame.domain.hardware import HardwareProfile
from fllame.domain.recipe import Recipe
from fllame.models.architecture import ModelArchitecture
from fllame.models.sizing import SizingConfig

_NO_GPU = HardwareProfile(
    gpu_name=None, gpu_count=0, vram_gb_per_gpu=None, ram_gb=None, chip_family="none"
)


def _grace_blackwell(ram_gb: float | None) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="NVIDIA GB10",
        gpu_count=1,
        vram_gb_per_gpu=None,
        ram_gb=ram_gb,
        chip_family="grace_blackwell",
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
        recipe, hf_cache_dir=Path("/home/user/.cache/huggingface"), hardware=_NO_GPU
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
        recipe, hf_cache_dir=Path("/home/alice/.cache/huggingface/hub"), hardware=_NO_GPU
    )

    assert service["volumes"] == ["${HOME}/.cache/huggingface/hub:/root/.cache/huggingface"]


def test_build_service_volume_uses_bare_home_variable_when_cache_is_home_itself(monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: Path("/home/alice"))
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(recipe, hf_cache_dir=Path("/home/alice"), hardware=_NO_GPU)

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
        recipe, hf_cache_dir=Path("/mnt/models/hf-cache"), hardware=_NO_GPU
    )

    assert service["volumes"] == ["/mnt/models/hf-cache:/root/.cache/huggingface"]


def test_build_service_command_has_no_duplicate_port():
    """recipe.port is only for the host mapping - not re-inserted into
    the container's argv on top of whatever --port the command already
    has (or vLLM's own default when it has none)."""
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo --port 9000", image="img")

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=_NO_GPU)

    assert service["command"].count("--port") == 1


def test_build_service_without_explicit_port_defaults_to_8000():
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=_NO_GPU)

    assert service["command"] == ["org/demo", "--gpu-memory-utilization", "0.92"]
    assert service["ports"] == ["8000:8000"]


def test_build_service_gpus_all_is_unconditional():
    """`gpus: "all"` is a hard default, not a recipe-level knob - pin
    specific device IDs by hand-editing the generated compose.yaml."""
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=_NO_GPU)

    assert service["gpus"] == "all"


def test_build_service_ipc_host_is_unconditional():
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=_NO_GPU)

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

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=_NO_GPU)

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


def test_default_gpu_memory_utilization_discrete_gpu_uses_flat_cap():
    profile = HardwareProfile(
        gpu_name="NVIDIA A100 80GB PCIe",
        gpu_count=1,
        vram_gb_per_gpu=80.0,
        ram_gb=256.0,
        chip_family="nvidia",
    )

    assert default_gpu_memory_utilization(profile) == 0.92


def test_default_gpu_memory_utilization_no_gpu_falls_back_to_flat_cap():
    assert default_gpu_memory_utilization(_NO_GPU) == 0.92


def test_default_gpu_memory_utilization_unified_memory_reserves_a_fixed_amount():
    # (32 - 5) / 32 = 0.84375, comfortably under the 0.92 cap.
    assert default_gpu_memory_utilization(_grace_blackwell(32.0)) == 27.0 / 32.0


def test_default_gpu_memory_utilization_unified_memory_still_capped_on_a_large_system():
    """A large enough unified-memory box would otherwise compute a
    reserved fraction above the flat cap - the cap always wins."""
    assert default_gpu_memory_utilization(_grace_blackwell(1000.0)) == 0.92


def test_default_gpu_memory_utilization_unified_memory_never_negative():
    assert default_gpu_memory_utilization(_grace_blackwell(2.0)) == 0.0


def test_default_gpu_memory_utilization_unified_memory_without_ram_reading_falls_back():
    assert default_gpu_memory_utilization(_grace_blackwell(None)) == 0.92


def test_build_service_sets_gpu_memory_utilization_by_default():
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(
        recipe, hf_cache_dir=Path("/cache"), hardware=_grace_blackwell(64.0)
    )

    assert service["command"] == ["org/demo", "--gpu-memory-utilization", "0.92"]


def test_build_service_never_overrides_an_explicit_gpu_memory_utilization():
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo --gpu-memory-utilization 0.75",
        image="img",
    )

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=_NO_GPU)

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

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=_NO_GPU)

    assert service["command"].count("--gpu-memory-utilization=0.75") == 1
    assert "--gpu-memory-utilization" not in service["command"]


def test_gpu_memory_utilization_error_none_when_recipe_sets_nothing():
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    assert gpu_memory_utilization_error(recipe, _NO_GPU) is None


def test_gpu_memory_utilization_error_none_for_a_safe_explicit_value():
    recipe = Recipe(
        handle="demo", command="vllm serve org/demo --gpu-memory-utilization 0.5", image="img"
    )

    assert gpu_memory_utilization_error(recipe, _NO_GPU) is None


def test_gpu_memory_utilization_error_flags_a_value_above_the_configured_ceiling():
    recipe = Recipe(
        handle="demo", command="vllm serve org/demo --gpu-memory-utilization 0.99", image="img"
    )

    error = gpu_memory_utilization_error(recipe, _NO_GPU)

    assert error is not None
    assert "0.99" in error
    assert "set-max-gpu-memory-utilization" in error


def test_gpu_memory_utilization_error_respects_a_raised_configured_ceiling():
    recipe = Recipe(
        handle="demo", command="vllm serve org/demo --gpu-memory-utilization 0.99", image="img"
    )

    error = gpu_memory_utilization_error(
        recipe, _NO_GPU, SizingConfig(max_gpu_memory_utilization=0.99)
    )

    assert error is None


def test_gpu_memory_utilization_error_flags_a_non_positive_value():
    recipe = Recipe(
        handle="demo", command="vllm serve org/demo --gpu-memory-utilization 0", image="img"
    )

    assert gpu_memory_utilization_error(recipe, _NO_GPU) is not None


def test_gpu_memory_utilization_error_flags_an_unparseable_value():
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo --gpu-memory-utilization notanumber",
        image="img",
    )

    error = gpu_memory_utilization_error(recipe, _NO_GPU)

    assert error is not None
    assert "isn't a number" in error


def test_build_service_leaves_an_invalid_explicit_value_untouched():
    """`build_service` never silently overrides an explicit choice -
    even an unsafe one - `gpu_memory_utilization_error` is what
    actually stops such a recipe from being built."""
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo", command="vllm serve org/demo --gpu-memory-utilization 0.99", image="img"
    )

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=_NO_GPU)

    assert service["command"].count("--gpu-memory-utilization") == 1
    assert "0.99" in service["command"]


def test_validate_gpu_memory_utilization_none_for_a_safe_value():
    service = {"command": ["org/demo", "--gpu-memory-utilization", "0.8"]}

    assert validate_gpu_memory_utilization(service) is None


def test_validate_gpu_memory_utilization_refuses_when_missing():
    service = {"command": ["org/demo"]}

    error = validate_gpu_memory_utilization(service)

    assert error is not None
    assert "no --gpu-memory-utilization" in error


def test_validate_gpu_memory_utilization_refuses_when_too_high():
    service = {"command": ["org/demo", "--gpu-memory-utilization", "1.0"]}

    error = validate_gpu_memory_utilization(service)

    assert error is not None
    assert "1.0" in error


def test_validate_gpu_memory_utilization_refuses_when_unparseable():
    service = {"command": ["org/demo", "--gpu-memory-utilization", "garbage"]}

    error = validate_gpu_memory_utilization(service)

    assert error is not None
    assert "isn't a number" in error


def test_validate_gpu_memory_utilization_respects_configured_ceiling():
    service = {"command": ["org/demo", "--gpu-memory-utilization", "0.95"]}

    assert (
        validate_gpu_memory_utilization(service, SizingConfig(max_gpu_memory_utilization=0.95))
        is None
    )


def test_validate_gpu_memory_utilization_no_command_at_all_refuses():
    assert validate_gpu_memory_utilization({}) is not None


_DISCRETE_GPU = HardwareProfile(
    gpu_name="NVIDIA A100 80GB PCIe",
    gpu_count=1,
    vram_gb_per_gpu=80.0,
    ram_gb=256.0,
    chip_family="nvidia",
)

_DENSE_ARCH = ModelArchitecture(
    num_layers=32, num_kv_heads=8, head_dim=128, max_context_length=131072
)


def test_build_service_recipes_own_max_model_len_is_respected(monkeypatch):
    """Nothing else is even checked once the recipe sets its own value -
    the architecture/cache lookups it would otherwise need are never
    invoked."""
    called = []
    monkeypatch.setattr(
        vllm_backend, "read_architecture", lambda repo_id: called.append(repo_id) or None
    )
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo", command="vllm serve org/demo --max-model-len 2048", image="img"
    )

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=_DISCRETE_GPU)

    assert service["command"].count("--max-model-len") == 1
    assert "2048" in service["command"]
    assert called == []


def test_build_service_unsupported_architecture_skips_silently(monkeypatch):
    monkeypatch.setattr(vllm_backend, "local_estimate_vram_gb", lambda repo_id: 16.0)
    monkeypatch.setattr(vllm_backend, "read_architecture", lambda repo_id: None)
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=_DISCRETE_GPU)

    assert "--max-model-len" not in service["command"]


def test_build_service_uncached_model_skips_silently(monkeypatch):
    monkeypatch.setattr(vllm_backend, "local_estimate_vram_gb", lambda repo_id: None)
    monkeypatch.setattr(vllm_backend, "read_architecture", lambda repo_id: _DENSE_ARCH)
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=_DISCRETE_GPU)

    assert "--max-model-len" not in service["command"]


def test_build_service_generous_budget_injects_value_within_architectural_ceiling(monkeypatch):
    monkeypatch.setattr(vllm_backend, "local_estimate_vram_gb", lambda repo_id: 16.0)
    monkeypatch.setattr(vllm_backend, "read_architecture", lambda repo_id: _DENSE_ARCH)
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=_DISCRETE_GPU)

    assert service["command"].count("--max-model-len") == 1
    index = service["command"].index("--max-model-len")
    injected = int(service["command"][index + 1])
    assert 0 < injected <= _DENSE_ARCH.max_context_length


def test_build_service_shortfall_injects_nothing(monkeypatch):
    monkeypatch.setattr(vllm_backend, "local_estimate_vram_gb", lambda repo_id: 15.0)
    monkeypatch.setattr(vllm_backend, "read_architecture", lambda repo_id: _DENSE_ARCH)
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")
    tiny_gpu = HardwareProfile(
        gpu_name="tiny", gpu_count=1, vram_gb_per_gpu=20.0, ram_gb=64.0, chip_family="nvidia"
    )

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"), hardware=tiny_gpu)

    assert "--max-model-len" not in service["command"]


def test_max_model_len_shortfall_none_for_generous_budget(monkeypatch):
    monkeypatch.setattr(vllm_backend, "local_estimate_vram_gb", lambda repo_id: 16.0)
    monkeypatch.setattr(vllm_backend, "read_architecture", lambda repo_id: _DENSE_ARCH)
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    assert max_model_len_shortfall(recipe, _DISCRETE_GPU) is None


def test_max_model_len_shortfall_names_repo_and_numbers_when_too_small(monkeypatch):
    monkeypatch.setattr(vllm_backend, "local_estimate_vram_gb", lambda repo_id: 15.0)
    monkeypatch.setattr(vllm_backend, "read_architecture", lambda repo_id: _DENSE_ARCH)
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")
    tiny_gpu = HardwareProfile(
        gpu_name="tiny", gpu_count=1, vram_gb_per_gpu=20.0, ram_gb=64.0, chip_family="nvidia"
    )

    message = max_model_len_shortfall(recipe, tiny_gpu)

    assert message is not None
    assert "org/demo" in message
    assert "4096" in message
    assert "15.0 GB" in message


def test_max_model_len_shortfall_matches_build_service_no_injection():
    """Same underlying resolution - a shortfall and no injected flag
    always come together."""
    recipe = Recipe(handle="demo", command="vllm serve org/demo --max-model-len 2048", image="img")

    assert max_model_len_shortfall(recipe, _DISCRETE_GPU) is None


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
