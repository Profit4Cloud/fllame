from pathlib import Path

from fllame.backends.vllm import VllmServingBackend
from fllame.domain.recipe import Recipe


def test_build_service_with_gpus():
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo --port 9000 --max-model-len 8192",
        image="vllm/vllm-openai:v0.27.1",
        env={"FOO": "bar"},
    )

    service = backend.build_service(recipe, hf_cache_dir=Path("/home/user/.cache/huggingface"))

    assert service["image"] == "vllm/vllm-openai:v0.27.1"
    assert service["entrypoint"] == ["vllm", "serve"]
    assert service["command"] == ["org/demo", "--port", "9000", "--max-model-len", "8192"]
    assert service["ports"] == ["9000:9000"]
    assert service["environment"] == {"HF_HOME": "/root/.cache/huggingface", "FOO": "bar"}
    assert service["volumes"] == ["/home/user/.cache/huggingface:/root/.cache/huggingface"]
    assert service["deploy"]["resources"]["reservations"]["devices"][0]["driver"] == "nvidia"


def test_build_service_command_has_no_duplicate_port():
    """recipe.port is only for the host mapping - not re-inserted into
    the container's argv on top of whatever --port the command already
    has (or vLLM's own default when it has none)."""
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo --port 9000", image="img")

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"))

    assert service["command"].count("--port") == 1


def test_build_service_without_explicit_port_defaults_to_8000():
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"))

    assert service["command"] == ["org/demo"]
    assert service["ports"] == ["8000:8000"]


def test_build_service_without_gpus():
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img", gpus="none")

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"))

    assert "deploy" not in service


def test_build_service_with_preinstall_wraps_command_in_a_shell():
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo --tensor-parallel-size 1",
        image="vllm/vllm-openai:v0.27.1",
        preinstall=["pip install -U transformers"],
    )

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"))

    assert "build" not in service
    assert service["image"] == "vllm/vllm-openai:v0.27.1"
    assert service["entrypoint"] == ["sh", "-c"]
    assert service["command"] == [
        "pip install -U transformers && exec vllm serve org/demo --tensor-parallel-size 1"
    ]


def test_build_service_with_multiple_preinstall_steps_in_order():
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo",
        image="img",
        preinstall=["pip install foo", "pip install bar"],
    )

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"))

    assert service["command"] == ["pip install foo && pip install bar && exec vllm serve org/demo"]


def test_build_service_preinstall_shell_escapes_vllm_serve_args():
    """Preinstall entries are joined in as trusted shell text verbatim,
    but the vllm serve portion still needs escaping now that it's
    embedded in a shell string rather than passed as literal argv."""
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo",
        command='vllm serve org/demo --served-model-name "my model"',
        image="img",
        preinstall=["pip install foo"],
    )

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"))

    assert service["command"] == [
        "pip install foo && exec vllm serve org/demo --served-model-name 'my model'"
    ]
