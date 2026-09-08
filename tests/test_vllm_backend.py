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


def test_render_dockerfile_none_without_preinstall():
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    assert backend.render_dockerfile(recipe) is None


def test_render_dockerfile_with_preinstall():
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo",
        image="vllm/vllm-openai:v0.27.1",
        preinstall=["pip install -U transformers", "pip install foo"],
    )

    dockerfile = backend.render_dockerfile(recipe)

    assert dockerfile == (
        "FROM vllm/vllm-openai:v0.27.1\n"
        "RUN pip install -U transformers\n"
        "RUN pip install foo\n"
    )


def test_build_service_with_preinstall_builds_instead_of_bare_image():
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo",
        image="vllm/vllm-openai:v0.27.1",
        preinstall=["pip install -U transformers"],
    )

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"))

    assert service["build"] == {"context": "."}
    assert service["image"] == "fllame-demo:latest"
