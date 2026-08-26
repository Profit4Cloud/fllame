from pathlib import Path

from fllame.backends.vllm import VllmServingBackend
from fllame.domain.recipe import Recipe


def test_build_service_with_gpus():
    backend = VllmServingBackend()
    recipe = Recipe(
        handle="demo",
        repo_id="org/demo",
        image="vllm/vllm-openai:v0.27.1",
        port=9000,
        serve_args=["--max-model-len", "8192"],
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


def test_build_service_without_gpus():
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", repo_id="org/demo", image="img", gpus="none")

    service = backend.build_service(recipe, hf_cache_dir=Path("/cache"))

    assert "deploy" not in service
