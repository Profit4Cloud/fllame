from fllame.backends.vllm import VllmServingBackend
from fllame.domain.recipe import Recipe


def test_build_argv():
    backend = VllmServingBackend()
    recipe = Recipe(handle="demo", repo_id="org/demo", serve_args=["--max-model-len", "8192"])

    argv = backend.build_argv(recipe, port=9000)

    assert argv == [
        "vllm",
        "serve",
        "org/demo",
        "--port",
        "9000",
        "--max-model-len",
        "8192",
    ]
