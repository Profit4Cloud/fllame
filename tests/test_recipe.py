import pytest

from fllame.domain.recipe import Recipe, RecipeError


def test_from_dict_minimal():
    recipe = Recipe.from_dict(
        "demo", {"command": "vllm serve org/demo", "image": "vllm/vllm-openai:v0.27.1"}
    )
    assert recipe.handle == "demo"
    assert recipe.repo_id == "org/demo"
    assert recipe.command == "vllm serve org/demo"
    assert recipe.image == "vllm/vllm-openai:v0.27.1"
    assert recipe.backend == "vllm"
    assert recipe.port == 8000
    assert recipe.gpus == "all"


def test_from_dict_missing_command():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {"image": "vllm/vllm-openai:v0.27.1"})


def test_from_dict_malformed_command():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {"command": "docker run img", "image": "img"})


def test_from_dict_command_derives_serve_args_and_port():
    recipe = Recipe.from_dict(
        "demo", {"command": "vllm serve org/demo --port 9001 --max-model-len 8192", "image": "img"}
    )
    assert recipe.repo_id == "org/demo"
    assert recipe.serve_args == ["--port", "9001", "--max-model-len", "8192"]
    assert recipe.port == 9001


def test_from_dict_missing_image_defaults_to_none():
    recipe = Recipe.from_dict("demo", {"command": "vllm serve org/demo"})
    assert recipe.image is None


def test_from_dict_handle_mismatch():
    with pytest.raises(RecipeError):
        Recipe.from_dict(
            "demo", {"command": "vllm serve org/demo", "image": "img", "handle": "other"}
        )


def test_from_dict_rejects_non_vllm_backend():
    with pytest.raises(RecipeError):
        Recipe.from_dict(
            "demo", {"command": "vllm serve org/demo", "image": "img", "backend": "llamacpp"}
        )


def test_from_dict_gpus_none():
    recipe = Recipe.from_dict(
        "demo", {"command": "vllm serve org/demo", "image": "img", "gpus": "none"}
    )
    assert recipe.gpus == "none"


def test_from_dict_rejects_invalid_gpus():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {"command": "vllm serve org/demo", "image": "img", "gpus": "2"})


def test_from_dict_rejects_env_hf_home():
    with pytest.raises(RecipeError):
        Recipe.from_dict(
            "demo",
            {"command": "vllm serve org/demo", "image": "img", "env": {"HF_HOME": "/somewhere"}},
        )


def test_from_dict_preinstall_defaults_to_empty():
    recipe = Recipe.from_dict("demo", {"command": "vllm serve org/demo", "image": "img"})
    assert recipe.preinstall == []


def test_from_dict_preinstall_round_trips():
    recipe = Recipe.from_dict(
        "demo",
        {
            "command": "vllm serve org/demo",
            "image": "img",
            "preinstall": ["pip install -U transformers"],
        },
    )
    assert recipe.preinstall == ["pip install -U transformers"]


def test_from_dict_rejects_empty_preinstall_entry():
    with pytest.raises(RecipeError):
        Recipe.from_dict(
            "demo", {"command": "vllm serve org/demo", "image": "img", "preinstall": [""]}
        )


def test_from_dict_rejects_non_string_preinstall_entry():
    with pytest.raises(RecipeError):
        Recipe.from_dict(
            "demo", {"command": "vllm serve org/demo", "image": "img", "preinstall": [123]}
        )


def test_to_dict_round_trips_through_from_dict():
    """`to_dict()` canonicalizes `command` into a multi-line block (see
    `test_to_dict_renders_command_as_multiline_block`), so it's not
    byte-identical to whatever `command` was originally authored as -
    re-parsing that canonical form should still be semantically
    unchanged, which is what this checks via `to_dict()` again rather
    than raw dataclass equality.
    """
    recipe = Recipe.from_dict(
        "demo",
        {
            "command": "vllm serve org/demo --max-model-len 8192",
            "image": "img",
            "description": "a demo recipe",
            "env": {"FOO": "bar"},
            "preinstall": ["pip install -U transformers"],
        },
    )

    reloaded = Recipe.from_dict("demo", recipe.to_dict())

    assert reloaded.to_dict() == recipe.to_dict()
    assert reloaded.repo_id == recipe.repo_id
    assert reloaded.serve_args == recipe.serve_args


def test_to_dict_renders_command_as_multiline_block():
    recipe = Recipe.from_dict(
        "demo",
        {"command": "vllm serve org/demo --tensor-parallel-size 1 --enable-auto-tool-choice"},
    )

    assert recipe.to_dict()["command"] == (
        "vllm serve org/demo \\\n--tensor-parallel-size 1 \\\n--enable-auto-tool-choice"
    )


def test_to_dict_omits_unset_optional_fields():
    recipe = Recipe.from_dict("demo", {"command": "vllm serve org/demo"})

    data = recipe.to_dict()

    assert "image" not in data
    assert "env" not in data
    assert "preinstall" not in data
    assert "description" not in data
    assert "gpus" not in data
    assert data["command"] == "vllm serve org/demo"


def test_to_dict_includes_gpus_only_when_not_default():
    recipe = Recipe.from_dict(
        "demo", {"command": "vllm serve org/demo", "gpus": "none"}
    )

    assert recipe.to_dict()["gpus"] == "none"
