import pytest

from fllame.domain.recipe import Recipe, RecipeError


def test_from_dict_minimal():
    recipe = Recipe.from_dict("demo", {"repo_id": "org/demo", "image": "vllm/vllm-openai:v0.27.1"})
    assert recipe.handle == "demo"
    assert recipe.repo_id == "org/demo"
    assert recipe.image == "vllm/vllm-openai:v0.27.1"
    assert recipe.backend == "vllm"
    assert recipe.port == 8000
    assert recipe.gpus == "all"


def test_from_dict_missing_repo_id():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {"image": "vllm/vllm-openai:v0.27.1"})


def test_from_dict_missing_image_defaults_to_none():
    recipe = Recipe.from_dict("demo", {"repo_id": "org/demo"})
    assert recipe.image is None


def test_from_dict_handle_mismatch():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {"repo_id": "org/demo", "image": "img", "handle": "other"})


def test_from_dict_rejects_non_vllm_backend():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {"repo_id": "org/demo", "image": "img", "backend": "llamacpp"})


def test_from_dict_gpus_none():
    recipe = Recipe.from_dict("demo", {"repo_id": "org/demo", "image": "img", "gpus": "none"})
    assert recipe.gpus == "none"


def test_from_dict_rejects_invalid_gpus():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {"repo_id": "org/demo", "image": "img", "gpus": "2"})


def test_from_dict_rejects_env_hf_home():
    with pytest.raises(RecipeError):
        Recipe.from_dict(
            "demo",
            {"repo_id": "org/demo", "image": "img", "env": {"HF_HOME": "/somewhere"}},
        )


def test_from_dict_preinstall_defaults_to_empty():
    recipe = Recipe.from_dict("demo", {"repo_id": "org/demo", "image": "img"})
    assert recipe.preinstall == []


def test_from_dict_preinstall_round_trips():
    recipe = Recipe.from_dict(
        "demo",
        {"repo_id": "org/demo", "image": "img", "preinstall": ["pip install -U transformers"]},
    )
    assert recipe.preinstall == ["pip install -U transformers"]


def test_from_dict_rejects_empty_preinstall_entry():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {"repo_id": "org/demo", "image": "img", "preinstall": [""]})


def test_from_dict_rejects_non_string_preinstall_entry():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {"repo_id": "org/demo", "image": "img", "preinstall": [123]})
