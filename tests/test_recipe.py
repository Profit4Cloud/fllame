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


def test_from_dict_missing_image():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {"repo_id": "org/demo"})


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
