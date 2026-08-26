import pytest

from fllame.domain.recipe import Recipe, RecipeError


def test_from_dict_minimal():
    recipe = Recipe.from_dict("demo", {"repo_id": "org/demo"})
    assert recipe.handle == "demo"
    assert recipe.repo_id == "org/demo"
    assert recipe.backend == "vllm"
    assert recipe.port == 8000


def test_from_dict_missing_repo_id():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {})


def test_from_dict_handle_mismatch():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {"repo_id": "org/demo", "handle": "other"})


def test_from_dict_rejects_non_vllm_backend():
    with pytest.raises(RecipeError):
        Recipe.from_dict("demo", {"repo_id": "org/demo", "backend": "llamacpp"})
