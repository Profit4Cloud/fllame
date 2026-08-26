from pathlib import Path

import pytest

from fllame.domain.recipe import RecipeError
from fllame.recipes.store import RecipeStore


def test_list_and_load(tmp_path: Path):
    (tmp_path / "demo.yaml").write_text("repo_id: org/demo\nport: 9001\n")
    store = RecipeStore(tmp_path)

    assert store.list_handles() == ["demo"]
    recipe = store.load("demo")
    assert recipe.repo_id == "org/demo"
    assert recipe.port == 9001


def test_missing_recipe_raises(tmp_path: Path):
    store = RecipeStore(tmp_path)

    with pytest.raises(RecipeError):
        store.load("nope")


def test_missing_directory_lists_nothing(tmp_path: Path):
    store = RecipeStore(tmp_path / "does-not-exist")

    assert store.list_handles() == []
