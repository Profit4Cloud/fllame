from pathlib import Path

import pytest

from fllame.domain.recipe import Recipe, RecipeError
from fllame.recipes.store import RecipeStore, autofix_whitespace


def test_list_and_load(tmp_path: Path):
    (tmp_path / "demo.yaml").write_text(
        "image: vllm/vllm-openai:v0.27.1\ncommand: vllm serve org/demo --port 9001\n"
    )
    store = RecipeStore(tmp_path)

    assert store.list_handles() == ["demo"]
    recipe = store.load("demo")
    assert recipe.repo_id == "org/demo"
    assert recipe.port == 9001


def test_missing_recipe_raises(tmp_path: Path):
    store = RecipeStore(tmp_path)

    with pytest.raises(RecipeError):
        store.load("nope")


def test_load_malformed_yaml_raises_recipe_error_not_yaml_error(tmp_path: Path):
    (tmp_path / "demo.yaml").write_text(
        "image: vllm/vllm-openai:v0.27.1\n\tcommand: vllm serve org/demo\n"
    )
    store = RecipeStore(tmp_path)

    with pytest.raises(RecipeError, match="invalid YAML"):
        store.load("demo")


def test_load_non_mapping_content_raises_recipe_error(tmp_path: Path):
    (tmp_path / "demo.yaml").write_text("- just\n- a\n- list\n")
    store = RecipeStore(tmp_path)

    with pytest.raises(RecipeError, match="mapping"):
        store.load("demo")


def test_load_empty_file_raises_missing_command(tmp_path: Path):
    (tmp_path / "demo.yaml").write_text("")
    store = RecipeStore(tmp_path)

    with pytest.raises(RecipeError, match="command"):
        store.load("demo")


def test_autofix_whitespace_expands_tabs_and_normalizes_line_endings():
    text = "image: img\r\nenv:\r\n\tFOO: bar  \r\ncommand: vllm serve org/demo\r\n"

    fixed = autofix_whitespace(text)

    assert "\t" not in fixed
    assert "\r" not in fixed
    assert "  \n" not in fixed  # trailing whitespace stripped
    assert fixed == "image: img\nenv:\n  FOO: bar\ncommand: vllm serve org/demo\n"


def test_autofix_whitespace_is_noop_on_clean_text():
    text = "image: img\ncommand: vllm serve org/demo\n"

    assert autofix_whitespace(text) == text


def test_missing_directory_lists_nothing(tmp_path: Path):
    store = RecipeStore(tmp_path / "does-not-exist")

    assert store.list_handles() == []


def test_next_available_handle_no_collision(tmp_path: Path):
    store = RecipeStore(tmp_path)

    assert store.next_available_handle("demo") == "demo"


def test_next_available_handle_suffixes_on_collision(tmp_path: Path):
    (tmp_path / "demo.yaml").write_text("command: vllm serve org/demo\nimage: img\n")
    store = RecipeStore(tmp_path)

    assert store.next_available_handle("demo") == "demo_2"


def test_next_available_handle_skips_multiple_collisions(tmp_path: Path):
    (tmp_path / "demo.yaml").write_text("command: vllm serve org/demo\nimage: img\n")
    (tmp_path / "demo_2.yaml").write_text("command: vllm serve org/demo\nimage: img\n")
    store = RecipeStore(tmp_path)

    assert store.next_available_handle("demo") == "demo_3"


def test_save_then_load_round_trips(tmp_path: Path):
    """save() canonicalizes `command` into a multi-line block, so the
    loaded recipe's `command` field isn't byte-identical to the one
    Recipe(...) was constructed with here - checked via to_dict() (which
    is idempotent under that canonicalization) rather than raw equality.
    """
    store = RecipeStore(tmp_path / "nested")
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo --max-model-len 8192",
        image="vllm/vllm-openai:v0.27.1",
        env={"FOO": "bar"},
    )

    store.save(recipe)
    loaded = store.load("demo")

    assert loaded.to_dict() == recipe.to_dict()
    assert loaded.repo_id == recipe.repo_id
    assert loaded.serve_args == recipe.serve_args


def test_save_omits_empty_env(tmp_path: Path):
    store = RecipeStore(tmp_path)
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    store.save(recipe)

    text = (tmp_path / "demo.yaml").read_text()
    assert "env:" not in text


def test_save_omits_image_when_unset(tmp_path: Path):
    store = RecipeStore(tmp_path)
    recipe = Recipe(handle="demo", command="vllm serve org/demo")

    store.save(recipe)

    text = (tmp_path / "demo.yaml").read_text()
    assert "image:" not in text
    assert store.load("demo").image is None


def test_load_recipe_without_image_key(tmp_path: Path):
    (tmp_path / "demo.yaml").write_text("command: vllm serve org/demo\n")
    store = RecipeStore(tmp_path)

    assert store.load("demo").image is None


def test_save_then_load_round_trips_preinstall(tmp_path: Path):
    store = RecipeStore(tmp_path)
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo",
        image="img",
        preinstall=["pip install -U transformers"],
    )

    store.save(recipe)
    loaded = store.load("demo")

    assert loaded == recipe


def test_save_omits_empty_preinstall(tmp_path: Path):
    store = RecipeStore(tmp_path)
    recipe = Recipe(handle="demo", command="vllm serve org/demo", image="img")

    store.save(recipe)

    text = (tmp_path / "demo.yaml").read_text()
    assert "preinstall:" not in text


def test_save_writes_command_last(tmp_path: Path):
    """`command` is the copy-pasteable part - kept last in the file so
    it's easy to find and select regardless of what else the recipe
    sets."""
    store = RecipeStore(tmp_path)
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo",
        image="img",
        env={"FOO": "bar"},
        preinstall=["pip install -U transformers"],
    )

    store.save(recipe)

    lines = [line for line in (tmp_path / "demo.yaml").read_text().splitlines() if line]
    assert lines[-1] == "command: vllm serve org/demo"


def test_save_renders_multi_arg_command_as_literal_block(tmp_path: Path):
    store = RecipeStore(tmp_path)
    recipe = Recipe(
        handle="demo",
        command="vllm serve org/demo --tensor-parallel-size 1 --enable-auto-tool-choice",
    )

    store.save(recipe)

    assert (tmp_path / "demo.yaml").read_text() == (
        "command: |-\n"
        "  vllm serve org/demo \\\n"
        "  --tensor-parallel-size 1 \\\n"
        "  --enable-auto-tool-choice\n"
    )
    # Round-trips back to the same tokens, backslashes and all.
    assert store.load("demo").serve_args == [
        "--tensor-parallel-size",
        "1",
        "--enable-auto-tool-choice",
    ]


def test_remove_deletes_file(tmp_path: Path):
    (tmp_path / "demo.yaml").write_text("command: vllm serve org/demo\nimage: img\n")
    store = RecipeStore(tmp_path)

    store.remove("demo")

    assert not (tmp_path / "demo.yaml").exists()


def test_remove_missing_recipe_raises(tmp_path: Path):
    store = RecipeStore(tmp_path)

    with pytest.raises(RecipeError):
        store.remove("nope")
