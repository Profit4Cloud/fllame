from pathlib import Path

from typer.testing import CliRunner

from fllame.cli import app

runner = CliRunner()


def test_recipe_list_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLLAME_RECIPES_DIR", str(tmp_path))

    result = runner.invoke(app, ["recipe", "list"])

    assert result.exit_code == 0
    assert "No recipes found" in result.stdout


def test_recipe_show(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLLAME_RECIPES_DIR", str(tmp_path))
    (tmp_path / "demo.yaml").write_text("repo_id: org/demo\n")

    result = runner.invoke(app, ["recipe", "show", "demo"])

    assert result.exit_code == 0
    assert "org/demo" in result.stdout
    assert "vllm serve org/demo" in result.stdout


def test_recipe_show_missing_handle(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLLAME_RECIPES_DIR", str(tmp_path))

    result = runner.invoke(app, ["recipe", "show", "nope"])

    assert result.exit_code == 1
