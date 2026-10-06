import io
import json
import time
from pathlib import Path

import pytest
import yaml
from huggingface_hub.errors import HfHubHTTPError
from typer.testing import CliRunner

import fllame.cli as cli
from fllame import config
from fllame.cli import app
from fllame.domain.hardware import HardwareProfile
from fllame.models.discovery import ModelCandidate
from fllame.models.updater import UpdateStatus
from fllame.models.vram import VramEstimate, VramPart

runner = CliRunner()


def _write_recipe(tmp_path: Path, handle: str = "demo") -> None:
    directory = tmp_path / handle
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "recipe.yaml").write_text(
        "image: vllm/vllm-openai:v0.27.1\ncommand: vllm serve org/demo\n"
    )


def _write_compose(tmp_path: Path, handle: str = "demo") -> None:
    """Writes HANDLE's compose.yaml via `recipe build`, the only command
    that ever does - callers must already have `is_model_cached` mocked
    True."""
    result = runner.invoke(app, ["recipe", "build", handle])
    assert result.exit_code == 0, result.output


_NO_HARDWARE_SIGNAL = HardwareProfile(
    gpu_name=None,
    gpu_count=0,
    vram_gb_per_gpu=None,
    ram_gb=None,
    chip_family="none",
    supported_quantizations=[],
    scanned_at="",
)


def _isolate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FLLAME_RECIPES_DIR", str(tmp_path))
    monkeypatch.setenv("FLLAME_CONFIG_FILE", str(tmp_path / "config.yaml"))
    # `serve`'s pre-flight VRAM sanity check silently skips without a
    # memory-budget figure to compare against - this is the default for
    # every test so it never fires (and never scans this machine's real
    # HF cache) unless a test explicitly opts into exercising it (see
    # the test_serve_vram_* tests below).
    monkeypatch.setattr(cli, "scan_hardware", lambda: _NO_HARDWARE_SIGNAL)
    # `recipe build`'s Docker validation step (a real `docker build` or
    # `docker compose pull`) defaults to a fake success so ordinary
    # tests never need a real docker daemon - tests exercising the
    # invoked command or a failure override `cli.subprocess.run`
    # themselves afterward.
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: _FakeCompletedProcess())


def _dialogue_input(*parts: str) -> str:
    """Builds stdin input for `recipe add`'s guided dialogue: each part
    is one line. A blank ("") part accepts the image prompt's prefilled
    default, or ends whichever of the preinstall/env/command blocks is
    currently being read.
    """
    return "\n".join(parts) + "\n"


class _FakeCompletedProcess:
    returncode = 0
    stdout = ""
    stderr = ""


def _capturing_run(captured: dict):
    def fake_run(command, **kwargs):
        captured["command"] = command
        return _FakeCompletedProcess()

    return fake_run


def _capturing_run_all(commands: list):
    def fake_run(command, **kwargs):
        commands.append(command)
        return _FakeCompletedProcess()

    return fake_run


class _FakeSubprocessResult:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_compose_ps_json(containers_by_handle: dict[str, list[dict]]):
    """A `cli.subprocess.run` replacement that answers `docker compose
    ps --format json` for HANDLE from `containers_by_handle`, and
    succeeds trivially for everything else (`recipe build`'s own
    `docker build`/`docker compose pull` during test setup)."""

    def fake_run(command, **kwargs):
        if command[-2:] == ["--format", "json"]:
            handle = Path(command[command.index("-f") + 1]).parent.name
            containers = containers_by_handle.get(handle, [])
            return _FakeSubprocessResult(stdout=json.dumps(containers))
        return _FakeCompletedProcess()

    return fake_run


def test_print_table_pads_columns_to_widest_cell(capsys):
    cli._print_table(
        ["REPO_ID", "QUANT", "PARAMS"],
        [["org/short", "awq", "7.0B"], ["org/a-much-longer-name", "gptq", "70.0B"]],
    )

    lines = capsys.readouterr().out.splitlines()
    # Same column boundary on every line - the whole point of a table.
    assert lines[0].index("QUANT") == lines[1].index("awq") == lines[2].index("gptq")


def test_format_count_compact_thousands_and_millions():
    assert cli._format_count(None) == "unknown"
    assert cli._format_count(999) == "999"
    assert cli._format_count(12_345) == "12.3k"
    assert cli._format_count(1_234_567) == "1.2M"


def test_format_relative_time_buckets(monkeypatch):
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 1, 1, tzinfo=UTC)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(cli, "datetime", _FixedDatetime)

    assert cli._format_relative_time(None) == "unknown"
    assert cli._format_relative_time(now - timedelta(seconds=5)) == "a few seconds ago"
    assert cli._format_relative_time(now - timedelta(days=3)) == "3 days ago"
    assert cli._format_relative_time(now - timedelta(days=1)) == "1 day ago"


def test_help_flag_short_alias():
    result = runner.invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_recipe_list_empty(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["recipe", "list"])

    assert result.exit_code == 0
    assert "No recipes found" in result.stdout


def test_recipe_show(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    result = runner.invoke(app, ["recipe", "show", "demo"])

    assert result.exit_code == 0
    assert "org/demo" in result.stdout
    assert "vllm/vllm-openai:v0.27.1" in result.stdout


def test_recipe_show_missing_handle(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["recipe", "show", "nope"])

    assert result.exit_code == 1


def test_recipe_add_dialogue_collects_env_and_command(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pasted = _dialogue_input(
        "",  # no preinstall commands
        "FOO=bar",
        "",  # end env vars
        "vllm serve meta-llama/Llama-3-8B-Instruct --port 8000",
    )

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "vllm/vllm-openai:v0.27.1"],
        input=pasted,
    )

    assert result.exit_code == 0
    saved = tmp_path / "llama-3-8b-instruct" / "recipe.yaml"
    assert saved.is_file()
    assert "meta-llama/Llama-3-8B-Instruct" in saved.read_text()
    assert "FOO: bar" in saved.read_text()


def test_recipe_add_accepts_vllm_serve_line_as_trailing_args(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "recipe",
            "add",
            "--image",
            "vllm/vllm-openai:v0.27.1",
            "vllm",
            "serve",
            "Qwen/Qwen3-8B-FP8",
            "--tensor-parallel-size",
            "1",
            "--enable-auto-tool-choice",
        ],
    )

    assert result.exit_code == 0
    saved = tmp_path / "qwen3-8b-fp8" / "recipe.yaml"
    assert saved.is_file()
    text = saved.read_text()
    assert "Qwen/Qwen3-8B-FP8" in text
    assert "--tensor-parallel-size" in text
    assert "--enable-auto-tool-choice" in text


def test_recipe_add_trailing_args_bad_paste_still_validates(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "img:v1", "org/repo", "-x"],
    )

    assert result.exit_code == 1
    assert list(tmp_path.glob("*.yaml")) == []


def test_recipe_add_second_recipe_for_same_model_gets_suffixed(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pasted = _dialogue_input("", "", "vllm serve org/demo")

    runner.invoke(app, ["recipe", "add", "--image", "img:v1"], input=pasted)
    result = runner.invoke(app, ["recipe", "add", "--image", "img:v1"], input=pasted)

    assert result.exit_code == 0
    assert (tmp_path / "demo" / "recipe.yaml").is_file()
    assert (tmp_path / "demo_2" / "recipe.yaml").is_file()


def test_recipe_add_pinned_image_no_warning(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "vllm/vllm-openai:v0.27.1"],
        input=_dialogue_input("", "", "vllm serve org/demo"),
    )

    assert "unpinned" not in result.output


def test_recipe_add_unpinned_image_warns(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "vllm/vllm-openai:latest"],
        input=_dialogue_input("", "", "vllm serve org/demo"),
    )

    assert "unpinned" in result.output


def test_recipe_add_rejects_bad_command(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "img:v1"],
        input=_dialogue_input("", "", "docker run img:v1"),
    )

    assert result.exit_code == 1
    assert list(tmp_path.glob("*/recipe.yaml")) == []


def test_recipe_add_dialogue_command_without_trailing_backslash(tmp_path: Path, monkeypatch):
    """Some model card/recipes.vllm.ai examples show one flag per line
    with no `\\` continuation marker at all - the dialogue shouldn't
    require one."""
    _isolate(tmp_path, monkeypatch)
    pasted = _dialogue_input(
        "",
        "",
        "vllm serve org/demo",
        "--tensor-parallel-size 1",
        "--enable-auto-tool-choice",
    )

    result = runner.invoke(app, ["recipe", "add", "--image", "img:v1"], input=pasted)

    assert result.exit_code == 0
    text = (tmp_path / "demo" / "recipe.yaml").read_text()
    assert "--tensor-parallel-size" in text
    assert "--enable-auto-tool-choice" in text


def test_recipe_add_no_command_given_is_an_error(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "img:v1"],
        input=_dialogue_input("", ""),
    )

    assert result.exit_code == 1
    assert list(tmp_path.glob("*/recipe.yaml")) == []


def test_recipe_add_dialogue_collects_preinstall_commands(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pasted = _dialogue_input(
        "pip install -U 'transformers>=5.8.0'",
        "",  # end preinstall
        "",  # no env vars
        "vllm serve org/demo",
    )

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "vllm/vllm-openai:v0.27.1"],
        input=pasted,
    )

    assert result.exit_code == 0
    text = (tmp_path / "demo" / "recipe.yaml").read_text()
    assert "preinstall:" in text
    assert "transformers>=5.8.0" in text


def test_recipe_add_replaces_uv_pip_install_and_notifies(tmp_path: Path, monkeypatch):
    """`uv pip install ...`, common on a pasted model card, targets the
    uv-managed venv `vllm serve` doesn't actually run in - fllame swaps
    in plain `pip install` automatically and tells the operator, since
    it silently changes what they typed."""
    _isolate(tmp_path, monkeypatch)
    pasted = _dialogue_input(
        'uv pip install -U "transformers>=5.8.0"',
        "",  # end preinstall
        "",  # no env vars
        "vllm serve org/demo",
    )

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "vllm/vllm-openai:v0.27.1"],
        input=pasted,
    )

    assert result.exit_code == 0
    assert "replaced 'uv pip install' with 'pip install'" in result.output
    text = (tmp_path / "demo" / "recipe.yaml").read_text()
    assert "uv pip install" not in text
    assert 'pip install -U "transformers>=5.8.0"' in text


def test_recipe_add_plain_pip_install_preinstall_unchanged_no_note(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pasted = _dialogue_input(
        "pip install -U transformers",
        "",  # end preinstall
        "",  # no env vars
        "vllm serve org/demo",
    )

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "vllm/vllm-openai:v0.27.1"],
        input=pasted,
    )

    assert result.exit_code == 0
    assert "replaced 'uv pip install'" not in result.output


def test_recipe_edit_missing_handle(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["recipe", "edit", "nope"])

    assert result.exit_code == 1


def test_recipe_edit_revalidates_after_editing(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    def fake_edit(filename):
        return None

    monkeypatch.setattr(cli.click, "edit", fake_edit)

    result = runner.invoke(app, ["recipe", "edit", "demo"])

    assert result.exit_code == 0
    assert "valid" in result.output


def test_recipe_edit_tolerates_stripped_command_indentation_and_renormalizes(
    tmp_path: Path, monkeypatch
):
    """The exact mistake reported: hand-editing strips what looks like
    meaningless leading whitespace from the command block, which would
    otherwise break the YAML literal block scalar outright. Not only
    should this load fine (RecipeStore.load's own leniency), the file
    should come back out re-normalized to fllame's canonical rendering,
    not left in the technically-fragile shape the edit left it in."""
    _isolate(tmp_path, monkeypatch)
    (tmp_path / "demo").mkdir(parents=True)
    (tmp_path / "demo" / "recipe.yaml").write_text(
        "image: vllm/vllm-openai:v0.27.1\n"
        "command: |-\n"
        "  vllm serve org/demo \\\n"
        "  --tensor-parallel-size 1\n"
    )

    def fake_edit(filename):
        p = Path(filename)
        p.write_text("\n".join(line.lstrip() for line in p.read_text().splitlines()) + "\n")

    monkeypatch.setattr(cli.click, "edit", fake_edit)

    result = runner.invoke(app, ["recipe", "edit", "demo"])

    assert result.exit_code == 0
    assert "saved and valid" in result.output
    text = (tmp_path / "demo" / "recipe.yaml").read_text()
    assert text == (
        "image: vllm/vllm-openai:v0.27.1\n"
        "command: |-\n"
        "  vllm serve org/demo \\\n"
        "  --tensor-parallel-size 1\n"
    )


def test_recipe_edit_reports_now_invalid_recipe_and_reverts_when_declined(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    original = (tmp_path / "demo" / "recipe.yaml").read_text()

    def fake_edit(filename):
        Path(filename).write_text("image: vllm/vllm-openai:v0.27.1\n")  # command now missing
        return None

    monkeypatch.setattr(cli.click, "edit", fake_edit)

    result = runner.invoke(app, ["recipe", "edit", "demo"], input="n\n")

    assert result.exit_code == 1
    assert "no longer a valid recipe" in result.output
    assert "reverted" in result.output
    assert (tmp_path / "demo" / "recipe.yaml").read_text() == original


def test_recipe_edit_reverts_when_confirm_is_aborted(tmp_path: Path, monkeypatch):
    """A closed stdin (Ctrl-D) or Ctrl-C on the reopen-or-revert prompt
    raises click.exceptions.Abort - this must still revert the file, not
    leave the now-invalid recipe.yaml on disk with nothing to undo it."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    original = (tmp_path / "demo" / "recipe.yaml").read_text()

    def fake_edit(filename):
        Path(filename).write_text("image: vllm/vllm-openai:v0.27.1\n")  # command now missing
        return None

    def fake_confirm(*args, **kwargs):
        raise cli.click.exceptions.Abort()

    monkeypatch.setattr(cli.click, "edit", fake_edit)
    monkeypatch.setattr(cli.typer, "confirm", fake_confirm)

    result = runner.invoke(app, ["recipe", "edit", "demo"])

    assert result.exit_code == 1
    assert "reverted" in result.output
    assert (tmp_path / "demo" / "recipe.yaml").read_text() == original


def test_recipe_edit_reopens_editor_and_succeeds_when_accepted(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    calls = []

    def fake_edit(filename):
        calls.append(None)
        if len(calls) == 1:
            Path(filename).write_text("image: vllm/vllm-openai:v0.27.1\n")  # invalid
        else:
            Path(filename).write_text("command: vllm serve org/demo\n")  # now fixed

    monkeypatch.setattr(cli.click, "edit", fake_edit)

    result = runner.invoke(app, ["recipe", "edit", "demo"], input="y\n")

    assert result.exit_code == 0
    assert "saved and valid" in result.output
    assert len(calls) == 2


def test_recipe_edit_autofixes_tab_indentation_without_prompting(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    def fake_edit(filename):
        Path(filename).write_text(
            "image: vllm/vllm-openai:v0.27.1\nenv:\n\tFOO: bar\ncommand: vllm serve org/demo\n"
        )

    monkeypatch.setattr(cli.click, "edit", fake_edit)

    result = runner.invoke(app, ["recipe", "edit", "demo"])

    assert result.exit_code == 0
    assert "saved and valid" in result.output
    assert "\t" not in (tmp_path / "demo" / "recipe.yaml").read_text()


def test_recipe_remove_with_yes_flag(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    result = runner.invoke(app, ["recipe", "remove", "demo", "--yes"])

    assert result.exit_code == 0
    assert not (tmp_path / "demo" / "recipe.yaml").exists()


def test_recipe_remove_prompts_and_respects_no(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    result = runner.invoke(app, ["recipe", "remove", "demo"], input="n\n")

    assert result.exit_code == 0
    assert (tmp_path / "demo" / "recipe.yaml").exists()


def test_recipe_remove_deletes_its_generated_compose_folder(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    handle_folder = tmp_path / "demo"
    (handle_folder / "compose.yaml").write_text("services: {}\n")

    result = runner.invoke(app, ["recipe", "remove", "demo", "--yes"])

    assert result.exit_code == 0
    assert not handle_folder.exists()


def test_recipe_remove_missing_handle(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["recipe", "remove", "nope", "--yes"])

    assert result.exit_code == 1


def test_recipe_remove_warns_docker_image_is_not_cleaned_up(tmp_path: Path, monkeypatch):
    """The warning is the same whether or not the recipe had a
    Dockerfile - either way, Docker holds state fllame doesn't track."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    result = runner.invoke(app, ["recipe", "remove", "demo", "--yes"])

    assert result.exit_code == 0
    assert "docker image prune" in result.output or "docker rmi" in result.output


def test_recipe_remove_warns_docker_image_for_preinstall_recipe_too(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    directory = tmp_path / "demo"
    directory.mkdir(parents=True)
    (directory / "recipe.yaml").write_text(
        "image: img\npreinstall:\n- pip install foo\ncommand: vllm serve org/demo\n"
    )
    (directory / "Dockerfile").write_text("FROM img\nRUN pip install foo\n")

    result = runner.invoke(app, ["recipe", "remove", "demo", "--yes"])

    assert result.exit_code == 0
    assert "docker image prune" in result.output or "docker rmi" in result.output


def test_recipe_add_dialogue_prompts_for_image_when_no_default_configured(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["recipe", "add"],
        input=_dialogue_input("vllm/vllm-openai:v0.27.1", "", "", "vllm serve org/demo"),
    )

    assert result.exit_code == 0
    saved = tmp_path / "demo" / "recipe.yaml"
    assert "image: vllm/vllm-openai:v0.27.1" in saved.read_text()


def test_recipe_add_dialogue_accepting_configured_default_leaves_image_unset(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"])

    result = runner.invoke(
        app,
        ["recipe", "add"],
        input=_dialogue_input("", "", "", "vllm serve org/demo"),  # accept the prefilled default
    )

    assert result.exit_code == 0
    saved = tmp_path / "demo" / "recipe.yaml"
    # Not written into the recipe - it should keep following the
    # configured default even if that default changes later.
    assert "image:" not in saved.read_text()


def test_recipe_add_dialogue_overriding_configured_default_pins_image(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"])

    result = runner.invoke(
        app,
        ["recipe", "add"],
        input=_dialogue_input(
            "vllm/vllm-openai:v0.28.0", "", "", "vllm serve org/demo"
        ),  # typed something different from the prefilled default
    )

    assert result.exit_code == 0
    saved = tmp_path / "demo" / "recipe.yaml"
    assert "image: vllm/vllm-openai:v0.28.0" in saved.read_text()


def test_recipe_add_explicit_image_flag_overrides_configured_default(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"])

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "vllm/vllm-openai:v0.28.0"],
        input=_dialogue_input("", "", "vllm serve org/demo"),
    )

    assert result.exit_code == 0
    saved = tmp_path / "demo" / "recipe.yaml"
    assert "image: vllm/vllm-openai:v0.28.0" in saved.read_text()


def test_recipe_add_warns_when_configured_default_is_unpinned(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:latest"])

    result = runner.invoke(
        app,
        ["recipe", "add"],
        input=_dialogue_input("", "", "", "vllm serve org/demo"),  # accept the unpinned default
    )

    assert "unpinned" in result.output


def test_config_show_unset(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "unset" in result.stdout
    assert "vllm/vllm-openai:latest" in result.stdout


def test_config_set_and_show_default_image(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    set_result = runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"])
    show_result = runner.invoke(app, ["config", "show"])

    assert set_result.exit_code == 0
    assert "unpinned" not in set_result.output
    assert show_result.exit_code == 0
    assert "default_image: vllm/vllm-openai:v0.27.1" in show_result.stdout


def test_config_show_default_gpu_memory_utilization_defaults_to_0_92(tmp_path: Path, monkeypatch):
    """Unlike default_image, this setting is never shown as unset -
    it always has a concrete value, 0.92 until explicitly changed."""
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "default_gpu_memory_utilization: 0.92" in result.stdout


def test_config_set_default_image_warns_unpinned(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:latest"])

    assert "unpinned" in result.output


def _write_compose_with_image(tmp_path: Path, handle: str, image: str) -> Path:
    """A minimal, already-generated-looking compose.yaml, with a hand
    edit (a custom `shm_size:`) alongside the `image:` line - used to
    verify that batch image updates touch only the image value."""
    directory = tmp_path / handle
    directory.mkdir(parents=True, exist_ok=True)
    compose_path = directory / "compose.yaml"
    compose_path.write_text(f"services:\n  {handle}:\n    image: {image}\n    shm_size: 2gb\n")
    return compose_path


def test_config_set_default_image_no_prior_default_skips_batch_update(tmp_path: Path, monkeypatch):
    """With no previously configured default, there's no old value to
    search compose.yaml files for - nothing to prompt about."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    _write_compose_with_image(tmp_path, "demo", "vllm/vllm-openai:latest")

    result = runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"])

    assert result.exit_code == 0
    assert "compose.yaml" not in result.output


def test_config_set_default_image_offers_batch_update_and_applies_on_confirm(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.26.0"])
    compose_path = _write_compose_with_image(tmp_path, "demo", "vllm/vllm-openai:v0.26.0")

    result = runner.invoke(
        app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"], input="y\n"
    )

    assert result.exit_code == 0
    assert str(compose_path) in result.output
    text = compose_path.read_text()
    assert "image: vllm/vllm-openai:v0.27.1" in text
    # Nothing else in the hand-edited file was touched.
    assert "shm_size: 2gb" in text


def test_config_set_default_image_batch_update_declined_leaves_files_untouched(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.26.0"])
    compose_path = _write_compose_with_image(tmp_path, "demo", "vllm/vllm-openai:v0.26.0")

    result = runner.invoke(
        app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"], input="n\n"
    )

    assert result.exit_code == 0
    assert "image: vllm/vllm-openai:v0.26.0" in compose_path.read_text()


def test_config_set_default_image_never_touches_a_custom_pinned_image(tmp_path: Path, monkeypatch):
    """A compose.yaml whose image doesn't literally match the previous
    default - a recipe-level pin, or a hand edit - is never listed or
    replaced, confirmation or not."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.26.0"])
    compose_path = _write_compose_with_image(tmp_path, "demo", "custom/pinned-image:v1")

    result = runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"])

    assert result.exit_code == 0
    assert "compose.yaml" not in result.output
    assert "image: custom/pinned-image:v1" in compose_path.read_text()


def _write_dockerfile_and_compose(tmp_path: Path, handle: str, image: str) -> Path:
    """A minimal, already-generated-looking Dockerfile/compose.yaml pair
    for a preinstall recipe - compose.yaml's `image` is the local build
    tag, never the base image, so only the Dockerfile's `FROM` line can
    ever match a previous default."""
    directory = tmp_path / handle
    directory.mkdir(parents=True, exist_ok=True)
    dockerfile_path = directory / "Dockerfile"
    dockerfile_path.write_text(f"FROM {image}\nRUN pip install -U transformers\n")
    (directory / "compose.yaml").write_text(
        f"services:\n  {handle}:\n    image: fllame-{handle}:latest\n"
    )
    return dockerfile_path


def test_config_set_default_image_finds_and_updates_a_preinstall_recipes_dockerfile(
    tmp_path: Path, monkeypatch
):
    """compose.yaml's `image` is the local build tag for a preinstall
    recipe, never the base image - so it's the Dockerfile's `FROM` line
    that needs to be found and updated instead."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.26.0"])
    dockerfile_path = _write_dockerfile_and_compose(tmp_path, "demo", "vllm/vllm-openai:v0.26.0")

    result = runner.invoke(
        app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"], input="y\n"
    )

    assert result.exit_code == 0
    assert str(dockerfile_path) in result.output
    text = dockerfile_path.read_text()
    assert text.startswith("FROM vllm/vllm-openai:v0.27.1\n")
    # Nothing else in the file was touched.
    assert "RUN pip install -U transformers" in text
    # The local image is rebuilt automatically from the patched Dockerfile.
    assert "building 'fllame-demo:latest'" in result.output


def test_config_set_default_image_dockerfile_rebuild_failure_still_keeps_the_patch(
    tmp_path: Path, monkeypatch
):
    """The Dockerfile edit already applied is not rolled back just
    because the follow-up rebuild failed - only the local image is
    stale until a retry succeeds."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.26.0"])
    dockerfile_path = _write_dockerfile_and_compose(tmp_path, "demo", "vllm/vllm-openai:v0.26.0")

    class _FailedProcess:
        returncode = 1

    def fake_run(command):
        if command[:2] == ["docker", "build"]:
            return _FailedProcess()
        return _FakeCompletedProcess()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = runner.invoke(
        app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"], input="y\n"
    )

    assert result.exit_code == 0
    assert "Failed to build 'fllame-demo:latest'" in result.output
    assert "fllame recipe build demo" in result.output
    assert dockerfile_path.read_text().startswith("FROM vllm/vllm-openai:v0.27.1\n")


def test_config_set_default_image_never_touches_compose_yaml_for_a_preinstall_recipe(
    tmp_path: Path, monkeypatch
):
    """The compose.yaml half of a preinstall recipe is never even
    inspected - it already points at the local build tag, not the base
    image, so it can never match a previous default."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.26.0"])
    _write_dockerfile_and_compose(tmp_path, "demo", "vllm/vllm-openai:v0.26.0")
    compose_path = tmp_path / "demo" / "compose.yaml"
    original_compose = compose_path.read_text()

    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"], input="y\n")

    assert compose_path.read_text() == original_compose


def test_config_set_default_image_confirm_prompt_explains_text_replace_and_dockerfile_rebuild(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.26.0"])
    _write_compose_with_image(tmp_path, "demo", "vllm/vllm-openai:v0.26.0")

    result = runner.invoke(
        app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"], input="n\n"
    )

    assert "text replace" in result.output
    assert "docker build" in result.output


def test_recipe_show_falls_back_to_configured_default_image(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    (tmp_path / "demo").mkdir(parents=True)
    (tmp_path / "demo" / "recipe.yaml").write_text("command: vllm serve org/demo\n")
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"])

    result = runner.invoke(app, ["recipe", "show", "demo"])

    assert result.exit_code == 0
    assert "vllm/vllm-openai:v0.27.1" in result.stdout


def test_recipe_show_falls_back_to_hardcoded_image_when_nothing_configured(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    (tmp_path / "demo").mkdir(parents=True)
    (tmp_path / "demo" / "recipe.yaml").write_text("command: vllm serve org/demo\n")

    result = runner.invoke(app, ["recipe", "show", "demo"])

    assert result.exit_code == 0
    assert "vllm/vllm-openai:latest" in result.stdout


def test_hardware_scan_with_gpu(monkeypatch):
    monkeypatch.setattr(
        cli,
        "scan_hardware",
        lambda: HardwareProfile(
            gpu_name="NVIDIA A100 80GB PCIe",
            gpu_count=1,
            vram_gb_per_gpu=80.0,
            ram_gb=256.0,
            chip_family="nvidia",
            supported_quantizations=["awq", "gptq", "fp8"],
            scanned_at="2026-01-01T00:00:00+00:00",
        ),
    )

    result = runner.invoke(app, ["hardware", "scan"])

    assert result.exit_code == 0
    assert "NVIDIA A100 80GB PCIe x1" in result.stdout
    assert "80.0 GB" in result.stdout
    assert "awq, gptq, fp8" in result.stdout


def test_hardware_scan_unified_memory_gpu(monkeypatch):
    monkeypatch.setattr(
        cli,
        "scan_hardware",
        lambda: HardwareProfile(
            gpu_name="NVIDIA GB10",
            gpu_count=1,
            vram_gb_per_gpu=None,
            ram_gb=128.0,
            chip_family="grace_blackwell",
            supported_quantizations=["awq", "gptq", "fp8", "fp4", "nvfp4"],
            scanned_at="2026-01-01T00:00:00+00:00",
        ),
    )

    result = runner.invoke(app, ["hardware", "scan"])

    assert result.exit_code == 0
    assert "NVIDIA GB10 x1" in result.stdout
    assert "unified memory" in result.stdout


def test_hardware_scan_no_gpu(monkeypatch):
    monkeypatch.setattr(
        cli,
        "scan_hardware",
        lambda: HardwareProfile(
            gpu_name=None,
            gpu_count=0,
            vram_gb_per_gpu=None,
            ram_gb=16.0,
            chip_family="none",
            supported_quantizations=[],
            scanned_at="2026-01-01T00:00:00+00:00",
        ),
    )

    result = runner.invoke(app, ["hardware", "scan"])

    assert result.exit_code == 0
    assert "none detected" in result.stdout


def test_model_pull_downloads_given_repo_id(monkeypatch):
    """`model pull` takes a repo_id directly, not a recipe handle -
    `model` commands never depend on recipes (see CLAUDE.md,
    "Layering")."""
    monkeypatch.setattr(cli, "pull_model", lambda repo_id: f"/cache/{repo_id}")

    result = runner.invoke(app, ["model", "pull", "org/demo"])

    assert result.exit_code == 0
    assert "org/demo" in result.stdout


def test_model_pull_hub_error_gives_friendly_message_not_traceback(monkeypatch):
    """A transient Hub failure mid-download (rate limiting, a connection
    blip) must never surface as a raw traceback - see CLAUDE.md/git
    history for the real-world case this guards against."""

    def raise_error(repo_id):
        raise HfHubHTTPError("429 Client Error: Too Many Requests")

    monkeypatch.setattr(cli, "pull_model", raise_error)

    result = runner.invoke(app, ["model", "pull", "org/demo"])

    assert result.exit_code == 1
    assert "429 Client Error" in result.output
    assert "resume rather than start over" in result.output
    assert "Traceback" not in result.output


def test_model_pull_permission_error_gives_friendly_message(monkeypatch):
    """The HF cache is shared - something else (e.g. a `serve` container
    that ran as root) can leave files behind the operator's own account
    can't write to. This must never surface as a raw traceback either,
    and the message should point at the fix (chown), not just print the
    bare OS error."""

    def raise_error(repo_id):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "pull_model", raise_error)

    result = runner.invoke(app, ["model", "pull", "org/demo"])

    assert result.exit_code == 1
    assert "chown" in result.output
    assert "Traceback" not in result.output


def test_model_list_empty(monkeypatch):
    monkeypatch.setattr(cli, "list_cached_models", lambda: [])

    result = runner.invoke(app, ["model", "list"])

    assert result.exit_code == 0
    assert "No models cached" in result.stdout


def test_model_list_shows_cached_repos(monkeypatch):
    class _FakeRepo:
        repo_id = "org/demo"
        size_on_disk_str = "16.1GB"
        last_modified_str = "2 days ago"

    monkeypatch.setattr(cli, "list_cached_models", lambda: [_FakeRepo()])

    result = runner.invoke(app, ["model", "list"])

    assert result.exit_code == 0
    assert "org/demo" in result.stdout
    assert "16.1GB" in result.stdout
    assert "2 days ago" in result.stdout


def test_model_update_no_cached_models(monkeypatch):
    monkeypatch.setattr(cli, "list_cached_models", lambda: [])

    result = runner.invoke(app, ["model", "update"])

    assert result.exit_code == 0
    assert "No models cached" in result.stdout


def test_model_update_reports_up_to_date_and_stale(monkeypatch):
    class _FakeRepo:
        def __init__(self, repo_id):
            self.repo_id = repo_id

    monkeypatch.setattr(
        cli, "list_cached_models", lambda: [_FakeRepo("org/fresh"), _FakeRepo("org/stale")]
    )

    def fake_check(repo_id):
        if repo_id == "org/fresh":
            return UpdateStatus(repo_id=repo_id, cached_revision="a", latest_revision="a")
        return UpdateStatus(repo_id=repo_id, cached_revision="a", latest_revision="b")

    monkeypatch.setattr(cli, "check_for_update", fake_check)

    result = runner.invoke(app, ["model", "update"])

    assert result.exit_code == 0
    assert "org/fresh" in result.stdout
    assert "up to date" in result.stdout
    assert "org/stale" in result.stdout
    assert "stale" in result.stdout


def test_model_update_checks_only_given_repo_id(monkeypatch):
    """`model update REPO_ID` takes a repo_id directly, not a recipe
    handle - same layering reasoning as `model pull`."""
    seen = []

    def fake_check(repo_id):
        seen.append(repo_id)
        return UpdateStatus(repo_id=repo_id, cached_revision="a", latest_revision="a")

    monkeypatch.setattr(cli, "check_for_update", fake_check)

    result = runner.invoke(app, ["model", "update", "org/demo"])

    assert result.exit_code == 0
    assert seen == ["org/demo"]


def test_model_update_apply_repulls_stale_models_only(monkeypatch):
    class _FakeRepo:
        def __init__(self, repo_id):
            self.repo_id = repo_id

    monkeypatch.setattr(
        cli, "list_cached_models", lambda: [_FakeRepo("org/fresh"), _FakeRepo("org/stale")]
    )

    def fake_check(repo_id):
        stale = repo_id == "org/stale"
        return UpdateStatus(
            repo_id=repo_id, cached_revision="a", latest_revision="b" if stale else "a"
        )

    monkeypatch.setattr(cli, "check_for_update", fake_check)
    pulled = []
    monkeypatch.setattr(cli, "pull_model", lambda repo_id: pulled.append(repo_id))

    result = runner.invoke(app, ["model", "update", "--apply"])

    assert result.exit_code == 0
    assert pulled == ["org/stale"]
    assert "updated" in result.stdout


def test_model_update_apply_reports_interrupted_download_and_continues(monkeypatch):
    """A stale repo whose re-pull hits a transient Hub failure gets its
    own row rather than crashing the whole table - the remaining repos'
    statuses still get reported."""

    class _FakeRepo:
        def __init__(self, repo_id):
            self.repo_id = repo_id

    monkeypatch.setattr(
        cli, "list_cached_models", lambda: [_FakeRepo("org/broken"), _FakeRepo("org/fresh")]
    )

    def fake_check(repo_id):
        stale = repo_id == "org/broken"
        return UpdateStatus(
            repo_id=repo_id, cached_revision="a", latest_revision="b" if stale else "a"
        )

    monkeypatch.setattr(cli, "check_for_update", fake_check)

    def fake_pull(repo_id):
        raise HfHubHTTPError("boom")

    monkeypatch.setattr(cli, "pull_model", fake_pull)

    result = runner.invoke(app, ["model", "update", "--apply"])

    assert result.exit_code == 0
    assert "org/broken" in result.stdout
    assert "download interrupted" in result.stdout
    assert "org/fresh" in result.stdout
    assert "up to date" in result.stdout


def test_model_update_apply_reports_permission_error_and_continues(monkeypatch):
    class _FakeRepo:
        def __init__(self, repo_id):
            self.repo_id = repo_id

    monkeypatch.setattr(
        cli, "list_cached_models", lambda: [_FakeRepo("org/broken"), _FakeRepo("org/fresh")]
    )

    def fake_check(repo_id):
        stale = repo_id == "org/broken"
        return UpdateStatus(
            repo_id=repo_id, cached_revision="a", latest_revision="b" if stale else "a"
        )

    monkeypatch.setattr(cli, "check_for_update", fake_check)

    def fake_pull(repo_id):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "pull_model", fake_pull)

    result = runner.invoke(app, ["model", "update", "--apply"])

    assert result.exit_code == 0
    assert "org/broken" in result.stdout
    assert "permission denied" in result.stdout
    assert "chown" in result.output
    assert "org/fresh" in result.stdout
    assert "up to date" in result.stdout


def test_model_update_reports_not_cached_for_never_pulled_repo(monkeypatch):
    monkeypatch.setattr(
        cli,
        "check_for_update",
        lambda repo_id: UpdateStatus(repo_id=repo_id, cached_revision=None, latest_revision="a"),
    )

    result = runner.invoke(app, ["model", "update", "org/never-pulled"])

    assert result.exit_code == 0
    assert "not cached" in result.stdout


def test_model_update_hub_unreachable_gives_friendly_error(monkeypatch):
    class _FakeRepo:
        repo_id = "org/demo"

    monkeypatch.setattr(cli, "list_cached_models", lambda: [_FakeRepo()])

    def fake_check(repo_id):
        raise HfHubHTTPError("boom")

    monkeypatch.setattr(cli, "check_for_update", fake_check)

    result = runner.invoke(app, ["model", "update"])

    assert result.exit_code == 1
    assert "unreachable" in result.output


def _hardware_profile(**overrides) -> HardwareProfile:
    defaults = dict(
        gpu_name="NVIDIA A100 80GB PCIe",
        gpu_count=1,
        vram_gb_per_gpu=80.0,
        ram_gb=256.0,
        chip_family="nvidia",
        supported_quantizations=["awq", "gptq", "fp8"],
        scanned_at="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return HardwareProfile(**defaults)


def test_model_scan_defaults_use_hardware_scan(monkeypatch):
    monkeypatch.setattr(cli, "scan_hardware", lambda: _hardware_profile())
    captured = {}

    def fake_search_models(**kwargs):
        captured.update(kwargs)
        return [ModelCandidate("org/demo-7B-AWQ", "awq", 7.0, 14.0, 100, 1000, None)]

    monkeypatch.setattr(cli, "search_models", fake_search_models)

    result = runner.invoke(app, ["model", "scan"])

    assert result.exit_code == 0
    assert "org/demo-7B-AWQ" in result.stdout
    assert "14.0 GB" in result.stdout
    assert "DL TOTAL" in result.stdout
    assert "1.0k" in result.stdout  # DL TOTAL
    assert "DL 30D" in result.stdout
    assert "100" in result.stdout  # DL 30D
    assert "unknown" in result.stdout  # UPDATED, since last_modified=None
    assert "QUANT" in result.stdout  # multiple quantizations searched, so shown
    assert set(captured["quantizations"]) == {"awq", "gptq", "fp8"}
    assert captured["max_params_billion"] is None
    # NVIDIA A100 80GB, discrete (no unified-memory OS reserve): 80 * 0.85.
    assert captured["max_size_gb"] == 68.0
    assert captured["exclude_unknown_size"] is False


def test_model_scan_omits_quant_column_for_single_quantization(monkeypatch):
    def fake_search_models(**kwargs):
        return [ModelCandidate("org/demo-7B-AWQ", "awq", 7.0, 14.0, 100, 1000, None)]

    monkeypatch.setattr(cli, "search_models", fake_search_models)

    result = runner.invoke(app, ["model", "scan", "--quant", "awq", "--max-size", "40"])

    assert result.exit_code == 0
    assert "QUANT" not in result.stdout
    assert "org/demo-7B-AWQ" in result.stdout


def test_model_scan_explicit_overrides_never_touch_hardware(monkeypatch):
    def fail_if_called():
        raise AssertionError("scan_hardware should not be called when both overrides are given")

    monkeypatch.setattr(cli, "scan_hardware", fail_if_called)
    captured = {}

    def fake_search_models(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "search_models", fake_search_models)

    result = runner.invoke(
        app, ["model", "scan", "--quant", "gptq", "--max-size", "40", "--max-params", "13"]
    )

    assert result.exit_code == 0
    assert captured["quantizations"] == ["gptq"]
    assert captured["max_size_gb"] == 40.0
    assert captured["exclude_unknown_size"] is True
    assert captured["max_params_billion"] == 13.0


def test_model_scan_max_size_alone_still_needs_hardware_for_quantizations(monkeypatch):
    monkeypatch.setattr(cli, "scan_hardware", lambda: _hardware_profile())
    captured = {}

    def fake_search_models(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "search_models", fake_search_models)

    result = runner.invoke(app, ["model", "scan", "--max-size", "40"])

    assert result.exit_code == 0
    assert set(captured["quantizations"]) == {"awq", "gptq", "fp8"}
    assert captured["max_size_gb"] == 40.0
    assert captured["exclude_unknown_size"] is True


def test_model_scan_quant_alone_still_needs_hardware_for_max_size(monkeypatch):
    monkeypatch.setattr(cli, "scan_hardware", lambda: _hardware_profile())
    captured = {}

    def fake_search_models(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "search_models", fake_search_models)

    result = runner.invoke(app, ["model", "scan", "--quant", "gptq"])

    assert result.exit_code == 0
    assert captured["quantizations"] == ["gptq"]
    assert captured["max_size_gb"] == 68.0
    assert captured["exclude_unknown_size"] is False


def test_model_scan_no_supported_quantizations_gives_friendly_error(monkeypatch):
    monkeypatch.setattr(cli, "scan_hardware", lambda: _hardware_profile(supported_quantizations=[]))

    result = runner.invoke(app, ["model", "scan"])

    assert result.exit_code == 1
    assert "--quant" in result.output


def test_model_scan_hub_unreachable_gives_friendly_error(monkeypatch):
    monkeypatch.setattr(cli, "scan_hardware", lambda: _hardware_profile())

    def fake_search_models(**kwargs):
        raise HfHubHTTPError("boom")

    monkeypatch.setattr(cli, "search_models", fake_search_models)

    result = runner.invoke(app, ["model", "scan"])

    assert result.exit_code == 1
    assert "unreachable" in result.output


def test_model_scan_no_results(monkeypatch):
    monkeypatch.setattr(cli, "scan_hardware", lambda: _hardware_profile())
    monkeypatch.setattr(cli, "search_models", lambda **kwargs: [])

    result = runner.invoke(app, ["model", "scan"])

    assert result.exit_code == 0
    assert "No matching models" in result.stdout


def test_serve_verifies_cache_then_invokes_docker_compose_up(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    assert captured["command"][:3] == ["docker", "compose", "-f"]
    assert captured["command"][-3:] == ["up", "-d", "demo"]


def test_serve_uses_recipes_own_compose_folder_and_project(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    command = captured["command"]
    assert command[command.index("-f") + 1] == str(tmp_path / "demo" / "compose.yaml")
    assert command[command.index("-p") + 1] == "fllame-demo"


def test_serve_never_uses_build_flag(tmp_path: Path, monkeypatch):
    """`docker compose up` never gets Docker's own image-build flag -
    `recipe build` already built/validated the image, `serve` only ever
    runs what's already there."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    assert "--build" not in captured["command"]


def test_serve_requires_compose_already_built(tmp_path: Path, monkeypatch):
    """`serve` never writes or regenerates `compose.yaml` itself -
    that's `recipe build`'s job alone."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    called = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command: called.append("docker"))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 1
    assert "fllame recipe build" in result.output
    assert called == []
    assert not (tmp_path / "demo" / "compose.yaml").exists()


def test_recipe_build_with_preinstall_writes_dockerfile_and_builds_local_image(
    tmp_path: Path, monkeypatch
):
    """A recipe with `preinstall` gets a `Dockerfile` next to
    compose.yaml, and compose.yaml's `image:` points at the local tag
    that Dockerfile is built into - not the base image."""
    _isolate(tmp_path, monkeypatch)
    (tmp_path / "demo").mkdir(parents=True)
    (tmp_path / "demo" / "recipe.yaml").write_text(
        "image: vllm/vllm-openai:v0.27.1\n"
        "preinstall:\n"
        "- pip install -U transformers\n"
        "command: vllm serve org/demo\n"
    )
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    dockerfile_text = (tmp_path / "demo" / "Dockerfile").read_text()
    assert dockerfile_text == "FROM vllm/vllm-openai:v0.27.1\nRUN pip install -U transformers\n"
    # `preinstall` is now a Dockerfile-build-time concern - it never
    # appears embedded in compose.yaml's command/entrypoint.
    compose_text = (tmp_path / "demo" / "compose.yaml").read_text()
    assert "pip install -U transformers" not in compose_text
    assert "image: fllame-demo:latest" in compose_text
    assert captured["command"] == [
        "docker",
        "build",
        "-t",
        "fllame-demo:latest",
        str(tmp_path / "demo"),
    ]


def test_recipe_build_without_preinstall_never_writes_a_dockerfile(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    assert not (tmp_path / "demo" / "Dockerfile").exists()


def test_recipe_build_without_preinstall_validates_image_via_compose_pull(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    assert captured["command"][:3] == ["docker", "compose", "-f"]
    assert captured["command"][-1] == "pull"
    compose_text = (tmp_path / "demo" / "compose.yaml").read_text()
    assert "image: vllm/vllm-openai:v0.27.1" in compose_text


def test_recipe_build_dockerfile_left_untouched_when_recipe_has_no_preinstall(
    tmp_path: Path, monkeypatch
):
    """`Dockerfile` is a real, permanent, hand-editable artifact once a
    recipe has one - a later build of a recipe with no `preinstall`
    doesn't delete it out from under the user."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    handle_folder = tmp_path / "demo"
    (handle_folder / "Dockerfile").write_text("FROM img\nRUN echo hand-edited\n")
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    assert (handle_folder / "Dockerfile").read_text() == "FROM img\nRUN echo hand-edited\n"


def test_recipe_build_docker_build_failure_gives_friendly_error(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    (tmp_path / "demo").mkdir(parents=True)
    (tmp_path / "demo" / "recipe.yaml").write_text(
        "image: vllm/vllm-openai:v0.27.1\n"
        "preinstall:\n"
        "- pip install -U transformers\n"
        "command: vllm serve org/demo\n"
    )
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    class _FailedProcess:
        returncode = 1

    monkeypatch.setattr(cli.subprocess, "run", lambda command: _FailedProcess())

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 1
    assert "docker build" in result.output
    assert "fllame recipe build demo" in result.output


def test_recipe_build_docker_compose_pull_failure_gives_friendly_error(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    class _FailedProcess:
        returncode = 1

    monkeypatch.setattr(cli.subprocess, "run", lambda command: _FailedProcess())

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 1
    assert "docker compose pull" in result.output
    assert "fllame recipe build demo" in result.output


def test_serve_runs_hand_edited_compose_without_warning(tmp_path: Path, monkeypatch):
    """fllame doesn't track what it last wrote any more - a hand-edited
    compose.yaml (or Dockerfile) is trusted outright, no prompt, no
    hash comparison. Single-admin tool: a hand edit means knowing what
    you're doing."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    compose_path = tmp_path / "demo" / "compose.yaml"
    compose_path.write_text(compose_path.read_text() + "# hand-edited\n")
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run({}))

    result = runner.invoke(app, ["serve", "demo"])  # no stdin needed - never prompts

    assert result.exit_code == 0
    assert "hand-edited" not in result.output
    assert "no longer matches" not in result.output


def test_config_set_default_image_pulls_once_on_success(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.26.0"])
    _write_compose_with_image(tmp_path, "demo", "vllm/vllm-openai:v0.26.0")
    pulls = []

    def fake_run(command):
        pulls.append(command)
        return _FakeCompletedProcess()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = runner.invoke(
        app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"], input="y\n"
    )

    assert result.exit_code == 0
    assert pulls == [["docker", "pull", "vllm/vllm-openai:v0.27.1"]]


def test_config_set_default_image_pull_failure_changes_nothing(tmp_path: Path, monkeypatch):
    """The pull is a precondition, not an afterthought - a failure
    leaves both the persisted default and every file untouched, and
    never even reaches the affected-file search or its prompt."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.26.0"])
    compose_path = _write_compose_with_image(tmp_path, "demo", "vllm/vllm-openai:v0.26.0")

    class _FailedProcess:
        returncode = 1

    monkeypatch.setattr(cli.subprocess, "run", lambda command: _FailedProcess())

    result = runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"])

    assert result.exit_code == 1
    assert "Failed to pull image" in result.output
    assert "file(s) still use" not in result.output
    show_result = runner.invoke(app, ["config", "show"])
    assert "default_image: vllm/vllm-openai:v0.26.0" in show_result.stdout
    assert "image: vllm/vllm-openai:v0.26.0" in compose_path.read_text()


def test_config_set_default_image_pulls_even_with_no_recipes(tmp_path: Path, monkeypatch):
    """The pull validates the image itself, independent of whether any
    recipe would even be affected by the change."""
    _isolate(tmp_path, monkeypatch)
    pulls = []
    monkeypatch.setattr(
        cli.subprocess, "run", lambda command: pulls.append(command) or _FakeCompletedProcess()
    )

    result = runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:v0.27.1"])

    assert result.exit_code == 0
    assert pulls == [["docker", "pull", "vllm/vllm-openai:v0.27.1"]]


def test_serve_always_runs_detached(tmp_path: Path, monkeypatch):
    """There's no foreground mode any more - `serve` always runs `up
    -d`, since a well-defined "did it start OK" exit code is what lets
    it clear the drift markers below right after a successful start."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    assert captured["command"][-3:] == ["up", "-d", "demo"]


def test_serve_has_no_detach_option_any_more():
    result = runner.invoke(app, ["serve", "--help"])

    assert "--detach" not in result.output


def test_serve_sets_container_name_on_the_compose_service(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)

    compose_text = (tmp_path / "demo" / "compose.yaml").read_text()

    assert "container_name: fllame-demo" in compose_text


def test_serve_prints_status_and_docker_logs_hints_on_success(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run({}))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    assert "docker logs -f fllame-demo" in result.output
    assert "fllame status demo --watch" in result.output
    assert "can take several minutes" in result.output


def test_serve_omits_docker_logs_hint_on_failure(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)

    class _FailedProcess:
        returncode = 1

    monkeypatch.setattr(cli.subprocess, "run", lambda command: _FailedProcess())

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 1
    assert "docker logs" not in result.output
    assert "fllame status" not in result.output


def test_serve_unknown_handle_never_checks_cache_or_calls_docker(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: called.append("cache-check"))
    monkeypatch.setattr(cli.subprocess, "run", lambda command: called.append("docker"))

    result = runner.invoke(app, ["serve", "nope"])

    assert result.exit_code == 1
    assert called == []


def test_recipe_build_container_always_sets_hf_hub_offline(tmp_path: Path, monkeypatch):
    """HF_HUB_OFFLINE=1 is unconditional (VllmServingBackend bakes it
    into every generated service) - the model is always already fully
    downloaded by the time the container runs, so vLLM has no
    legitimate need to reach the Hub itself."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    compose_text = (config.recipe_dir("demo") / "compose.yaml").read_text()
    assert "HF_HUB_OFFLINE" in compose_text


def test_recipe_build_sets_gpu_memory_utilization_by_default(tmp_path: Path, monkeypatch):
    """Left to vLLM's own default, a recipe with no explicit
    `--gpu-memory-utilization` can reserve the whole GPU - starving the
    rest of the box on a unified-memory machine. `recipe build` always
    bakes in a safe value unless the recipe's own command already sets
    one (see `fllame/backends/vllm.py`)."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    compose_text = (config.recipe_dir("demo") / "compose.yaml").read_text()
    assert "--gpu-memory-utilization" in compose_text
    assert "0.92" in compose_text


def test_recipe_build_default_gpu_memory_utilization_same_regardless_of_chip_family(
    tmp_path: Path, monkeypatch
):
    """The default is a flat, configured value now - not computed from
    hardware at all, so unified memory and a discrete GPU get the exact
    same number."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    monkeypatch.setattr(
        cli,
        "scan_hardware",
        lambda: HardwareProfile(
            gpu_name="NVIDIA GB10",
            gpu_count=1,
            vram_gb_per_gpu=None,
            ram_gb=32.0,
            chip_family="grace_blackwell",
        ),
    )

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    compose_text = (config.recipe_dir("demo") / "compose.yaml").read_text()
    assert "--gpu-memory-utilization" in compose_text
    assert "0.92" in compose_text


def test_recipe_build_uses_configured_default_gpu_memory_utilization(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    runner.invoke(app, ["config", "set-default-gpu-memory-utilization", "0.6"])

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    compose_text = (config.recipe_dir("demo") / "compose.yaml").read_text()
    assert "0.60" in compose_text
    assert "0.92" not in compose_text


def test_recipe_build_respects_recipes_own_gpu_memory_utilization(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    directory = tmp_path / "demo"
    directory.mkdir(parents=True)
    (directory / "recipe.yaml").write_text(
        "image: vllm/vllm-openai:v0.27.1\n"
        "command: vllm serve org/demo --gpu-memory-utilization 0.6\n"
    )
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    compose_text = (config.recipe_dir("demo") / "compose.yaml").read_text()
    assert compose_text.count("--gpu-memory-utilization") == 1
    assert "0.6" in compose_text
    assert "0.92" not in compose_text


def test_recipe_build_never_validates_an_explicit_gpu_memory_utilization(
    tmp_path: Path, monkeypatch
):
    """The recipe wins outright - `recipe build` writes whatever value
    it's given verbatim and never second-guesses it, however
    implausible; `serve`'s own check is the only thing that still
    catches an obviously-wrong value."""
    _isolate(tmp_path, monkeypatch)
    directory = tmp_path / "demo"
    directory.mkdir(parents=True)
    (directory / "recipe.yaml").write_text(
        "image: vllm/vllm-openai:v0.27.1\n"
        "command: vllm serve org/demo --gpu-memory-utilization 0.99\n"
    )
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    compose_text = (config.recipe_dir("demo") / "compose.yaml").read_text()
    assert "0.99" in compose_text


def test_config_set_default_gpu_memory_utilization(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    set_result = runner.invoke(app, ["config", "set-default-gpu-memory-utilization", "0.85"])
    show_result = runner.invoke(app, ["config", "show"])

    assert set_result.exit_code == 0
    assert show_result.exit_code == 0
    assert "default_gpu_memory_utilization: 0.85" in show_result.stdout


def test_config_set_default_gpu_memory_utilization_rejects_out_of_range(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)

    assert runner.invoke(app, ["config", "set-default-gpu-memory-utilization", "0"]).exit_code == 1
    result = runner.invoke(app, ["config", "set-default-gpu-memory-utilization", "--", "1.5"])
    assert result.exit_code == 1


def test_config_set_default_gpu_memory_utilization_allows_one(tmp_path: Path, monkeypatch):
    """1.0 is the mathematical ceiling, not a policy opinion - allowed,
    unlike the old configurable-ceiling design."""
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["config", "set-default-gpu-memory-utilization", "1"])

    assert result.exit_code == 0


def _write_compose_with_gpu_memory_utilization(tmp_path: Path, handle: str, value: float) -> Path:
    """A minimal, already-generated-looking compose.yaml, with a hand
    edit (a custom `shm_size:`) alongside the baked-in
    --gpu-memory-utilization value - used to verify that batch updates
    touch only that one value."""
    directory = tmp_path / handle
    directory.mkdir(parents=True, exist_ok=True)
    compose_path = directory / "compose.yaml"
    compose_path.write_text(
        f"services:\n  {handle}:\n    command:\n    - org/{handle}\n"
        f"    - --gpu-memory-utilization\n    - '{value:.2f}'\n    shm_size: 2gb\n"
    )
    return compose_path


def test_config_set_default_gpu_memory_utilization_skips_batch_update_with_no_compose_files(
    tmp_path: Path, monkeypatch
):
    """No recipe has been built yet - nothing to search, nothing to
    prompt about."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    result = runner.invoke(app, ["config", "set-default-gpu-memory-utilization", "0.6"])

    assert result.exit_code == 0
    assert "compose.yaml" not in result.output


def test_config_set_default_gpu_memory_utilization_finds_recipes_still_on_the_builtin_0_92(
    tmp_path: Path, monkeypatch
):
    """0.92 is a real, concrete default even before anyone ever runs
    `config set-default-gpu-memory-utilization` - a fresh recipe built
    against it is just as much "using the previous default" as one
    built after an explicit change."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    compose_path = _write_compose_with_gpu_memory_utilization(tmp_path, "demo", 0.92)

    result = runner.invoke(
        app, ["config", "set-default-gpu-memory-utilization", "0.6"], input="y\n"
    )

    assert result.exit_code == 0
    assert "- '0.60'" in compose_path.read_text()


def test_config_set_default_gpu_memory_utilization_offers_batch_update_and_applies_on_confirm(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    compose_path = _write_compose_with_gpu_memory_utilization(tmp_path, "demo", 0.92)

    result = runner.invoke(
        app, ["config", "set-default-gpu-memory-utilization", "0.6"], input="y\n"
    )

    assert result.exit_code == 0
    assert str(compose_path) in result.output
    text = compose_path.read_text()
    assert "- '0.60'" in text
    # Nothing else in the hand-edited file was touched.
    assert "shm_size: 2gb" in text


def test_config_set_default_gpu_memory_utilization_batch_update_declined_leaves_files_untouched(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    compose_path = _write_compose_with_gpu_memory_utilization(tmp_path, "demo", 0.92)

    result = runner.invoke(
        app, ["config", "set-default-gpu-memory-utilization", "0.6"], input="n\n"
    )

    assert result.exit_code == 0
    assert "- '0.92'" in compose_path.read_text()


def test_config_set_default_gpu_memory_utilization_never_touches_a_different_value(
    tmp_path: Path, monkeypatch
):
    """A compose.yaml whose baked-in value doesn't literally match the
    previous default - the recipe's own command set one explicitly, or
    a hand edit - is never listed or replaced, confirmation or not."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    compose_path = _write_compose_with_gpu_memory_utilization(tmp_path, "demo", 0.7)

    result = runner.invoke(app, ["config", "set-default-gpu-memory-utilization", "0.6"])

    assert result.exit_code == 0
    assert "compose.yaml" not in result.output
    assert "- '0.70'" in compose_path.read_text()


def test_config_set_default_gpu_memory_utilization_confirm_prompt_explains_text_replace(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    _write_compose_with_gpu_memory_utilization(tmp_path, "demo", 0.92)

    result = runner.invoke(
        app, ["config", "set-default-gpu-memory-utilization", "0.6"], input="n\n"
    )

    assert "text replace" in result.output
    assert "hand edits are kept as-is" in result.output


def test_serve_refuses_when_compose_yaml_has_no_gpu_memory_utilization(tmp_path: Path, monkeypatch):
    """The exact scenario this check exists for: a compose.yaml that
    somehow ended up without --gpu-memory-utilization (hand-edited, or
    left over from before fllame always injected one) must never be
    served - regardless of -y."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    compose_path = config.recipe_dir("demo") / "compose.yaml"
    compose_path.write_text(compose_path.read_text().replace("--gpu-memory-utilization", "--foo"))
    docker_calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command: docker_calls.append(command))

    result = runner.invoke(app, ["serve", "demo", "--yes"])

    assert result.exit_code == 1
    assert "no --gpu-memory-utilization" in result.output
    assert docker_calls == []


def test_serve_refuses_when_compose_yaml_gpu_memory_utilization_too_high(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    compose_path = config.recipe_dir("demo") / "compose.yaml"
    compose_path.write_text(compose_path.read_text().replace("0.92", "1.5"))
    docker_calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command: docker_calls.append(command))

    result = runner.invoke(app, ["serve", "demo", "--yes"])

    assert result.exit_code == 1
    assert "1.5" in result.output
    assert docker_calls == []


def test_serve_proceeds_when_compose_yaml_gpu_memory_utilization_is_safe(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    assert captured["command"][-3:] == ["up", "-d", "demo"]


def test_recipe_build_warns_on_tensor_parallel_size_mismatch(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    directory = tmp_path / "demo"
    directory.mkdir(parents=True)
    (directory / "recipe.yaml").write_text(
        "image: vllm/vllm-openai:v0.27.1\n"
        "command: vllm serve org/demo --tensor-parallel-size 4\n"
    )
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    monkeypatch.setattr(
        cli,
        "scan_hardware",
        lambda: HardwareProfile(
            gpu_name="NVIDIA A100 80GB PCIe",
            gpu_count=1,
            vram_gb_per_gpu=80.0,
            ram_gb=256.0,
            chip_family="nvidia",
        ),
    )

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    assert "--tensor-parallel-size" in result.output


def test_recipe_build_no_tensor_parallel_size_warning_when_matching(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    directory = tmp_path / "demo"
    directory.mkdir(parents=True)
    (directory / "recipe.yaml").write_text(
        "image: vllm/vllm-openai:v0.27.1\n"
        "command: vllm serve org/demo --tensor-parallel-size 1\n"
    )
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    monkeypatch.setattr(
        cli,
        "scan_hardware",
        lambda: HardwareProfile(
            gpu_name="NVIDIA A100 80GB PCIe",
            gpu_count=1,
            vram_gb_per_gpu=80.0,
            ram_gb=256.0,
            chip_family="nvidia",
        ),
    )

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    assert "--tensor-parallel-size" not in result.output


def test_recipe_build_no_tensor_parallel_size_warning_when_no_gpu_detected(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    # _isolate's own default already scans no hardware at all.

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    assert "--tensor-parallel-size" not in result.output


def test_recipe_build_cache_location_unchanged_never_warns(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    monkeypatch.setattr(config, "hf_cache_dir", lambda: Path("/cache"))

    # First build generates compose.yaml against this same cache location.
    assert runner.invoke(app, ["recipe", "build", "demo"]).exit_code == 0

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    assert "cache location has changed" not in result.output


def test_recipe_build_cache_location_changed_aborts_when_declined(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    monkeypatch.setattr(config, "hf_cache_dir", lambda: Path("/cache/old"))
    assert runner.invoke(app, ["recipe", "build", "demo"]).exit_code == 0

    monkeypatch.setattr(config, "hf_cache_dir", lambda: Path("/cache/new"))
    result = runner.invoke(app, ["recipe", "build", "demo"], input="n\n")

    assert result.exit_code == 1
    assert "cache location has changed" in result.output
    assert "/cache/old" in (tmp_path / "demo" / "compose.yaml").read_text()


def test_recipe_build_cache_location_changed_continues_when_confirmed(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    monkeypatch.setattr(config, "hf_cache_dir", lambda: Path("/cache/old"))
    assert runner.invoke(app, ["recipe", "build", "demo"]).exit_code == 0

    monkeypatch.setattr(config, "hf_cache_dir", lambda: Path("/cache/new"))
    result = runner.invoke(app, ["recipe", "build", "demo"], input="y\n")

    assert result.exit_code == 0
    assert "cache location has changed" in result.output
    assert "/cache/new" in (tmp_path / "demo" / "compose.yaml").read_text()


def test_recipe_build_yes_flag_skips_cache_location_confirmation_but_still_warns(
    tmp_path: Path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    monkeypatch.setattr(config, "hf_cache_dir", lambda: Path("/cache/old"))
    assert runner.invoke(app, ["recipe", "build", "demo"]).exit_code == 0

    monkeypatch.setattr(config, "hf_cache_dir", lambda: Path("/cache/new"))
    result = runner.invoke(app, ["recipe", "build", "demo", "--yes"])  # no stdin needed

    assert result.exit_code == 0
    assert "cache location has changed" in result.output
    assert "/cache/new" in (tmp_path / "demo" / "compose.yaml").read_text()


_GPU_WITH_BUDGET = HardwareProfile(
    gpu_name="NVIDIA A100 80GB PCIe",
    gpu_count=1,
    vram_gb_per_gpu=80.0,
    ram_gb=256.0,
    chip_family="nvidia",
    supported_quantizations=["awq", "gptq", "fp8"],
    scanned_at="2026-01-01T00:00:00+00:00",
)


def test_serve_vram_warning_aborts_when_declined(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "scan_hardware", lambda: _GPU_WITH_BUDGET)
    monkeypatch.setattr(
        cli, "local_estimate_vram_gb", lambda repo_id: 100.0
    )  # over the 68 GB budget
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    called = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command: called.append("docker"))

    result = runner.invoke(app, ["serve", "demo"], input="n\n")

    assert result.exit_code == 1
    assert "estimated at 100.0 GB" in result.output
    assert called == []


def test_serve_vram_warning_continues_when_confirmed(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "scan_hardware", lambda: _GPU_WITH_BUDGET)
    monkeypatch.setattr(cli, "local_estimate_vram_gb", lambda repo_id: 100.0)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run({}))

    result = runner.invoke(app, ["serve", "demo"], input="y\n")

    assert result.exit_code == 0
    assert "estimated at 100.0 GB" in result.output


def test_serve_yes_flag_skips_confirmation_but_still_warns(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "scan_hardware", lambda: _GPU_WITH_BUDGET)
    monkeypatch.setattr(cli, "local_estimate_vram_gb", lambda repo_id: 100.0)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run({}))

    result = runner.invoke(app, ["serve", "demo", "--yes"])  # no stdin needed

    assert result.exit_code == 0
    assert "estimated at 100.0 GB" in result.output


def test_serve_vram_check_silent_when_estimate_fits_budget(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "scan_hardware", lambda: _GPU_WITH_BUDGET)
    monkeypatch.setattr(cli, "local_estimate_vram_gb", lambda repo_id: 10.0)  # well under budget
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run({}))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    assert "warning" not in result.output


def test_serve_vram_check_silent_when_estimate_unknown(tmp_path: Path, monkeypatch):
    """No cached `.safetensors` files to measure yet (shouldn't happen
    right after a successful pull, but a GGUF-only download would still
    hit this) come back as `None` - never treated as a positive "it
    fits" signal, but also never a warning without a real number."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "scan_hardware", lambda: _GPU_WITH_BUDGET)
    monkeypatch.setattr(cli, "local_estimate_vram_gb", lambda repo_id: None)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run({}))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    assert "warning" not in result.output


def test_serve_cache_miss_gives_friendly_error(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: False)
    called = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command: called.append("docker"))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 1
    assert "fllame model pull" in result.output
    assert called == []


def test_recipe_build_fails_when_model_not_cached(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: False)

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 1
    assert "fllame model pull" in result.output
    assert not (tmp_path / "demo" / "compose.yaml").exists()


def test_recipe_build_writes_compose_when_model_cached(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    assert (tmp_path / "demo" / "compose.yaml").is_file()


def test_recipe_build_unknown_handle(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["recipe", "build", "nope"])

    assert result.exit_code == 1


def test_recipe_add_plain_neither_pulls_nor_builds(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pulled = []
    monkeypatch.setattr(
        cli, "pull_model", lambda repo_id, offline=False: pulled.append((repo_id, offline))
    )

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "img:v1"],
        input=_dialogue_input("", "", "vllm serve org/demo"),
    )

    assert result.exit_code == 0
    assert pulled == []
    assert not (tmp_path / "demo" / "compose.yaml").exists()


def test_recipe_add_pull_downloads_the_model(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pulled = []
    monkeypatch.setattr(
        cli, "pull_model", lambda repo_id, offline=False: pulled.append((repo_id, offline))
    )

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "img:v1", "--pull"],
        input=_dialogue_input("", "", "vllm serve org/demo"),
    )

    assert result.exit_code == 0
    assert pulled == [("org/demo", False)]


def test_recipe_add_pull_hub_error_gives_friendly_message(tmp_path: Path, monkeypatch):
    """A transient Hub failure during `recipe add --pull`'s download
    must not crash with a raw traceback - same fix as `model pull`."""
    _isolate(tmp_path, monkeypatch)

    def raise_error(repo_id, offline=False):
        raise HfHubHTTPError("boom")

    monkeypatch.setattr(cli, "pull_model", raise_error)

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "img:v1", "--pull"],
        input=_dialogue_input("", "", "vllm serve org/demo"),
    )

    assert result.exit_code == 1
    assert "resume rather than start over" in result.output
    # The recipe itself is still saved even though the pull step failed.
    assert (tmp_path / "demo" / "recipe.yaml").is_file()


def test_recipe_add_pull_permission_error_gives_friendly_message(tmp_path: Path, monkeypatch):
    """Same fix as `model pull` - a permission error writing to the
    shared HF cache must not crash with a raw traceback."""
    _isolate(tmp_path, monkeypatch)

    def raise_error(repo_id, offline=False):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "pull_model", raise_error)

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "img:v1", "--pull"],
        input=_dialogue_input("", "", "vllm serve org/demo"),
    )

    assert result.exit_code == 1
    assert "chown" in result.output
    # The recipe itself is still saved even though the pull step failed.
    assert (tmp_path / "demo" / "recipe.yaml").is_file()


def test_recipe_add_build_without_pull_fails_when_not_cached(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: False)

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "img:v1", "--build"],
        input=_dialogue_input("", "", "vllm serve org/demo"),
    )

    assert result.exit_code == 1
    assert "fllame model pull" in result.output
    # The recipe itself is still saved even though the build step failed.
    assert (tmp_path / "demo" / "recipe.yaml").is_file()
    assert not (tmp_path / "demo" / "compose.yaml").exists()


def test_recipe_add_pull_and_build_together(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pulled = []
    monkeypatch.setattr(
        cli, "pull_model", lambda repo_id, offline=False: pulled.append((repo_id, offline))
    )
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "img:v1", "--pull", "--build"],
        input=_dialogue_input("", "", "vllm serve org/demo"),
    )

    assert result.exit_code == 0
    # --pull's own download; --build's local cache-presence check.
    assert pulled == [("org/demo", False)]
    assert (tmp_path / "demo" / "compose.yaml").is_file()


def test_status_invokes_docker_compose_ps_with_json_format(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert captured["command"][-4:] == ["ps", "--all", "--format", "json"]


def test_status_prints_one_table_across_every_recipe(tmp_path: Path, monkeypatch):
    """One shared table, RECIPE_ID first - not a separate `docker compose
    ps` table per recipe. COMMAND, CREATED, and SERVICE (always the
    same as RECIPE) are dropped. A recipe with no container at all
    still gets a row, filled in from compose.yaml instead of docker."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path, handle="demo-a")
    _write_recipe(tmp_path, handle="demo-b")
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path, handle="demo-a")
    _write_compose(tmp_path, handle="demo-b")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        _fake_compose_ps_json(
            {
                "demo-a": [
                    {
                        "Name": "fllame-demo-a",
                        "Image": "vllm/vllm-openai:v0.27.1",
                        "Command": "vllm serve org/demo-a",
                        "Service": "demo-a",
                        "Created": "2024-01-01T00:00:00Z",
                        "Status": "Up 5 minutes",
                        "Publishers": [
                            {
                                "URL": "0.0.0.0",
                                "TargetPort": 8000,
                                "PublishedPort": 8000,
                                "Protocol": "tcp",
                            },
                            {
                                "URL": "::",
                                "TargetPort": 8000,
                                "PublishedPort": 8000,
                                "Protocol": "tcp",
                            },
                        ],
                    }
                ],
                "demo-b": [],
            }
        ),
    )

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    header_lines = [line for line in lines if line.startswith("RECIPE")]
    assert len(header_lines) == 1
    assert header_lines[0].split() == ["RECIPE_ID", "NAME", "IMAGE", "STATUS", "PORTS"]
    assert "COMMAND" not in result.output
    assert "CREATED" not in result.output
    # demo-a: a live container, straight from docker.
    assert "demo-a" in result.output
    assert "fllame-demo-a" in result.output
    assert "Up 5 minutes" in result.output
    assert "0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp" in result.output
    # demo-b: built, but no container yet - filled in from compose.yaml instead.
    assert "demo-b" in result.output
    assert "fllame-demo-b" in result.output
    assert "Never started" in result.output
    assert "vllm/vllm-openai:v0.27.1" in result.output
    assert "8000:8000" in result.output


def test_status_no_recipes(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "No recipes found" in result.stdout


def test_status_skips_invalid_recipe_with_warning(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path, handle="good")
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path, handle="good")
    (tmp_path / "bad").mkdir(parents=True)
    (tmp_path / "bad" / "recipe.yaml").write_text("image: img\n")  # missing command
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        _fake_compose_ps_json({"good": [{"Name": "fllame-good", "Service": "good"}]}),
    )

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "warning" in result.output
    assert "fllame-good" in result.output
    # The invalid recipe is only ever named in the warning, never in the table.
    table_lines = [line for line in result.output.splitlines() if not line.startswith("warning")]
    assert "bad" not in "\n".join(table_lines)


def test_status_reports_unbuilt_recipe_without_calling_docker(tmp_path: Path, monkeypatch):
    """`status` never regenerates `compose.yaml` - a recipe that hasn't
    been built yet still gets a row, filled in from recipe.yaml alone,
    without ever invoking docker."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    called = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: called.append("docker"))

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "demo" in result.output
    assert "Not built" in result.output
    assert "vllm/vllm-openai:v0.27.1" in result.output
    assert "8000:8000" in result.output
    assert called == []


def test_status_falls_back_to_the_configured_port_for_a_stopped_container(
    tmp_path: Path, monkeypatch
):
    """A stopped container reports no live port bindings at all (empty
    `Publishers`), unlike a running one - PORTS should still show the
    configured port instead of going blank."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        _fake_compose_ps_json(
            {
                "demo": [
                    {
                        "Name": "fllame-demo",
                        "Image": "vllm/vllm-openai:v0.27.1",
                        "Status": "Exited (0) 12 days ago",
                        "Publishers": [],
                    }
                ]
            }
        ),
    )

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Exited (0) 12 days ago" in result.output
    assert "8000:8000" in result.output


def test_status_still_shows_a_built_recipe_when_docker_compose_ps_fails(
    tmp_path: Path, monkeypatch
):
    """A recipe stays in the table even when `docker compose ps` itself
    errors (e.g. the daemon is unreachable) - dropping it would make a
    built recipe disappear while an unbuilt one still shows, which is
    backwards. STATUS says plainly that it couldn't be confirmed."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)

    class _FailedProcess:
        returncode = 1
        stdout = ""
        stderr = "Cannot connect to the Docker daemon\n"

    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: _FailedProcess())

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 1
    assert "Cannot connect to the Docker daemon" in result.output
    assert "demo" in result.output
    assert "Unknown" in result.output
    assert "vllm/vllm-openai:v0.27.1" in result.output


def test_stop_invokes_docker_compose_stop(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["stop", "demo"])

    assert result.exit_code == 0
    assert captured["command"][-2:] == ["stop", "demo"]


def test_stop_requires_compose_already_built(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    called = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command: called.append("docker"))

    result = runner.invoke(app, ["stop", "demo"])

    assert result.exit_code == 1
    assert "fllame recipe build" in result.output
    assert called == []


def test_stop_without_recipe_id_stops_every_running_recipe(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    for handle in ("demo-a", "demo-b", "demo-c"):
        _write_recipe(tmp_path, handle=handle)
        _write_compose(tmp_path, handle=handle)
    _write_recipe(tmp_path, handle="unbuilt")
    states = {"demo-a": "running", "demo-b": "exited", "demo-c": "running"}
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[-2:] == ["--format", "json"]:
            handle = Path(command[command.index("-f") + 1]).parent.name
            return _FakeSubprocessResult(stdout=json.dumps([{"State": states[handle]}]))
        return _FakeCompletedProcess()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 0
    stops = [command[-2:] for command in commands if "stop" in command]
    assert stops == [["stop", "demo-a"], ["stop", "demo-c"]]


def test_stop_without_recipe_id_reports_nothing_running(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", _fake_compose_ps_json({"demo": []}))

    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 0
    assert "Nothing is running." in result.output


def test_docker_not_found_gives_friendly_error(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)

    def raise_not_found(command, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(cli.subprocess, "run", raise_not_found)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 1
    assert "docker" in result.output.lower()


def _write_vram_recipe(tmp_path: Path, command: str, handle: str = "demo") -> None:
    directory = tmp_path / handle
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "recipe.yaml").write_text(f"command: {command}\n")


def _capture_vram_estimate(monkeypatch) -> dict:
    captured = {}

    def fake_estimate_vram(repo_id, **kwargs):
        captured.update(repo_id=repo_id, **kwargs)
        return VramEstimate(
            parts=[
                VramPart("Weights", 20.0, "cached .safetensors files"),
                VramPart("KV cache", 8.0, "the KV formula"),
            ],
            notes=[],
        )

    monkeypatch.setattr(cli, "estimate_vram", fake_estimate_vram)
    return captured


def test_recipe_vram_prints_one_number_from_the_recipes_flags(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_vram_recipe(
        tmp_path,
        "vllm serve org/demo --max-model-len 262144 --max-num-seqs 8 --kv-cache-dtype fp8 "
        "--tensor-parallel-size 2",
    )
    captured = _capture_vram_estimate(monkeypatch)

    result = runner.invoke(app, ["recipe", "vram", "demo"])

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "28.0 GB"
    assert captured == {
        "repo_id": "org/demo",
        "max_model_len": 262144,
        "max_num_seqs": 8,
        "kv_cache_dtype": "fp8",
        "tensor_parallel_size": 2,
    }


def test_recipe_vram_options_override_the_recipe(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_vram_recipe(tmp_path, "vllm serve org/demo --max-model-len 262144 --max-num-seqs 8")
    captured = _capture_vram_estimate(monkeypatch)

    result = runner.invoke(
        app, ["recipe", "vram", "demo", "--max-model-len", "32K", "--max-num-seqs", "1"]
    )

    assert result.exit_code == 0, result.output
    assert captured["max_model_len"] == 32768
    assert captured["max_num_seqs"] == 1
    assert captured["kv_cache_dtype"] == "auto"


def test_recipe_vram_requires_both_limits_without_inventing_defaults(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_vram_recipe(tmp_path, "vllm serve org/demo")
    _capture_vram_estimate(monkeypatch)

    neither = runner.invoke(app, ["recipe", "vram", "demo"])
    one = runner.invoke(app, ["recipe", "vram", "demo", "--max-model-len", "32768"])

    assert neither.exit_code == 1
    assert "--max-model-len and --max-num-seqs not found in recipe 'demo'" in neither.output
    assert one.exit_code == 1
    assert "--max-num-seqs not found" in one.output
    assert "--max-model-len" not in one.output.split("not found")[0]


def test_recipe_vram_details_shows_each_part_and_its_formula(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_vram_recipe(tmp_path, "vllm serve org/demo --max-model-len 32768 --max-num-seqs 1")
    _capture_vram_estimate(monkeypatch)

    result = runner.invoke(app, ["recipe", "vram", "demo", "--details"])

    assert result.exit_code == 0, result.output
    assert "Weights:" in result.stdout and "20.0 GB" in result.stdout
    assert "the KV formula" in result.stdout
    assert "Total:" in result.stdout and "28.0 GB" in result.stdout


def test_recipe_vram_reports_an_unpulled_model(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_vram_recipe(tmp_path, "vllm serve org/demo --max-model-len 32768 --max-num-seqs 1")
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: False)
    monkeypatch.setattr("fllame.models.vram.is_model_cached", lambda repo_id: False)

    result = runner.invoke(app, ["recipe", "vram", "demo"])

    assert result.exit_code == 1
    assert "'org/demo' is not pulled" in result.output


def test_recipe_vram_rejects_a_malformed_token_count(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_vram_recipe(tmp_path, "vllm serve org/demo --max-model-len lots --max-num-seqs 1")
    _capture_vram_estimate(monkeypatch)

    result = runner.invoke(app, ["recipe", "vram", "demo"])

    assert result.exit_code == 1
    assert "--max-model-len must be a whole number of tokens" in result.output


_RUNNING = [{"Service": "demo", "State": "running", "Name": "fllame-demo"}]


def _fake_bench_docker(
    commands: list,
    *,
    containers=_RUNNING,
    served=None,
    probe_ok=True,
    bench_returncode=0,
):
    served = served or {"id": "org/demo", "max_model_len": 32768}

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[-2:] == ["--format", "json"]:
            return _FakeSubprocessResult(stdout=json.dumps(containers))
        if "exec" not in command:
            return _FakeCompletedProcess()
        inner = command[command.index("demo", command.index("exec")) + 1 :]
        if inner[0] == "python3":
            stdout = json.dumps({"data": [served]}) if probe_ok else ""
            return _FakeSubprocessResult(stdout=stdout, returncode=0 if probe_ok else 1)
        if inner[:2] == ["vllm", "--version"]:
            return _FakeSubprocessResult(stdout="0.27.1\n")
        if inner[:3] == ["vllm", "bench", "serve"]:
            return _FakeSubprocessResult(stdout="bench output", returncode=bench_returncode)
        if inner[0] == "cat":
            concurrency = int(Path(inner[1]).stem[1:])
            return _FakeSubprocessResult(
                stdout=json.dumps({"completed": concurrency * 2, "request_throughput": 1.5})
            )
        return _FakeCompletedProcess()

    return fake_run


class _FakePopen:
    def __init__(self, fake_run, command, **kwargs):
        result = fake_run(command)
        self.args = command
        self.stdout = io.BytesIO((result.stdout + result.stderr).encode())
        self._returncode = result.returncode

    def wait(self) -> int:
        return self._returncode


def _patch_bench_docker(monkeypatch, fake_run) -> None:
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        cli.subprocess, "Popen", lambda command, **kwargs: _FakePopen(fake_run, command)
    )


def _bench_commands(commands: list) -> list[list[str]]:
    return [c for c in commands if "bench" in c and "serve" in c]


def _setup_bench(tmp_path: Path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)


def test_bench_requires_compose_built(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    result = runner.invoke(app, ["bench", "demo"])

    assert result.exit_code == 1
    assert "fllame recipe build demo" in result.output


def test_bench_requires_running_container(tmp_path: Path, monkeypatch):
    _setup_bench(tmp_path, monkeypatch)
    commands = []
    _patch_bench_docker(monkeypatch, _fake_bench_docker(commands, containers=[]))

    result = runner.invoke(app, ["bench", "demo"])

    assert result.exit_code == 1
    assert "fllame serve demo" in result.output
    assert not _bench_commands(commands)


def test_bench_reports_server_still_loading(tmp_path: Path, monkeypatch):
    _setup_bench(tmp_path, monkeypatch)
    commands = []
    _patch_bench_docker(monkeypatch, _fake_bench_docker(commands, probe_ok=False))

    result = runner.invoke(app, ["bench", "demo"])

    assert result.exit_code == 1
    assert "still loading" in result.output
    assert "docker logs -f fllame-demo" in result.output


def test_bench_rejects_lengths_over_max_model_len(tmp_path: Path, monkeypatch):
    _setup_bench(tmp_path, monkeypatch)
    commands = []
    served = {"id": "org/demo", "max_model_len": 8192}
    _patch_bench_docker(monkeypatch, _fake_bench_docker(commands, served=served))

    result = runner.invoke(app, ["bench", "demo"])

    assert result.exit_code == 1
    assert "8192" in result.output
    assert not _bench_commands(commands)
    assert not (tmp_path / "demo" / "bench").exists()


def test_bench_rejects_bad_levels_before_touching_docker(tmp_path: Path, monkeypatch):
    _setup_bench(tmp_path, monkeypatch)
    commands = []
    _patch_bench_docker(monkeypatch, _fake_bench_docker(commands))

    result = runner.invoke(app, ["bench", "demo", "--concurrency", "4", "--num-prompts", "6"])

    assert result.exit_code == 1
    assert "not a multiple" in result.output
    assert commands == []


def test_bench_runs_each_level_in_container_and_prints_table(tmp_path: Path, monkeypatch):
    _setup_bench(tmp_path, monkeypatch)
    commands = []
    _patch_bench_docker(monkeypatch, _fake_bench_docker(commands))

    result = runner.invoke(app, ["bench", "demo", "--concurrency", "1,4"])

    assert result.exit_code == 0, result.output
    bench_commands = _bench_commands(commands)
    assert [c[c.index("--max-concurrency") + 1] for c in bench_commands] == ["1", "4"]
    for command in bench_commands:
        assert command[command.index("-p") + 1] == "fllame-demo"
        assert command[command.index("exec") + 1] == "-T"
        assert "demo" in command[command.index("exec") :]
        assert command[command.index("--model") + 1] == "org/demo"
        assert command[command.index("--base-url") + 1] == "http://localhost:8000"
    lines = result.stdout.splitlines()
    assert lines[0].split()[:3] == ["CONC", "PROMPTS", "FAILED"]
    assert lines[1].split()[:4] == ["1", "10", "8", "0.67"]
    assert lines[2].split()[:4] == ["4", "12", "4", "0.67"]


def test_bench_saves_reproducible_run_folder(tmp_path: Path, monkeypatch):
    _setup_bench(tmp_path, monkeypatch)
    commands = []
    _patch_bench_docker(monkeypatch, _fake_bench_docker(commands))

    result = runner.invoke(app, ["bench", "demo", "--concurrency", "2", "--input-len", "100"])

    assert result.exit_code == 0, result.output
    (run_dir,) = (tmp_path / "demo" / "bench").iterdir()
    assert {p.name for p in run_dir.iterdir()} == {
        "recipe.yaml",
        "compose.yaml",
        "config.yaml",
        "params.yaml",
        "c2.json",
        "results.txt",
    }
    compose = (tmp_path / "demo" / "compose.yaml").read_text()
    assert (run_dir / "compose.yaml").read_text() == compose
    params = yaml.safe_load((run_dir / "params.yaml").read_text())
    assert params["input_len"] == 100
    assert params["levels"] == [{"concurrency": 2, "num_prompts": 10}]
    assert params["vllm_version"] == "0.27.1"
    assert params["commands"][0].startswith("vllm bench serve")
    settings = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert set(settings) == {"default_image", "default_gpu_memory_utilization"}
    assert json.loads((run_dir / "c2.json").read_text())["completed"] == 4
    assert str(run_dir) in result.stdout


def test_bench_stops_on_failed_level_and_cleans_up(tmp_path: Path, monkeypatch):
    _setup_bench(tmp_path, monkeypatch)
    commands = []
    _patch_bench_docker(monkeypatch, _fake_bench_docker(commands, bench_returncode=1))

    result = runner.invoke(app, ["bench", "demo", "--concurrency", "1,4"])

    assert result.exit_code == 1
    assert "bench output" in result.output
    assert "failed at concurrency 1" in result.output
    assert len(_bench_commands(commands)) == 1
    assert any("rm" in c for c in commands)


def test_bench_help_explains_every_column():
    result = runner.invoke(app, ["bench", "-h"])

    assert result.exit_code == 0
    for header in ("CONC", "FAILED", "S/REQ", "OUT TOK/S", "TPOT MS"):
        assert header in result.stdout


def test_bench_ctrl_c_stops_bench_inside_container(tmp_path: Path, monkeypatch):
    _setup_bench(tmp_path, monkeypatch)
    commands = []
    fake_docker = _fake_bench_docker(commands)

    def fake_run(command, **kwargs):
        if "bench" in command and "serve" in command:
            commands.append(command)
            raise KeyboardInterrupt
        return fake_docker(command, **kwargs)

    _patch_bench_docker(monkeypatch, fake_run)

    result = runner.invoke(app, ["bench", "demo", "--concurrency", "1,4"])

    assert result.exit_code == 130
    assert "CONC" not in result.stdout
    (bench_command,) = _bench_commands(commands)
    result_dir = bench_command[bench_command.index("--result-dir") + 1]
    stop = next(c for c in commands if cli._STOP_BENCH_SCRIPT in c)
    assert stop[-1] == result_dir
    assert "stopped `vllm bench serve`" in result.output
    assert any(c[-3:] == ["rm", "-rf", result_dir] for c in commands)


class _SlowStream:
    def __init__(self, chunks: list[str]):
        self._chunks = [chunk.encode() for chunk in chunks]

    def read1(self, size: int) -> bytes:
        time.sleep(0.25)
        return self._chunks.pop(0) if self._chunks else b""


def test_bench_progress_mirrors_latest_output_line_behind_prefix(monkeypatch):
    chunks = ["Starting run\n  0%|    | 0/10\r", " 20%|#   | 2/10 [02:36<10:25]\r"]
    process = _FakePopen(lambda command: _FakeCompletedProcess(), ["docker"])
    process.stdout = _SlowStream(chunks)
    commands = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        return process

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    shown = []
    monkeypatch.setattr(
        cli.sys.stdout, "write", lambda text: shown.append(text) or len(text), raising=False
    )

    result = cli._run_showing_progress("demo", "Concurrency: 1 | ", ["vllm", "bench", "serve"])

    assert result.stdout == "".join(chunks)
    progress = [text for text in shown if "Concurrency" in text]
    assert progress[0].endswith("Concurrency: 1 | ")
    assert progress[1].endswith("Concurrency: 1 |   0%|    | 0/10")
    assert progress[-1].endswith("Concurrency: 1 |  20%|#   | 2/10 [02:36<10:25]")
    assert any(arg.startswith("COLUMNS=") for arg in commands[0])


def test_bench_rerun_within_same_minute_replaces_run_folder(tmp_path: Path, monkeypatch):
    _setup_bench(tmp_path, monkeypatch)
    _patch_bench_docker(monkeypatch, _fake_bench_docker([]))
    fixed = cli.datetime(2026, 9, 24, 12, 0, 5).astimezone()

    class _FixedDatetime(cli.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(cli, "datetime", _FixedDatetime)

    assert runner.invoke(app, ["bench", "demo", "--concurrency", "1,4"]).exit_code == 0
    assert runner.invoke(app, ["bench", "demo", "--concurrency", "2"]).exit_code == 0

    (run_dir,) = (tmp_path / "demo" / "bench").iterdir()
    assert run_dir.name == "20260924-1200"
    assert sorted(p.name for p in run_dir.glob("c*.json")) == ["c2.json"]


def test_bench_ctrl_c_keeps_finished_levels(tmp_path: Path, monkeypatch):
    _setup_bench(tmp_path, monkeypatch)
    commands = []
    fake_docker = _fake_bench_docker(commands)

    def fake_run(command, **kwargs):
        if (
            "--max-concurrency" in command
            and command[command.index("--max-concurrency") + 1] == "8"
        ):
            raise KeyboardInterrupt
        return fake_docker(command, **kwargs)

    _patch_bench_docker(monkeypatch, fake_run)

    result = runner.invoke(app, ["bench", "demo", "--concurrency", "1,4,8"])

    assert result.exit_code == 130
    (run_dir,) = (tmp_path / "demo" / "bench").iterdir()
    assert sorted(p.name for p in run_dir.glob("c*.json")) == ["c1.json", "c4.json"]
    results = (run_dir / "results.txt").read_text().splitlines()
    assert [line.split()[0] for line in results] == ["CONC", "1", "4"]


def _fake_status_docker(probe_stdout: str, captured: list):
    def fake_run(command, **kwargs):
        if command[-2:] == ["--format", "json"]:
            container = {
                "Name": "fllame-demo",
                "Image": "vllm/vllm-openai:v0.27.1",
                "State": "running",
                "Status": "Up 5 minutes",
                "Publishers": [],
            }
            return _FakeSubprocessResult(stdout=json.dumps([container]))
        captured.append(command)
        return _FakeSubprocessResult(stdout=probe_stdout)

    return fake_run


@pytest.mark.parametrize(
    "probe_stdout", ["loading model\n", "error\n", "not responding\n", "ready\n"]
)
def test_status_appends_readiness_to_a_running_container(tmp_path: Path, monkeypatch, probe_stdout):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    probes = []
    monkeypatch.setattr(cli.subprocess, "run", _fake_status_docker(probe_stdout, probes))

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert f"{probe_stdout.strip().capitalize()} (5 minutes)" in result.output
    assert probes[0][-2:] == [cli._READINESS_SCRIPT, "http://localhost:8000"]


def test_status_reports_unknown_when_the_readiness_probe_cannot_run(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", _fake_status_docker("", []))

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Unknown (5 minutes)" in result.output


def test_status_does_not_probe_a_stopped_container(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    container = {"Name": "fllame-demo", "State": "exited", "Status": "Exited (0) 1 hour ago"}
    monkeypatch.setattr(cli.subprocess, "run", _fake_compose_ps_json({"demo": [container]}))
    probed = []
    monkeypatch.setattr(cli, "_probe_readiness", probed.append)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert probed == []


def _watch_status(tmp_path: Path, monkeypatch, states: list[str], args: list[str]):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", _fake_status_docker("", []))
    remaining = list(states)
    monkeypatch.setattr(cli, "_probe_readiness", lambda handle: remaining.pop(0))
    sleeps = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    return runner.invoke(app, ["status", *args]), remaining, sleeps


def test_status_watch_rechecks_until_ready(tmp_path: Path, monkeypatch):
    result, remaining, sleeps = _watch_status(
        tmp_path,
        monkeypatch,
        ["loading model", "loading model", "not responding", "ready"],
        ["--watch"],
    )

    assert result.exit_code == 0
    assert remaining == []
    assert sleeps == [cli._WATCH_INTERVAL_SECONDS] * 3
    # Over a pipe, a table is printed only when a state changes.
    assert result.output.count("RECIPE_ID") == 3
    assert result.output.count("Loading model") == 1
    assert result.output.rstrip().endswith("8000:8000")
    assert "Ready (5 minutes)" in result.output.splitlines()[-1]


def test_status_watch_stops_on_error_with_exit_code_1(tmp_path: Path, monkeypatch):
    result, remaining, _ = _watch_status(
        tmp_path, monkeypatch, ["loading model", "error", "ready"], ["-w"]
    )

    assert result.exit_code == 1
    assert remaining == ["ready"]
    assert "Error (5 minutes)" in result.output


def test_status_without_watch_checks_once(tmp_path: Path, monkeypatch):
    result, remaining, sleeps = _watch_status(tmp_path, monkeypatch, ["loading model", "ready"], [])

    assert result.exit_code == 0
    assert remaining == ["ready"]
    assert sleeps == []


def test_status_watch_stops_on_ctrl_c(tmp_path: Path, monkeypatch):
    def interrupt(seconds):
        raise KeyboardInterrupt

    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", _fake_status_docker("loading model\n", []))
    monkeypatch.setattr(cli.time, "sleep", interrupt)

    result = runner.invoke(app, ["status", "--watch"])

    assert result.exit_code == 130


def test_status_recipe_id_shows_only_that_recipe(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path, handle="demo-a")
    _write_recipe(tmp_path, handle="demo-b")

    result = runner.invoke(app, ["status", "demo-b"])

    assert result.exit_code == 0
    assert "demo-b" in result.output
    assert "demo-a" not in result.output


def test_status_unknown_recipe_id(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["status", "nope"])

    assert result.exit_code == 1


def test_status_watch_exits_1_when_the_container_stops(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "is_model_cached", lambda repo_id: True)
    _write_compose(tmp_path)
    passes = [
        {"Name": "fllame-demo", "State": "running", "Status": "Up 1 minute"},
        {"Name": "fllame-demo", "State": "exited", "Status": "Exited (1) 1 second ago"},
    ]

    def fake_run(command, **kwargs):
        return _FakeSubprocessResult(stdout=json.dumps([passes.pop(0)]))

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "_probe_readiness", lambda handle: "loading model")
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)

    result = runner.invoke(app, ["status", "--watch"])

    assert result.exit_code == 1
    assert "Exited (1)" in result.output


def test_readiness_status_lowercases_docker_uptime():
    assert cli._readiness_status("loading model", "Up About a minute") == (
        "Loading model (about a minute)"
    )
