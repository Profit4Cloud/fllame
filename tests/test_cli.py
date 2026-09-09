from pathlib import Path

from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from typer.testing import CliRunner

import fllame.cli as cli
from fllame import config
from fllame.cli import app
from fllame.domain.hardware import HardwareProfile
from fllame.models.discovery import ModelCandidate

runner = CliRunner()


def _write_recipe(tmp_path: Path, handle: str = "demo") -> None:
    directory = tmp_path / handle
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "recipe.yaml").write_text(
        "image: vllm/vllm-openai:v0.27.1\ncommand: vllm serve org/demo\n"
    )


def _isolate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FLLAME_RECIPES_DIR", str(tmp_path))
    monkeypatch.setenv("FLLAME_CONFIG_FILE", str(tmp_path / "config.yaml"))


def _dialogue_input(*parts: str) -> str:
    """Builds stdin input for `recipe add`'s guided dialogue: each part
    is one line. A blank ("") part accepts the image prompt's prefilled
    default, or ends whichever of the preinstall/env/command blocks is
    currently being read.
    """
    return "\n".join(parts) + "\n"


class _FakeCompletedProcess:
    returncode = 0


def _capturing_run(captured: dict):
    def fake_run(command):
        captured["command"] = command
        return _FakeCompletedProcess()

    return fake_run


def _capturing_run_all(commands: list):
    def fake_run(command):
        commands.append(command)
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
    text = (tmp_path / "demo" / "recipe.yaml").read_text()
    assert "preinstall:" in text
    assert "transformers>=5.8.0" in text


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


def test_config_set_default_image_warns_unpinned(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["config", "set-default-image", "vllm/vllm-openai:latest"])

    assert "unpinned" in result.output


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


def test_model_pull_downloads_recipes_model(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "pull_model", lambda repo_id: f"/cache/{repo_id}")

    result = runner.invoke(app, ["model", "pull", "demo"])

    assert result.exit_code == 0
    assert "org/demo" in result.stdout


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
    assert "100" in result.stdout  # DOWNLOADS
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


def test_serve_pulls_then_invokes_docker_compose_up(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    pulled = []
    monkeypatch.setattr(
        cli, "pull_model", lambda repo_id, offline=False: pulled.append((repo_id, offline))
    )
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["serve", "demo", "--detach"])

    assert result.exit_code == 0
    assert pulled == [("org/demo", False)]
    assert captured["command"][:3] == ["docker", "compose", "-f"]
    assert captured["command"][-3:] == ["up", "-d", "demo"]


def test_serve_uses_recipes_own_compose_folder_and_project(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "pull_model", lambda repo_id, offline=False: None)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    command = captured["command"]
    assert command[command.index("-f") + 1] == str(tmp_path / "demo" / "compose.yaml")
    assert command[command.index("-p") + 1] == "fllame-demo"


def test_serve_never_uses_build_flag(tmp_path: Path, monkeypatch):
    """No separate build step exists at all, with or without a
    preinstall step - preinstall runs as part of the container's own
    startup command instead (see backends/vllm.py)."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "pull_model", lambda repo_id, offline=False: None)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    assert "--build" not in captured["command"]


def test_serve_with_preinstall_writes_no_dockerfile(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    (tmp_path / "demo").mkdir(parents=True)
    (tmp_path / "demo" / "recipe.yaml").write_text(
        "image: vllm/vllm-openai:v0.27.1\n"
        "preinstall:\n"
        "- pip install -U transformers\n"
        "command: vllm serve org/demo\n"
    )
    monkeypatch.setattr(cli, "pull_model", lambda repo_id, offline=False: None)
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run({}))

    result = runner.invoke(app, ["serve", "demo", "--detach"])

    assert result.exit_code == 0
    compose_text = (tmp_path / "demo" / "compose.yaml").read_text()
    assert "pip install -U transformers" in compose_text
    assert not (tmp_path / "demo" / "Dockerfile").exists()


def test_serve_removes_stale_dockerfile_from_before(tmp_path: Path, monkeypatch):
    """A Dockerfile left over from an older fllame version's
    build-a-custom-image approach is cleaned up on the next serve."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    handle_folder = tmp_path / "demo"
    (handle_folder / "Dockerfile").write_text("FROM img\n")
    monkeypatch.setattr(cli, "pull_model", lambda repo_id, offline=False: None)
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run({}))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    assert not (handle_folder / "Dockerfile").exists()


def test_serve_foreground_omits_detach_flag(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "pull_model", lambda repo_id, offline=False: None)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    assert captured["command"][-2:] == ["up", "demo"]


def test_serve_unknown_handle_never_pulls_or_calls_docker(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(cli, "pull_model", lambda repo_id, offline=False: called.append("pull"))
    monkeypatch.setattr(cli.subprocess, "run", lambda command: called.append("docker"))

    result = runner.invoke(app, ["serve", "nope"])

    assert result.exit_code == 1
    assert called == []


def test_serve_offline_passes_offline_to_pull_step(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    pulled = []
    monkeypatch.setattr(
        cli, "pull_model", lambda repo_id, offline=False: pulled.append((repo_id, offline))
    )
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run({}))

    result = runner.invoke(app, ["serve", "demo", "--offline"])

    assert result.exit_code == 0
    assert pulled == [("org/demo", True)]


def test_serve_container_is_always_offline_regardless_of_flag(tmp_path: Path, monkeypatch):
    """HF_HUB_OFFLINE=1 is unconditional (VllmServingBackend bakes it
    into every generated service) - the model is always already fully
    downloaded by the time the container runs, so vLLM has no
    legitimate need to reach the Hub itself. --offline only controls
    whether the pull step itself is allowed to touch the network."""
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "pull_model", lambda repo_id, offline=False: None)
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run({}))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    compose_text = (config.recipe_dir("demo") / "compose.yaml").read_text()
    assert "HF_HUB_OFFLINE" in compose_text


def test_serve_offline_cache_miss_gives_friendly_error(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    def fake_pull(repo_id, offline=False):
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(cli, "pull_model", fake_pull)
    called = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command: called.append("docker"))

    result = runner.invoke(app, ["serve", "demo", "--offline"])

    assert result.exit_code == 1
    assert "fllame model pull" in result.output
    assert called == []


def test_recipe_build_fails_when_model_not_cached(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(
        cli,
        "pull_model",
        lambda repo_id, offline=False: (_ for _ in ()).throw(LocalEntryNotFoundError("nope")),
    )

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 1
    assert "fllame model pull" in result.output
    assert not (tmp_path / "demo" / "compose.yaml").exists()


def test_recipe_build_writes_compose_when_model_cached(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    pulled = []
    monkeypatch.setattr(
        cli, "pull_model", lambda repo_id, offline=False: pulled.append((repo_id, offline))
    )

    result = runner.invoke(app, ["recipe", "build", "demo"])

    assert result.exit_code == 0
    assert pulled == [("org/demo", True)]
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


def test_recipe_add_build_without_pull_fails_when_not_cached(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli,
        "pull_model",
        lambda repo_id, offline=False: (_ for _ in ()).throw(LocalEntryNotFoundError("nope")),
    )

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

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "img:v1", "--pull", "--build"],
        input=_dialogue_input("", "", "vllm serve org/demo"),
    )

    assert result.exit_code == 0
    # --pull's own download, then --build's offline cache check.
    assert pulled == [("org/demo", False), ("org/demo", True)]
    assert (tmp_path / "demo" / "compose.yaml").is_file()


def test_status_invokes_docker_compose_ps(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert captured["command"][-1] == "ps"


def test_status_shows_a_header_and_ps_per_recipe(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path, handle="demo-a")
    _write_recipe(tmp_path, handle="demo-b")
    commands = []
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run_all(commands))

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "== demo-a ==" in result.output
    assert "== demo-b ==" in result.output
    assert len(commands) == 2
    projects = {command[command.index("-p") + 1] for command in commands}
    assert projects == {"fllame-demo-a", "fllame-demo-b"}


def test_status_no_recipes(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "No recipes found" in result.stdout


def test_status_skips_invalid_recipe_with_warning(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path, handle="good")
    (tmp_path / "bad").mkdir(parents=True)
    (tmp_path / "bad" / "recipe.yaml").write_text("image: img\n")  # missing command
    commands = []
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run_all(commands))

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "warning" in result.output
    assert "== good ==" in result.output
    assert "== bad ==" not in result.output
    assert len(commands) == 1


def test_stop_invokes_docker_compose_stop(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["stop", "demo"])

    assert result.exit_code == 0
    assert captured["command"][-2:] == ["stop", "demo"]


def test_docker_not_found_gives_friendly_error(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    def raise_not_found(command):
        raise FileNotFoundError

    monkeypatch.setattr(cli.subprocess, "run", raise_not_found)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 1
    assert "docker" in result.output.lower()
