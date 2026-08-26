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
    (tmp_path / f"{handle}.yaml").write_text("repo_id: org/demo\nimage: vllm/vllm-openai:v0.27.1\n")


def _isolate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FLLAME_RECIPES_DIR", str(tmp_path))
    monkeypatch.setenv("FLLAME_STATE_DIR", str(tmp_path / "state"))


class _FakeCompletedProcess:
    returncode = 0


def _capturing_run(captured: dict):
    def fake_run(command):
        captured["command"] = command
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


def test_recipe_add_from_pasted_block(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    pasted = "export FOO=bar\nvllm serve meta-llama/Llama-3-8B-Instruct --port 8000\n"

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "vllm/vllm-openai:v0.27.1"],
        input=pasted,
    )

    assert result.exit_code == 0
    saved = tmp_path / "llama-3-8b-instruct.yaml"
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
    saved = tmp_path / "qwen3-8b-fp8.yaml"
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
    pasted = "vllm serve org/demo\n"

    runner.invoke(app, ["recipe", "add", "--image", "img:v1"], input=pasted)
    result = runner.invoke(app, ["recipe", "add", "--image", "img:v1"], input=pasted)

    assert result.exit_code == 0
    assert (tmp_path / "demo.yaml").is_file()
    assert (tmp_path / "demo_2.yaml").is_file()


def test_recipe_add_pinned_image_no_warning(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "vllm/vllm-openai:v0.27.1"],
        input="vllm serve org/demo\n",
    )

    assert "unpinned" not in result.output


def test_recipe_add_unpinned_image_warns(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "vllm/vllm-openai:latest"],
        input="vllm serve org/demo\n",
    )

    assert "unpinned" in result.output


def test_recipe_add_rejects_bad_paste(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["recipe", "add", "--image", "img:v1"],
        input="docker run img:v1\n",
    )

    assert result.exit_code == 1
    assert list(tmp_path.glob("*.yaml")) == []


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


def test_recipe_edit_reports_now_invalid_recipe(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    def fake_edit(filename):
        Path(filename).write_text("repo_id: org/demo\n")  # image now missing
        return None

    monkeypatch.setattr(cli.click, "edit", fake_edit)

    result = runner.invoke(app, ["recipe", "edit", "demo"])

    assert result.exit_code == 1
    assert "no longer a valid recipe" in result.output


def test_recipe_remove_with_yes_flag(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    result = runner.invoke(app, ["recipe", "remove", "demo", "--yes"])

    assert result.exit_code == 0
    assert not (tmp_path / "demo.yaml").exists()


def test_recipe_remove_prompts_and_respects_no(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)

    result = runner.invoke(app, ["recipe", "remove", "demo"], input="n\n")

    assert result.exit_code == 0
    assert (tmp_path / "demo.yaml").exists()


def test_recipe_remove_missing_handle(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    result = runner.invoke(app, ["recipe", "remove", "nope", "--yes"])

    assert result.exit_code == 1


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
        return [ModelCandidate("org/demo-7B-AWQ", "awq", 7.0, 100, 1000, None)]

    monkeypatch.setattr(cli, "search_models", fake_search_models)

    result = runner.invoke(app, ["model", "scan"])

    assert result.exit_code == 0
    assert "org/demo-7B-AWQ" in result.stdout
    assert set(captured["quantizations"]) == {"awq", "gptq", "fp8"}
    assert captured["max_params_billion"] is None


def test_model_scan_explicit_overrides_never_touch_hardware(monkeypatch):
    def fail_if_called():
        raise AssertionError("scan_hardware should not be called when both overrides are given")

    monkeypatch.setattr(cli, "scan_hardware", fail_if_called)
    captured = {}

    def fake_search_models(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "search_models", fake_search_models)

    result = runner.invoke(app, ["model", "scan", "--quant", "gptq", "--max-params", "13"])

    assert result.exit_code == 0
    assert captured["quantizations"] == ["gptq"]
    assert captured["ceiling_billion"] == {"gptq": 13.0}


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


def test_serve_offline_passes_offline_to_pull_and_sets_container_env(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    pulled = []
    monkeypatch.setattr(
        cli, "pull_model", lambda repo_id, offline=False: pulled.append((repo_id, offline))
    )
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["serve", "demo", "--offline"])

    assert result.exit_code == 0
    assert pulled == [("org/demo", True)]
    compose_text = config.compose_file_path().read_text()
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


def test_status_invokes_docker_compose_ps(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert captured["command"][-1] == "ps"


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
