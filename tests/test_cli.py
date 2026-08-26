from pathlib import Path

from typer.testing import CliRunner

import fllame.cli as cli
from fllame.cli import app
from fllame.domain.hardware import HardwareProfile

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


def test_pull_downloads_recipes_model(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "pull_model", lambda repo_id: f"/cache/{repo_id}")

    result = runner.invoke(app, ["pull", "demo"])

    assert result.exit_code == 0
    assert "org/demo" in result.stdout


def test_serve_pulls_then_invokes_docker_compose_up(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    pulled = []
    monkeypatch.setattr(cli, "pull_model", lambda repo_id: pulled.append(repo_id))
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["serve", "demo", "--detach"])

    assert result.exit_code == 0
    assert pulled == ["org/demo"]
    assert captured["command"][:3] == ["docker", "compose", "-f"]
    assert captured["command"][-3:] == ["up", "-d", "demo"]


def test_serve_foreground_omits_detach_flag(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    _write_recipe(tmp_path)
    monkeypatch.setattr(cli, "pull_model", lambda repo_id: None)
    captured = {}
    monkeypatch.setattr(cli.subprocess, "run", _capturing_run(captured))

    result = runner.invoke(app, ["serve", "demo"])

    assert result.exit_code == 0
    assert captured["command"][-2:] == ["up", "demo"]


def test_serve_unknown_handle_never_pulls_or_calls_docker(tmp_path: Path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(cli, "pull_model", lambda repo_id: called.append("pull"))
    monkeypatch.setattr(cli.subprocess, "run", lambda command: called.append("docker"))

    result = runner.invoke(app, ["serve", "nope"])

    assert result.exit_code == 1
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
