from pathlib import Path

from typer.testing import CliRunner

import fllame.cli as cli
from fllame.cli import app
from fllame.domain.hardware import HardwareProfile

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
