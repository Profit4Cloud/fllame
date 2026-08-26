import subprocess
from pathlib import Path

from fllame.hardware import scanner


def test_no_gpu_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(scanner.shutil, "which", lambda _: None)

    profile = scanner.scan_hardware()

    assert profile.has_gpu is False
    assert profile.chip_family == "none"
    assert profile.supported_quantizations == []


def test_generic_nvidia_gpu(monkeypatch):
    monkeypatch.setattr(scanner.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="NVIDIA A100 80GB PCIe, 81920\n"
        )

    monkeypatch.setattr(scanner.subprocess, "run", fake_run)

    profile = scanner.scan_hardware()

    assert profile.gpu_name == "NVIDIA A100 80GB PCIe"
    assert profile.gpu_count == 1
    assert profile.vram_gb_per_gpu == 80.0
    assert profile.chip_family == "nvidia"
    assert profile.supported_quantizations == ["awq", "gptq", "fp8"]


def test_grace_blackwell_gets_extra_quantizations(monkeypatch):
    monkeypatch.setattr(scanner.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="NVIDIA GB200, 196608\n")

    monkeypatch.setattr(scanner.subprocess, "run", fake_run)

    profile = scanner.scan_hardware()

    assert profile.chip_family == "grace_blackwell"
    assert "fp4" in profile.supported_quantizations
    assert "nvfp4" in profile.supported_quantizations


def test_unified_memory_gpu_reports_unknown_vram_instead_of_crashing(monkeypatch):
    # The exact `nvidia-smi` output on a DGX Spark's GB10 - unified
    # memory, no discrete VRAM pool to report as memory.total.
    monkeypatch.setattr(scanner.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="NVIDIA GB10, [N/A]\n")

    monkeypatch.setattr(scanner.subprocess, "run", fake_run)

    profile = scanner.scan_hardware()

    assert profile.has_gpu is True
    assert profile.gpu_name == "NVIDIA GB10"
    assert profile.vram_gb_per_gpu is None
    assert profile.chip_family == "grace_blackwell"


def test_multiple_gpus_counted(monkeypatch):
    monkeypatch.setattr(scanner.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="NVIDIA H100, 81920\nNVIDIA H100, 81920\n",
        )

    monkeypatch.setattr(scanner.subprocess, "run", fake_run)

    profile = scanner.scan_hardware()

    assert profile.gpu_count == 2
    assert profile.vram_gb_per_gpu == 80.0


def test_nvidia_smi_failure_reports_no_gpu(monkeypatch):
    monkeypatch.setattr(scanner.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(scanner.subprocess, "run", fake_run)

    profile = scanner.scan_hardware()

    assert profile.has_gpu is False


def test_ram_gb_parses_meminfo(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       65850000 kB\nMemFree:        1000 kB\n")

    ram_gb = scanner._ram_gb(meminfo)

    assert round(ram_gb, 1) == round(65850000 / (1024 * 1024), 1)


def test_ram_gb_missing_file_returns_none(tmp_path: Path):
    assert scanner._ram_gb(tmp_path / "does-not-exist") is None
