from fllame.domain.hardware import HardwareProfile
from fllame.models.sizing import max_params_billion, memory_budget_gb


def _profile(**overrides) -> HardwareProfile:
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


def test_memory_budget_discrete_gpu_uses_vram():
    assert memory_budget_gb(_profile()) == 80.0


def test_memory_budget_unified_memory_uses_ram():
    profile = _profile(
        gpu_name="NVIDIA GB10",
        vram_gb_per_gpu=None,
        ram_gb=128.0,
        chip_family="grace_blackwell",
    )
    assert memory_budget_gb(profile) == 128.0


def test_memory_budget_falls_back_to_ram_when_vram_unknown():
    profile = _profile(vram_gb_per_gpu=None, ram_gb=64.0, chip_family="nvidia")
    assert memory_budget_gb(profile) == 64.0


def test_memory_budget_none_when_nothing_known():
    profile = _profile(
        gpu_name=None, gpu_count=0, vram_gb_per_gpu=None, ram_gb=None, chip_family="none"
    )
    assert memory_budget_gb(profile) is None


def test_max_params_billion_discrete_no_os_reserve():
    ceiling = max_params_billion(budget_gb=80.0, unified_memory=False, quantization="awq")
    assert ceiling == (80.0 * 0.85) / 0.5


def test_max_params_billion_unified_subtracts_os_reserve():
    ceiling = max_params_billion(budget_gb=128.0, unified_memory=True, quantization="awq")
    assert ceiling == ((128.0 - 8.0) * 0.85) / 0.5


def test_max_params_billion_unknown_quantization_uses_default_bytes_per_param():
    ceiling = max_params_billion(budget_gb=80.0, unified_memory=False, quantization="int4")
    assert ceiling == (80.0 * 0.85) / 1.0


def test_max_params_billion_never_goes_negative_on_tiny_unified_budget():
    ceiling = max_params_billion(budget_gb=2.0, unified_memory=True, quantization="awq")
    assert ceiling == 0.0
