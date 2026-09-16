from fllame.domain.hardware import HardwareProfile
from fllame.models.architecture import ModelArchitecture
from fllame.models.sizing import (
    _ACTIVATION_OVERHEAD_GB,
    _MIN_USABLE_MAX_MODEL_LEN,
    DEFAULT_SIZING_CONFIG,
    SizingConfig,
    kv_cache_bytes_per_token,
    max_context_length_for_budget,
    memory_budget_gb,
    usable_memory_gb,
)


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


def test_usable_memory_discrete_no_os_reserve():
    usable = usable_memory_gb(budget_gb=80.0, unified_memory=False)
    assert usable == 80.0 * 0.85


def test_usable_memory_unified_subtracts_os_reserve():
    usable = usable_memory_gb(budget_gb=128.0, unified_memory=True)
    assert usable == (128.0 - 8.0) * 0.85


def test_usable_memory_never_goes_negative_on_tiny_unified_budget():
    usable = usable_memory_gb(budget_gb=2.0, unified_memory=True)
    assert usable == 0.0


_LLAMA_8B_ARCH = ModelArchitecture(
    num_layers=32, num_kv_heads=8, head_dim=128, max_context_length=8192
)


def test_kv_cache_bytes_per_token_hand_computed():
    # 2 (K and V) * 32 layers * 8 kv heads * 128 head_dim * 2 bytes = 131072.
    assert kv_cache_bytes_per_token(_LLAMA_8B_ARCH) == 131072


def test_max_context_length_for_budget_caps_at_architectural_ceiling():
    # An enormous budget would otherwise compute a context length far
    # past what the model was ever trained for.
    context_length = max_context_length_for_budget(
        arch=_LLAMA_8B_ARCH, weights_gb=16.0, total_budget_gb=1000.0
    )
    assert context_length == _LLAMA_8B_ARCH.max_context_length


def test_max_context_length_for_budget_tight_budget_below_ceiling():
    context_length = max_context_length_for_budget(
        arch=_LLAMA_8B_ARCH, weights_gb=16.0, total_budget_gb=20.0
    )
    assert 0 < context_length < _LLAMA_8B_ARCH.max_context_length


def test_max_context_length_for_budget_insufficient_budget_returns_zero():
    # Weights alone already exceed the budget.
    context_length = max_context_length_for_budget(
        arch=_LLAMA_8B_ARCH, weights_gb=16.0, total_budget_gb=10.0
    )
    assert context_length == 0


def test_max_context_length_for_budget_respects_configured_activation_overhead():
    """A larger activation/overhead allowance leaves less room for KV
    cache, so it can only shrink (never grow) the computed context
    length relative to the module default - this is exactly what makes
    `fllame config set-activation-overhead` an effective knob."""
    default_overhead_length = max_context_length_for_budget(
        arch=_LLAMA_8B_ARCH, weights_gb=1.0, total_budget_gb=10.0
    )

    larger_overhead_length = max_context_length_for_budget(
        arch=_LLAMA_8B_ARCH, weights_gb=1.0, total_budget_gb=10.0, activation_overhead_gb=8.0
    )

    assert 0 < larger_overhead_length < default_overhead_length


def test_sizing_config_defaults_match_module_constants():
    config = SizingConfig()

    assert config.min_usable_max_model_len == _MIN_USABLE_MAX_MODEL_LEN
    assert config.activation_overhead_gb == _ACTIVATION_OVERHEAD_GB


def test_default_sizing_config_is_a_sizing_config_with_module_defaults():
    assert DEFAULT_SIZING_CONFIG == SizingConfig()
