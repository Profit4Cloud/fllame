import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fllame.models import discovery
from fllame.models.discovery import search_models


@dataclass
class _FakeSafeTensorsInfo:
    total: int
    parameters: dict[str, int] = field(default_factory=dict)


@dataclass
class _FakeModelInfo:
    id: str
    tags: list[str] = field(default_factory=list)
    downloads: int | None = 0
    downloads_all_time: int | None = 0
    last_modified: datetime | None = None
    safetensors: _FakeSafeTensorsInfo | None = None


def _patch_list_models(monkeypatch, by_search_term: dict[str, list[_FakeModelInfo]]):
    def fake_list_models(*, search, expand, limit):
        return by_search_term.get(search, [])

    monkeypatch.setattr(discovery, "list_models", fake_list_models)


def _model_with_vram(id: str, tags: list[str], vram_gb: float, **kwargs) -> _FakeModelInfo:
    """A fake model whose safetensors metadata is one U8 (1-byte) tensor
    sized to produce exactly `vram_gb` of estimated VRAM."""
    element_count = round(vram_gb * 1024**3)
    return _FakeModelInfo(
        id=id,
        tags=tags,
        safetensors=_FakeSafeTensorsInfo(total=element_count, parameters={"U8": element_count}),
        **kwargs,
    )


def test_filters_by_declared_params_range(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {
            "awq": [
                _FakeModelInfo(id="org/tiny-1B-AWQ", tags=["awq"]),
                _FakeModelInfo(id="org/mid-7B-AWQ", tags=["awq"]),
                _FakeModelInfo(id="org/huge-70B-AWQ", tags=["awq"]),
            ]
        },
    )

    results = search_models(
        quantizations=["awq"],
        max_size_gb=1000.0,
        min_params_billion=5.0,
        max_params_billion=10.0,
    )

    assert [c.repo_id for c in results] == ["org/mid-7B-AWQ"]


def test_unknown_params_passes_with_no_bounds(monkeypatch):
    _patch_list_models(
        monkeypatch, {"awq": [_FakeModelInfo(id="org/no-size-marker-AWQ", tags=["awq"])]}
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert len(results) == 1
    assert results[0].params_billion is None


def test_unknown_params_excluded_once_a_bound_is_given(monkeypatch):
    _patch_list_models(
        monkeypatch, {"awq": [_FakeModelInfo(id="org/no-size-marker-AWQ", tags=["awq"])]}
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0, min_params_billion=1.0)

    assert results == []


def test_safetensors_total_wins_over_misleading_repo_id_number(monkeypatch):
    # Regression: "nvidia/Qwen3.8-2.4T-A95B-NVFP4" was parsed as 95B (its
    # active-parameter count) by the old repo_id-only regex, when the
    # Hub's own safetensors metadata reports the real total.
    _patch_list_models(
        monkeypatch,
        {
            "nvfp4": [
                _FakeModelInfo(
                    id="nvidia/Qwen3.8-2.4T-A95B-NVFP4",
                    tags=["nvfp4"],
                    safetensors=_FakeSafeTensorsInfo(total=1_300_000_000_000),
                )
            ]
        },
    )

    results = search_models(quantizations=["nvfp4"], max_size_gb=2_000_000.0)

    assert results[0].params_billion == 1300.0


def test_falls_back_to_repo_id_when_no_safetensors_metadata(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {"awq": [_FakeModelInfo(id="org/mid-7B-AWQ", tags=["awq"], safetensors=None)]},
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert results[0].params_billion == 7.0


def test_repo_id_fallback_understands_trillion_and_million_units(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {
            "nvfp4": [
                _FakeModelInfo(id="org/huge-2.4T-NVFP4", tags=["nvfp4"], safetensors=None),
                _FakeModelInfo(id="org/tiny-500M-NVFP4", tags=["nvfp4"], safetensors=None),
            ]
        },
    )

    results = search_models(quantizations=["nvfp4"], max_size_gb=100.0)

    by_id = {c.repo_id: c.params_billion for c in results}
    assert by_id["org/huge-2.4T-NVFP4"] == 2400.0
    assert by_id["org/tiny-500M-NVFP4"] == 0.5


def test_estimated_vram_computed_from_dtype_byte_breakdown(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {
            "nvfp4": [
                _FakeModelInfo(
                    id="nvidia/Qwen3.8-2.4T-A95B-NVFP4",
                    tags=["nvfp4"],
                    safetensors=_FakeSafeTensorsInfo(
                        total=1_300_000_000_000,
                        # 1 GiB of U8 (1 byte each) + 1 GiB of F16 (2 bytes each).
                        parameters={"U8": 1024**3, "F16": (1024**3) // 2},
                    ),
                )
            ]
        },
    )

    results = search_models(quantizations=["nvfp4"], max_size_gb=1_000_000.0)

    assert results[0].estimated_vram_gb == 2.0


def test_estimated_vram_unknown_without_safetensors_metadata(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {"awq": [_FakeModelInfo(id="org/mid-7B-AWQ", tags=["awq"], safetensors=None)]},
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert results[0].estimated_vram_gb is None


def test_estimated_vram_unknown_for_unrecognized_dtype(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {
            "awq": [
                _FakeModelInfo(
                    id="org/mid-7B-AWQ",
                    tags=["awq"],
                    safetensors=_FakeSafeTensorsInfo(
                        total=7_000_000_000, parameters={"SOME_NEW_DTYPE": 7_000_000_000}
                    ),
                )
            ]
        },
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert results[0].estimated_vram_gb is None


def test_known_vram_over_max_size_excluded(monkeypatch):
    _patch_list_models(
        monkeypatch, {"awq": [_model_with_vram("org/too-big-AWQ", ["awq"], vram_gb=20.0)]}
    )

    results = search_models(quantizations=["awq"], max_size_gb=10.0)

    assert results == []


def test_unknown_vram_passes_when_max_size_is_a_default(monkeypatch):
    _patch_list_models(
        monkeypatch, {"awq": [_FakeModelInfo(id="org/no-safetensors-AWQ", tags=["awq"])]}
    )

    results = search_models(quantizations=["awq"], max_size_gb=10.0)

    assert len(results) == 1


def test_unknown_vram_excluded_when_max_size_is_explicit(monkeypatch):
    _patch_list_models(
        monkeypatch, {"awq": [_FakeModelInfo(id="org/no-safetensors-AWQ", tags=["awq"])]}
    )

    results = search_models(quantizations=["awq"], max_size_gb=10.0, exclude_unknown_size=True)

    assert results == []


def test_quantization_match_via_tag_or_repo_id_suffix(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {
            "gptq": [
                _FakeModelInfo(id="org/tagged-7B", tags=["gptq"]),
                _FakeModelInfo(id="org/suffix-7B-GPTQ", tags=[]),
                _FakeModelInfo(id="org/not-quantized-7B", tags=["fp16"]),
            ]
        },
    )

    results = search_models(quantizations=["gptq"], max_size_gb=100.0)

    assert {c.repo_id for c in results} == {"org/tagged-7B", "org/suffix-7B-GPTQ"}


def test_gguf_excluded_via_tag(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {
            "nvfp4": [
                _FakeModelInfo(
                    id="cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF",
                    tags=["nvfp4", "gguf"],
                ),
                _FakeModelInfo(id="org/real-27B-NVFP4", tags=["nvfp4"]),
            ]
        },
    )

    results = search_models(quantizations=["nvfp4"], max_size_gb=100.0)

    assert {c.repo_id for c in results} == {"org/real-27B-NVFP4"}


def test_gguf_excluded_via_repo_id_suffix_without_tag(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {"nvfp4": [_FakeModelInfo(id="org/model-27B-NVFP4-GGUF", tags=["nvfp4"])]},
    )

    results = search_models(quantizations=["nvfp4"], max_size_gb=100.0)

    assert results == []


def test_deduplicates_across_quantization_searches(monkeypatch):
    same = _FakeModelInfo(id="org/dual-tagged-7B", tags=["awq", "gptq"])
    _patch_list_models(monkeypatch, {"awq": [same], "gptq": [same]})

    results = search_models(quantizations=["awq", "gptq"], max_size_gb=100.0)

    assert len(results) == 1


def test_max_results_caps_output(monkeypatch):
    candidates = [
        _FakeModelInfo(id=f"org/model-{i}-7B-AWQ", tags=["awq"], downloads=i) for i in range(5)
    ]
    _patch_list_models(monkeypatch, {"awq": candidates})

    results = search_models(quantizations=["awq"], max_size_gb=100.0, max_results=2)

    assert len(results) == 2


def test_ranking_prefers_more_downloads_and_closer_to_size_ceiling(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {
            "awq": [
                _model_with_vram(
                    "org/small-vram-AWQ", ["awq"], 1.0, downloads=10, downloads_all_time=10
                ),
                _model_with_vram(
                    "org/close-fit-vram-AWQ",
                    ["awq"],
                    9.0,
                    downloads=10000,
                    downloads_all_time=100000,
                    last_modified=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            ]
        },
    )

    results = search_models(quantizations=["awq"], max_size_gb=10.0)

    assert results[0].repo_id == "org/close-fit-vram-AWQ"


def test_ranking_weighs_params_closeness_only_when_max_params_given(monkeypatch):
    # Both candidates report identical estimated VRAM (same byte total,
    # split across a 2-byte and a 1-byte dtype respectively) but
    # different param counts, so this isolates the params-closeness half
    # of the fit score from the VRAM-closeness half.
    same_vram_half_params = _FakeModelInfo(
        id="org/half-of-params-ceiling-AWQ",
        tags=["awq"],
        safetensors=_FakeSafeTensorsInfo(total=1_000_000_000, parameters={"F16": 1_000_000_000}),
    )
    same_vram_at_params_ceiling = _FakeModelInfo(
        id="org/at-params-ceiling-AWQ",
        tags=["awq"],
        safetensors=_FakeSafeTensorsInfo(total=2_000_000_000, parameters={"U8": 2_000_000_000}),
    )
    _patch_list_models(
        monkeypatch, {"awq": [same_vram_half_params, same_vram_at_params_ceiling]}
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0, max_params_billion=2.0)

    assert results[0].repo_id == "org/at-params-ceiling-AWQ"


def test_quantization_searches_run_concurrently(monkeypatch):
    # Each quantization's Hub search is an independent network call;
    # simulating per-call latency and asserting on the total elapsed
    # time is a regression test that they run concurrently, not one
    # after another - sequential would take roughly 5x as long as any
    # one of them.
    per_call_delay = 0.2

    def fake_list_models(*, search, expand, limit):
        time.sleep(per_call_delay)
        return []

    monkeypatch.setattr(discovery, "list_models", fake_list_models)

    start = time.monotonic()
    search_models(quantizations=["awq", "gptq", "fp8", "fp4", "nvfp4"], max_size_gb=100.0)
    elapsed = time.monotonic() - start

    assert elapsed < per_call_delay * 3
