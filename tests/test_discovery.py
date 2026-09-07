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


def test_filters_by_declared_size_range(monkeypatch):
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
        ceiling_billion={"awq": 100.0},
        min_params_billion=5.0,
        max_params_billion=10.0,
    )

    assert [c.repo_id for c in results] == ["org/mid-7B-AWQ"]


def test_unknown_size_passes_with_no_bounds(monkeypatch):
    _patch_list_models(
        monkeypatch, {"awq": [_FakeModelInfo(id="org/no-size-marker-AWQ", tags=["awq"])]}
    )

    results = search_models(quantizations=["awq"], ceiling_billion={"awq": 100.0})

    assert len(results) == 1
    assert results[0].params_billion is None


def test_unknown_size_excluded_once_a_bound_is_given(monkeypatch):
    _patch_list_models(
        monkeypatch, {"awq": [_FakeModelInfo(id="org/no-size-marker-AWQ", tags=["awq"])]}
    )

    results = search_models(
        quantizations=["awq"], ceiling_billion={"awq": 100.0}, min_params_billion=1.0
    )

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

    results = search_models(quantizations=["nvfp4"], ceiling_billion={"nvfp4": 2000.0})

    assert results[0].params_billion == 1300.0


def test_falls_back_to_repo_id_when_no_safetensors_metadata(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {"awq": [_FakeModelInfo(id="org/mid-7B-AWQ", tags=["awq"], safetensors=None)]},
    )

    results = search_models(quantizations=["awq"], ceiling_billion={"awq": 100.0})

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

    results = search_models(quantizations=["nvfp4"], ceiling_billion={"nvfp4": 3000.0})

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

    results = search_models(quantizations=["nvfp4"], ceiling_billion={"nvfp4": 2000.0})

    assert results[0].estimated_vram_gb == 2.0


def test_estimated_vram_unknown_without_safetensors_metadata(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {"awq": [_FakeModelInfo(id="org/mid-7B-AWQ", tags=["awq"], safetensors=None)]},
    )

    results = search_models(quantizations=["awq"], ceiling_billion={"awq": 100.0})

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

    results = search_models(quantizations=["awq"], ceiling_billion={"awq": 100.0})

    assert results[0].estimated_vram_gb is None


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

    results = search_models(quantizations=["gptq"], ceiling_billion={"gptq": 100.0})

    assert {c.repo_id for c in results} == {"org/tagged-7B", "org/suffix-7B-GPTQ"}


def test_deduplicates_across_quantization_searches(monkeypatch):
    same = _FakeModelInfo(id="org/dual-tagged-7B", tags=["awq", "gptq"])
    _patch_list_models(monkeypatch, {"awq": [same], "gptq": [same]})

    results = search_models(
        quantizations=["awq", "gptq"], ceiling_billion={"awq": 100.0, "gptq": 100.0}
    )

    assert len(results) == 1


def test_max_results_caps_output(monkeypatch):
    candidates = [
        _FakeModelInfo(id=f"org/model-{i}-7B-AWQ", tags=["awq"], downloads=i) for i in range(5)
    ]
    _patch_list_models(monkeypatch, {"awq": candidates})

    results = search_models(quantizations=["awq"], ceiling_billion={"awq": 100.0}, max_results=2)

    assert len(results) == 2


def test_ranking_prefers_more_downloads_and_closer_to_ceiling(monkeypatch):
    _patch_list_models(
        monkeypatch,
        {
            "awq": [
                _FakeModelInfo(
                    id="org/small-1B-AWQ", tags=["awq"], downloads=10, downloads_all_time=10
                ),
                _FakeModelInfo(
                    id="org/close-fit-9B-AWQ",
                    tags=["awq"],
                    downloads=10000,
                    downloads_all_time=100000,
                    last_modified=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            ]
        },
    )

    results = search_models(quantizations=["awq"], ceiling_billion={"awq": 10.0})

    assert results[0].repo_id == "org/close-fit-9B-AWQ"
