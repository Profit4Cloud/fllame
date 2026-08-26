from dataclasses import dataclass, field
from datetime import UTC, datetime

from fllame.models import discovery
from fllame.models.discovery import search_models


@dataclass
class _FakeModelInfo:
    id: str
    tags: list[str] = field(default_factory=list)
    downloads: int | None = 0
    downloads_all_time: int | None = 0
    last_modified: datetime | None = None


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
