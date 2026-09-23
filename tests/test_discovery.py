import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from huggingface_hub.errors import HfHubHTTPError

from fllame.models import discovery
from fllame.models.discovery import search_models
from fllame.models.vram import RUNTIME_OVERHEAD_GB

_GIB = 1024**3

# 8 KiB per token per billion params x 32768 tokens = 0.25 GiB per billion.
_KV_GB_PER_BILLION_PARAMS = 0.25


def _expected_vram_gb(params_billion: float, weight_gb: float) -> float:
    return weight_gb + _KV_GB_PER_BILLION_PARAMS * params_billion + RUNTIME_OVERHEAD_GB


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


def _model(
    id: str,
    tags: list[str],
    *,
    params_billion: float | None = None,
    **kwargs,
) -> _FakeModelInfo:
    """`params_billion` is the Hub's own safetensors total, which only
    counts when `id` carries no size of its own."""
    return _FakeModelInfo(
        id=id,
        tags=tags,
        safetensors=(
            _FakeSafeTensorsInfo(total=round(params_billion * 1_000_000_000))
            if params_billion is not None
            else None
        ),
        **kwargs,
    )


def _patch_hub(
    monkeypatch,
    by_search_term: dict[str, list[_FakeModelInfo]],
    other_repos: list[_FakeModelInfo] = (),
) -> list[tuple[str, dict]]:
    """Returns the log of `model_info` calls, as (repo_id, kwargs)."""
    repos = {info.id: info for infos in by_search_term.values() for info in infos}
    repos.update({info.id: info for info in other_repos})
    calls: list[tuple[str, dict]] = []

    def fake_list_models(*, search, expand, limit):
        return by_search_term.get(search, [])

    def fake_model_info(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        if repo_id not in repos:
            raise HfHubHTTPError("404 Client Error")
        return repos[repo_id]

    monkeypatch.setattr(discovery, "list_models", fake_list_models)
    monkeypatch.setattr(discovery, "model_info", fake_model_info)
    return calls


def test_filters_by_params_range(monkeypatch):
    _patch_hub(
        monkeypatch,
        {
            "awq": [
                _model("org/tiny-1B-AWQ", ["awq"]),
                _model("org/mid-7B-AWQ", ["awq"]),
                _model("org/huge-70B-AWQ", ["awq"]),
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
    _patch_hub(monkeypatch, {"awq": [_model("org/no-size-marker-AWQ", ["awq"])]})

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert len(results) == 1
    assert results[0].params_billion is None


def test_unknown_params_excluded_once_a_bound_is_given(monkeypatch):
    _patch_hub(monkeypatch, {"awq": [_model("org/no-size-marker-AWQ", ["awq"])]})

    results = search_models(quantizations=["awq"], max_size_gb=100.0, min_params_billion=1.0)

    assert results == []


def test_params_from_the_repo_name_win_over_the_hub_total(monkeypatch):
    # Real case: a 27B AWQ build whose own Hub total counts packed values.
    _patch_hub(
        monkeypatch,
        {"awq": [_model("barrydeen/Qwen3.8-27B-AWQ-4bit", ["awq"], params_billion=11.08)]},
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert results[0].params_billion == 27.0


def test_params_fall_back_to_the_hub_total_without_a_size_in_the_name(monkeypatch):
    _patch_hub(
        monkeypatch,
        {"nvfp4": [_model("org/Qwen3.8-Flash-Next-NVFP4", ["nvfp4"], params_billion=92.7)]},
    )

    results = search_models(quantizations=["nvfp4"], max_size_gb=1000.0)

    assert results[0].params_billion == 92.7


@pytest.mark.parametrize(
    ("repo_id", "params_billion"),
    [
        ("org/Qwen3-30B-A3B-AWQ", 30.0),
        ("nvidia/Qwen3.8-2.4T-A95B-NVFP4", 2400.0),
        ("org/tiny-500M-AWQ", 0.5),
        ("org/Llama-3.1-8b-Instruct-AWQ", 8.0),
        ("org/model-4bit-AWQ", None),
        ("org/Mixtral-8x7B-AWQ", None),
        ("org7B/model-AWQ", None),
    ],
)
def test_params_billion_from_the_repo_name(repo_id, params_billion):
    assert discovery._params_billion(_FakeModelInfo(id=repo_id)) == params_billion


def test_estimated_vram_from_params_and_quantization(monkeypatch):
    _patch_hub(monkeypatch, {"fp8": [_model("org/model-10B-FP8", ["fp8"])]})

    results = search_models(quantizations=["fp8"], max_size_gb=100.0)

    assert results[0].estimated_vram_gb == pytest.approx(
        _expected_vram_gb(10.0, weight_gb=10e9 * 1.05 / _GIB)
    )


@pytest.mark.parametrize(
    ("repo_id", "tags"),
    [("org/model-10B-GPTQ-Int8", ["gptq"]), ("org/model-10B-GPTQ", ["gptq", "8-bit"])],
)
def test_eight_bit_gptq_detected_from_name_or_tag(monkeypatch, repo_id, tags):
    _patch_hub(monkeypatch, {"gptq": [_model(repo_id, tags)]})

    results = search_models(quantizations=["gptq"], max_size_gb=100.0)

    assert results[0].estimated_vram_gb == pytest.approx(
        _expected_vram_gb(10.0, weight_gb=10e9 * 1.05 / _GIB)
    )


def test_unknown_quantization_has_unknown_size(monkeypatch):
    _patch_hub(monkeypatch, {"weird": [_model("org/model-10B-WEIRD", ["weird"])]})

    results = search_models(quantizations=["weird"], max_size_gb=100.0)

    assert results[0].estimated_vram_gb is None


def test_estimate_over_max_size_excluded(monkeypatch):
    calls = _patch_hub(monkeypatch, {"awq": [_model("org/too-big-70B-AWQ", ["awq"])]})

    results = search_models(quantizations=["awq"], max_size_gb=10.0)

    assert results == []
    assert calls == []


def test_unknown_size_only_fills_up_after_every_known_fit(monkeypatch):
    _patch_hub(
        monkeypatch,
        {
            "awq": [
                _model("org/unknown-size-AWQ", ["awq"], downloads=1_000_000),
                _model("org/known-a-1B-AWQ", ["awq"], downloads=1000),
                _model("org/known-b-1B-AWQ", ["awq"], downloads=100),
            ]
        },
    )

    two = search_models(quantizations=["awq"], max_size_gb=100.0, max_results=2)
    three = search_models(quantizations=["awq"], max_size_gb=100.0, max_results=3)

    assert [c.repo_id for c in two] == ["org/known-a-1B-AWQ", "org/known-b-1B-AWQ"]
    assert [c.repo_id for c in three] == [
        "org/known-a-1B-AWQ",
        "org/known-b-1B-AWQ",
        "org/unknown-size-AWQ",
    ]


def test_unknown_size_excluded_when_max_size_is_explicit(monkeypatch):
    _patch_hub(monkeypatch, {"awq": [_model("org/no-size-marker-AWQ", ["awq"])]})

    results = search_models(quantizations=["awq"], max_size_gb=10.0, exclude_unknown_size=True)

    assert results == []


def test_quantization_match_via_tag_or_repo_id_suffix(monkeypatch):
    _patch_hub(
        monkeypatch,
        {
            "gptq": [
                _model("org/tagged-7B", ["gptq"]),
                _model("org/suffix-7B-GPTQ", []),
                _model("org/not-quantized-7B", ["fp16"]),
            ]
        },
    )

    results = search_models(quantizations=["gptq"], max_size_gb=100.0)

    assert {c.repo_id for c in results} == {"org/tagged-7B", "org/suffix-7B-GPTQ"}


def test_gguf_excluded_via_tag(monkeypatch):
    _patch_hub(
        monkeypatch,
        {
            "nvfp4": [
                _model("cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF", ["nvfp4", "gguf"]),
                _model("org/real-27B-NVFP4", ["nvfp4"]),
            ]
        },
    )

    results = search_models(quantizations=["nvfp4"], max_size_gb=100.0)

    assert {c.repo_id for c in results} == {"org/real-27B-NVFP4"}


def test_gguf_excluded_via_repo_id_suffix_without_tag(monkeypatch):
    _patch_hub(monkeypatch, {"nvfp4": [_model("org/model-27B-NVFP4-GGUF", ["nvfp4"])]})

    results = search_models(quantizations=["nvfp4"], max_size_gb=100.0)

    assert results == []


def test_deduplicates_across_quantization_searches(monkeypatch):
    same = _model("org/dual-tagged-7B", ["awq", "gptq"])
    _patch_hub(monkeypatch, {"awq": [same], "gptq": [same]})

    results = search_models(quantizations=["awq", "gptq"], max_size_gb=100.0)

    assert len(results) == 1


def test_max_results_caps_output(monkeypatch):
    candidates = [_model(f"org/model-{i}-AWQ", ["awq"], downloads=i) for i in range(5)]
    _patch_hub(monkeypatch, {"awq": candidates})

    results = search_models(quantizations=["awq"], max_size_gb=100.0, max_results=2)

    assert len(results) == 2


def test_ranking_prefers_more_downloads_and_more_params(monkeypatch):
    _patch_hub(
        monkeypatch,
        {
            "awq": [
                _model(
                    "org/small-AWQ",
                    ["awq"],
                    params_billion=1.0,
                    downloads=10,
                    downloads_all_time=10,
                ),
                _model(
                    "org/large-popular-AWQ",
                    ["awq"],
                    params_billion=9.0,
                    downloads=10000,
                    downloads_all_time=100000,
                ),
            ]
        },
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert results[0].repo_id == "org/large-popular-AWQ"


def test_ranking_by_downloads_survives_an_outlier_and_ignores_recency(monkeypatch):
    _patch_hub(
        monkeypatch,
        {
            "awq": [
                _model(
                    "org/few-downloads-just-updated-AWQ",
                    ["awq"],
                    params_billion=9.0,
                    downloads=500,
                    downloads_all_time=500,
                    last_modified=datetime(2026, 9, 20, tzinfo=UTC),
                ),
                _model(
                    "org/popular-AWQ",
                    ["awq"],
                    params_billion=9.0,
                    downloads=500_000,
                    downloads_all_time=500_000,
                    last_modified=datetime(2026, 8, 1, tzinfo=UTC),
                ),
                _model(
                    "org/outlier-AWQ",
                    ["awq"],
                    params_billion=9.0,
                    downloads=10_000_000,
                    downloads_all_time=100_000_000,
                    last_modified=datetime(2026, 8, 1, tzinfo=UTC),
                ),
            ]
        },
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert [r.repo_id for r in results] == [
        "org/outlier-AWQ",
        "org/popular-AWQ",
        "org/few-downloads-just-updated-AWQ",
    ]


def test_ranking_measures_params_against_max_params_when_given(monkeypatch):
    # Relative to the largest candidate, 1B vs 2B is a big fit gap that
    # outweighs 1M vs 700k downloads; against a 100B ceiling it's a
    # negligible one, so downloads decide instead.
    _patch_hub(
        monkeypatch,
        {
            "awq": [
                _model(
                    "org/popular-1B-AWQ",
                    ["awq"],
                    downloads=1_000_000,
                    downloads_all_time=1_000_000,
                ),
                _model(
                    "org/larger-2B-AWQ",
                    ["awq"],
                    downloads=700_000,
                    downloads_all_time=700_000,
                ),
            ]
        },
    )

    without_ceiling = search_models(quantizations=["awq"], max_size_gb=100.0)
    with_ceiling = search_models(quantizations=["awq"], max_size_gb=100.0, max_params_billion=100.0)

    assert without_ceiling[0].repo_id == "org/larger-2B-AWQ"
    assert with_ceiling[0].repo_id == "org/popular-1B-AWQ"


def test_ranking_lets_a_much_more_popular_model_beat_a_larger_one(monkeypatch):
    # Real case: a 92.7B repo with 23.7k downloads outranked 27B ones with millions.
    _patch_hub(
        monkeypatch,
        {
            "nvfp4": [
                _model(
                    "local-inference-lab/Qwen3.8-Flash-Next-NVFP4",
                    ["nvfp4"],
                    params_billion=92.7,
                    downloads=23_700,
                    downloads_all_time=23_700,
                ),
                _model(
                    "unsloth/Qwen3.8-27B-NVFP4",
                    ["nvfp4"],
                    downloads=3_200_000,
                    downloads_all_time=4_700_000,
                ),
            ]
        },
    )

    results = search_models(quantizations=["nvfp4"], max_size_gb=1000.0)

    assert results[0].repo_id == "unsloth/Qwen3.8-27B-NVFP4"


def test_downloads_score_runs_from_the_floor_to_the_most_downloaded_result():
    assert discovery._downloads_score(10_000, 1_000_000) == 0.0
    assert discovery._downloads_score(500, 1_000_000) == 0.0
    assert discovery._downloads_score(5_000, 8_000) == 0.0
    assert discovery._downloads_score(100_000, 1_000_000) == pytest.approx(0.5, abs=0.001)
    assert discovery._downloads_score(1_000_000, 1_000_000) == 1.0


def test_rarely_downloaded_models_come_after_every_popular_one(monkeypatch):
    _patch_hub(
        monkeypatch,
        {
            "nvfp4": [
                _model(
                    "huginnfork/Qwen3.8-Flash-Next-NVFP4-Abliterated",
                    ["nvfp4"],
                    params_billion=92.7,
                    downloads=512,
                    downloads_all_time=512,
                ),
                _model(
                    "org/Qwen3.8-27B-NVFP4", ["nvfp4"], downloads=10_000, downloads_all_time=12_000
                ),
            ]
        },
    )

    results = search_models(quantizations=["nvfp4"], max_size_gb=1000.0)

    assert [c.repo_id for c in results] == [
        "org/Qwen3.8-27B-NVFP4",
        "huginnfork/Qwen3.8-Flash-Next-NVFP4-Abliterated",
    ]


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


def test_query_shaped_like_a_repo_id_falls_back_to_a_direct_lookup(monkeypatch):
    """The fuzzy search finding nothing for an exact `org/repo` query
    (a real, observed case - the Hub's own search doesn't tokenize a
    pasted repo id the way it tokenizes ordinary free text) shouldn't
    be the end of the story - a direct lookup is authoritative."""
    _patch_hub(
        monkeypatch,
        {},
        other_repos=[_model("org/exact-repo-nvfp4", ["nvfp4"])],
    )

    results = search_models(
        quantizations=["nvfp4"], max_size_gb=100.0, query="org/exact-repo-nvfp4"
    )

    assert [c.repo_id for c in results] == ["org/exact-repo-nvfp4"]


def test_query_shaped_like_a_repo_id_skips_direct_lookup_when_search_already_found_it(
    monkeypatch,
):
    found = _model("org/exact-repo-nvfp4", ["nvfp4"])
    calls = _patch_hub(monkeypatch, {"nvfp4 org/exact-repo-nvfp4": [found]})

    results = search_models(
        quantizations=["nvfp4"], max_size_gb=100.0, query="org/exact-repo-nvfp4"
    )

    assert [c.repo_id for c in results] == ["org/exact-repo-nvfp4"]
    assert not any(kwargs.get("expand") == discovery._EXPAND for _, kwargs in calls)


def test_ordinary_multi_word_query_never_triggers_a_direct_lookup(monkeypatch):
    calls = _patch_hub(monkeypatch, {})

    results = search_models(quantizations=["nvfp4"], max_size_gb=100.0, query="qwen 3.8")

    assert results == []
    assert calls == []


def test_direct_lookup_failure_falls_back_to_empty_results(monkeypatch):
    _patch_hub(monkeypatch, {})

    results = search_models(quantizations=["nvfp4"], max_size_gb=100.0, query="org/does-not-exist")

    assert results == []
