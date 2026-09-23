import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from huggingface_hub.errors import HfHubHTTPError

from fllame.models import discovery
from fllame.models.discovery import search_models

_GIB = 1024**3
_MIN_OVERHEAD_GB = discovery._MIN_KV_CACHE_GB + discovery._MIN_RUNTIME_OVERHEAD_GB


@dataclass
class _FakeSafeTensorsInfo:
    total: int
    parameters: dict[str, int] = field(default_factory=dict)


@dataclass
class _FakeSibling:
    rfilename: str
    size: int | None


@dataclass
class _FakeModelInfo:
    id: str
    tags: list[str] = field(default_factory=list)
    downloads: int | None = 0
    downloads_all_time: int | None = 0
    last_modified: datetime | None = None
    safetensors: _FakeSafeTensorsInfo | None = None
    siblings: list[_FakeSibling] = field(default_factory=list)
    sha: str | None = None


def _model(
    id: str,
    tags: list[str],
    *,
    params_billion: float | None = None,
    weights_gb: float | None = None,
    base_model: str | None = None,
    **kwargs,
) -> _FakeModelInfo:
    if base_model is not None:
        tags = [*tags, f"base_model:{base_model}", f"base_model:quantized:{base_model}"]
    return _FakeModelInfo(
        id=id,
        tags=tags,
        safetensors=(
            _FakeSafeTensorsInfo(total=round(params_billion * 1_000_000_000))
            if params_billion is not None
            else None
        ),
        siblings=(
            [_FakeSibling("model.safetensors", round(weights_gb * _GIB))]
            if weights_gb is not None
            else []
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
                _model("org/tiny-AWQ", ["awq"], params_billion=1.0),
                _model("org/mid-AWQ", ["awq"], params_billion=7.0),
                _model("org/huge-AWQ", ["awq"], params_billion=70.0),
            ]
        },
    )

    results = search_models(
        quantizations=["awq"],
        max_size_gb=1000.0,
        min_params_billion=5.0,
        max_params_billion=10.0,
    )

    assert [c.repo_id for c in results] == ["org/mid-AWQ"]


def test_unknown_params_passes_with_no_bounds(monkeypatch):
    _patch_hub(monkeypatch, {"awq": [_model("org/mid-7B-AWQ", ["awq"])]})

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert len(results) == 1
    assert results[0].params_billion is None  # the "7B" in the repo_id isn't trusted


def test_unknown_params_excluded_once_a_bound_is_given(monkeypatch):
    _patch_hub(monkeypatch, {"awq": [_model("org/mid-7B-AWQ", ["awq"])]})

    results = search_models(quantizations=["awq"], max_size_gb=100.0, min_params_billion=1.0)

    assert results == []


def test_params_come_from_the_base_model_not_the_quantized_repo(monkeypatch):
    # Real case: a 27B AWQ build whose own Hub total counts packed values.
    _patch_hub(
        monkeypatch,
        {
            "awq": [
                _model(
                    "barrydeen/Qwen3.8-27B-AWQ-4bit",
                    ["awq"],
                    params_billion=11.08,
                    base_model="Qwen/Qwen3.8-27B",
                )
            ]
        },
        other_repos=[_model("Qwen/Qwen3.8-27B", [], params_billion=27.78)],
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert results[0].params_billion == 27.78


def test_params_fall_back_to_own_total_without_a_base_model_tag(monkeypatch):
    _patch_hub(monkeypatch, {"fp8": [_model("Qwen/Qwen3.8-27B-FP8", ["fp8"], params_billion=27.8)]})

    results = search_models(quantizations=["fp8"], max_size_gb=100.0)

    assert results[0].params_billion == 27.8


def test_params_fall_back_to_own_total_when_the_base_model_has_no_count(monkeypatch):
    _patch_hub(
        monkeypatch,
        {"awq": [_model("org/q-AWQ", ["awq"], params_billion=11.0, base_model="org/gone")]},
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert results[0].params_billion == 11.0


def test_merge_of_several_base_models_uses_own_total(monkeypatch):
    merge = _model("org/merge-AWQ", ["awq"], params_billion=11.0)
    merge.tags += ["base_model:org/a", "base_model:org/b"]
    _patch_hub(
        monkeypatch,
        {"awq": [merge]},
        other_repos=[
            _model("org/a", [], params_billion=27.0),
            _model("org/b", [], params_billion=27.0),
        ],
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert results[0].params_billion == 11.0


def test_each_base_model_is_looked_up_once(monkeypatch):
    calls = _patch_hub(
        monkeypatch,
        {
            "awq": [
                _model(f"org/quant-{i}-AWQ", ["awq"], base_model="Qwen/Qwen3.8-27B")
                for i in range(3)
            ]
        },
        other_repos=[_model("Qwen/Qwen3.8-27B", [], params_billion=27.78)],
    )

    search_models(quantizations=["awq"], max_size_gb=100.0)

    assert [repo_id for repo_id, _ in calls].count("Qwen/Qwen3.8-27B") == 1


def test_estimated_vram_is_weight_files_plus_minimum_overhead(monkeypatch):
    sharded = _FakeModelInfo(
        id="org/sharded-AWQ",
        tags=["awq"],
        siblings=[
            _FakeSibling("model-00001-of-00002.safetensors", 6 * _GIB),
            _FakeSibling("model-00002-of-00002.safetensors", 4 * _GIB),
            _FakeSibling("model.safetensors.index.json", 1000),
            _FakeSibling("pytorch_model.bin", 99 * _GIB),
            _FakeSibling("original/model.safetensors", 99 * _GIB),
        ],
    )
    _patch_hub(monkeypatch, {"awq": [sharded]})

    def fail_if_called(repo_id, *, revision):
        raise AssertionError("a single weight set needs no index lookup")

    monkeypatch.setattr(discovery, "_indexed_weight_files", fail_if_called)

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert results[0].estimated_vram_gb == 10.0 + _MIN_OVERHEAD_GB


def test_estimated_vram_reads_the_index_when_several_weight_sets_exist(monkeypatch):
    both_formats = _FakeModelInfo(
        id="org/both-formats-FP8",
        tags=["fp8"],
        sha="abc123",
        siblings=[
            _FakeSibling("consolidated.safetensors", 10 * _GIB),
            _FakeSibling("model-00001-of-00002.safetensors", 6 * _GIB),
            _FakeSibling("model-00002-of-00002.safetensors", 4 * _GIB),
            _FakeSibling("model.safetensors.index.json", 1000),
        ],
    )
    _patch_hub(monkeypatch, {"fp8": [both_formats]})
    indexed_lookups = []

    def fake_indexed_weight_files(repo_id, *, revision):
        indexed_lookups.append((repo_id, revision))
        return {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}

    monkeypatch.setattr(discovery, "_indexed_weight_files", fake_indexed_weight_files)

    results = search_models(quantizations=["fp8"], max_size_gb=100.0)

    assert results[0].estimated_vram_gb == 10.0 + _MIN_OVERHEAD_GB
    assert indexed_lookups == [("org/both-formats-FP8", "abc123")]


def test_estimated_vram_unknown_without_safetensors_files(monkeypatch):
    bin_only = _FakeModelInfo(
        id="org/bin-only-AWQ", tags=["awq"], siblings=[_FakeSibling("pytorch_model.bin", _GIB)]
    )
    _patch_hub(monkeypatch, {"awq": [bin_only]})

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert results[0].estimated_vram_gb is None


def test_known_vram_over_max_size_excluded(monkeypatch):
    _patch_hub(monkeypatch, {"awq": [_model("org/too-big-AWQ", ["awq"], weights_gb=20.0)]})

    results = search_models(quantizations=["awq"], max_size_gb=10.0)

    assert results == []


def test_minimum_overhead_counts_against_max_size(monkeypatch):
    _patch_hub(monkeypatch, {"awq": [_model("org/weights-fit-AWQ", ["awq"], weights_gb=9.0)]})

    results = search_models(quantizations=["awq"], max_size_gb=10.0)

    assert results == []


def test_unknown_vram_passes_when_max_size_is_a_default(monkeypatch):
    _patch_hub(monkeypatch, {"awq": [_model("org/no-safetensors-AWQ", ["awq"])]})

    results = search_models(quantizations=["awq"], max_size_gb=10.0)

    assert len(results) == 1


def test_unknown_vram_excluded_when_max_size_is_explicit(monkeypatch):
    _patch_hub(monkeypatch, {"awq": [_model("org/no-safetensors-AWQ", ["awq"])]})

    results = search_models(quantizations=["awq"], max_size_gb=10.0, exclude_unknown_size=True)

    assert results == []


def test_sizing_walks_the_ranked_list_only_until_enough_fit(monkeypatch):
    # Ranked by downloads: the top three are too big, so the first two
    # that fit are #4 and #5. Sized two at a time, #7 and #8 never are.
    candidates = [
        _model(
            f"org/rank-{rank}-AWQ",
            ["awq"],
            weights_gb=50.0 if rank <= 3 else 5.0,
            downloads=10 ** (9 - rank),
        )
        for rank in range(1, 9)
    ]
    calls = _patch_hub(monkeypatch, {"awq": candidates})

    results = search_models(quantizations=["awq"], max_size_gb=10.0, max_results=2)

    assert [c.repo_id for c in results] == ["org/rank-4-AWQ", "org/rank-5-AWQ"]
    sized = {repo_id for repo_id, kwargs in calls if kwargs.get("files_metadata")}
    assert sized == {f"org/rank-{rank}-AWQ" for rank in range(1, 7)}


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


def test_ranking_rewards_params_not_vram(monkeypatch):
    _patch_hub(
        monkeypatch,
        {
            "awq": [
                _model(
                    "org/more-vram-fewer-params-AWQ", ["awq"], params_billion=1.0, weights_gb=4.0
                ),
                _model(
                    "org/less-vram-more-params-AWQ", ["awq"], params_billion=2.0, weights_gb=2.0
                ),
            ]
        },
    )

    results = search_models(quantizations=["awq"], max_size_gb=100.0)

    assert results[0].repo_id == "org/less-vram-more-params-AWQ"


def test_ranking_measures_params_against_max_params_when_given(monkeypatch):
    # Relative to the largest candidate, 1B vs 2B is a big fit gap that
    # outweighs 1000 vs 100 downloads; against a 100B ceiling it's a
    # negligible one, so downloads decide instead.
    _patch_hub(
        monkeypatch,
        {
            "awq": [
                _model(
                    "org/popular-1B-AWQ",
                    ["awq"],
                    params_billion=1.0,
                    downloads=1000,
                    downloads_all_time=1000,
                ),
                _model(
                    "org/larger-2B-AWQ",
                    ["awq"],
                    params_billion=2.0,
                    downloads=100,
                    downloads_all_time=100,
                ),
            ]
        },
    )

    without_ceiling = search_models(quantizations=["awq"], max_size_gb=100.0)
    with_ceiling = search_models(quantizations=["awq"], max_size_gb=100.0, max_params_billion=100.0)

    assert without_ceiling[0].repo_id == "org/larger-2B-AWQ"
    assert with_ceiling[0].repo_id == "org/popular-1B-AWQ"


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


def test_repo_sizing_runs_concurrently(monkeypatch):
    per_call_delay = 0.2
    candidates = [_model(f"org/model-{i}-AWQ", ["awq"], weights_gb=1.0) for i in range(5)]
    _patch_hub(monkeypatch, {"awq": candidates})
    fake_model_info = discovery.model_info

    def slow_model_info(repo_id, **kwargs):
        time.sleep(per_call_delay)
        return fake_model_info(repo_id, **kwargs)

    monkeypatch.setattr(discovery, "model_info", slow_model_info)

    start = time.monotonic()
    search_models(quantizations=["awq"], max_size_gb=100.0)
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
        other_repos=[_model("org/exact-repo-nvfp4", ["nvfp4"], weights_gb=20.0)],
    )

    results = search_models(
        quantizations=["nvfp4"], max_size_gb=100.0, query="org/exact-repo-nvfp4"
    )

    assert [c.repo_id for c in results] == ["org/exact-repo-nvfp4"]


def test_query_shaped_like_a_repo_id_skips_direct_lookup_when_search_already_found_it(
    monkeypatch,
):
    found = _model("org/exact-repo-nvfp4", ["nvfp4"], weights_gb=20.0)
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
