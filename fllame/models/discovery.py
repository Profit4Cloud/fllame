"""Searches the HF Hub for models within an estimated-VRAM ceiling,
ranked best-first - a coarse heuristic, not a benchmarked guarantee.
"""

from __future__ import annotations

import concurrent.futures
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from typing import TypeVar

from huggingface_hub import ModelInfo, hf_hub_url, list_models, model_info
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status
from requests.exceptions import RequestException

_T = TypeVar("_T")
_R = TypeVar("_R")

# `list_models(search=...)` fetches every matching page without a
# `limit`, even though only `max_results` survives - capped at a
# multiple of it instead, trading a small chance of missing a stray
# lower-ranked match for not paging an entire result set.
_SEARCH_LIMIT_PER_QUANTIZATION_MULTIPLIER = 10

# `expand` is exclusive, not additive - a field omitted here is None on
# every result. "safetensors" rides the same request and gives the
# Hub's own tensor-element count ("Model size" on the model page).
_EXPAND = ["tags", "downloads", "downloadsAllTime", "lastModified", "safetensors"]

# The Hub's safetensors `total` is unreliable for a quantized repo - some
# count packed values (a 27B AWQ build reporting 11B), others unpacked -
# so a repo's `base_model:` tag is followed one level to the unquantized
# original, whose count is trustworthy. The tag also comes in
# `base_model:<relation>:<id>` form; only the bare one is read here.
_BASE_MODEL_TAG_PREFIX = "base_model:"

# Real on-disk weight size, since dtype element counts can't be turned
# into bytes reliably across packing formats. The index is only
# fetched when the repo holds more than one set of weights (e.g. a
# `consolidated.safetensors` alongside sharded `model-*` files).
_SHARDED_WEIGHTS_PATTERN = re.compile(r"model(-\d+-of-\d+)?\.safetensors")
_SAFETENSORS_INDEX = "model.safetensors.index.json"

# EST. VRAM is the floor for one request at a time at this context
# length - vLLM's own default is the model's full context, often far
# longer. KV cache is sized from the model's config.json at vLLM's
# default ("auto") KV dtype, i.e. the unquantized model's BF16/FP16.
_ASSUMED_MAX_MODEL_LEN = 32768
_KV_CACHE_BYTES_PER_ELEMENT = 2

# Only used when config.json is missing or its architecture isn't one
# `_kv_cache_bytes` understands: dense GQA models land roughly between
# 4 and 16 KiB per token per billion params.
_FALLBACK_KV_BYTES_PER_TOKEN_PER_BILLION_PARAMS = 8 * 1024

# CUDA context, activation workspace and CUDA graphs.
_MIN_RUNTIME_OVERHEAD_GB = 2.0

_MAX_CONCURRENT_LOOKUPS = 16

# Fit rewards parameter count, not VRAM: bigger is better, and anything
# over the VRAM or params bound is already filtered out. No recency
# term: the recent download count already sinks a stale model, and
# `lastModified` moves on any commit, even a README edit.
_FIT_WEIGHT = 0.3
_DOWNLOADS_ALL_TIME_WEIGHT = 0.3
_DOWNLOADS_RECENT_WEIGHT = 0.4

# Downloads span orders of magnitude, so they're scored on a fixed log
# scale - linear min-max across results let one huge outlier flatten
# everyone else to ~0. Fixed rather than relative so a score doesn't
# depend on what else the search returned. All-time counts run roughly
# 10x the ~30-day ones, hence the separate ceilings.
_DOWNLOADS_RECENT_LOG10_CEILING = 7.0
_DOWNLOADS_ALL_TIME_LOG10_CEILING = 8.0

# Deliberately 0.0, not a neutral average - missing evidence isn't a
# known middling fit.
_UNKNOWN_PARAMS_SCORE = 0.0


@dataclass(frozen=True)
class ModelCandidate:
    repo_id: str
    quantization: str
    params_billion: float | None
    estimated_vram_gb: float | None
    downloads: int | None
    downloads_all_time: int | None
    last_modified: datetime | None


def search_models(
    *,
    quantizations: list[str],
    max_size_gb: float,
    exclude_unknown_size: bool = False,
    min_params_billion: float | None = None,
    max_params_billion: float | None = None,
    query: str | None = None,
    max_results: int = 20,
) -> list[ModelCandidate]:
    """`exclude_unknown_size=True` drops a candidate with no safetensors
    weight files to check against `max_size_gb` - only use it when
    `max_size_gb` is the caller's own explicit ask, since it can't be
    shown to satisfy a bound nobody asked this candidate to be measured
    against otherwise. `min_params_billion`/`max_params_billion` apply
    the same rule implicitly: unset means an undetermined param count
    still passes; given, it no longer does.
    """
    exclude_unknown_params = min_params_billion is not None or max_params_billion is not None
    effective_min_params = min_params_billion if min_params_billion is not None else 0.0
    search_limit = max_results * _SEARCH_LIMIT_PER_QUANTIZATION_MULTIPLIER

    infos_per_quantization = _map_concurrently(
        lambda q: _fetch_quantization(q, query=query, search_limit=search_limit), quantizations
    )

    matched: list[tuple[str, ModelInfo]] = []
    seen_ids: set[str] = set()
    for q, infos in zip(quantizations, infos_per_quantization, strict=True):
        for info in infos:
            if info.id in seen_ids:
                continue
            if not _matches_quantization(info.id, info.tags, q):
                continue
            if _is_gguf(info.id, info.tags):
                continue
            seen_ids.add(info.id)
            matched.append((q, info))

    base_ids = sorted({b for _, info in matched if (b := _base_model_id(info.tags)) is not None})
    base_params = dict(
        zip(base_ids, _map_concurrently(_fetch_params_billion, base_ids), strict=True)
    )

    candidates: list[ModelCandidate] = []
    for q, info in matched:
        base_id = _base_model_id(info.tags)
        params = base_params.get(base_id) if base_id is not None else None
        if params is None:
            params = _safetensors_total_billion(info)

        if params is None:
            if exclude_unknown_params:
                continue
        elif params < effective_min_params or (
            max_params_billion is not None and params > max_params_billion
        ):
            continue

        candidates.append(
            ModelCandidate(
                repo_id=info.id,
                quantization=q,
                params_billion=params,
                estimated_vram_gb=None,
                downloads=info.downloads,
                downloads_all_time=info.downloads_all_time,
                last_modified=info.last_modified,
            )
        )

    # A quantized repo's config.json can be missing or rewritten by the
    # quantizer; the base model's describes the same architecture, and
    # is fetched once for every quant of it.
    config_repo_ids = {info.id: _base_model_id(info.tags) or info.id for _, info in matched}

    ranked = _rank(candidates, max_params_billion=max_params_billion)
    return _take_fitting(
        ranked,
        config_repo_ids=config_repo_ids,
        max_size_gb=max_size_gb,
        exclude_unknown_size=exclude_unknown_size,
        max_results=max_results,
    )


def _take_fitting(
    ranked: list[ModelCandidate],
    *,
    config_repo_ids: dict[str, str],
    max_size_gb: float,
    exclude_unknown_size: bool,
    max_results: int,
) -> list[ModelCandidate]:
    """Sizing a repo costs requests of its own, so it happens only after
    ranking, walking down the list a batch at a time until
    `max_results` fit - instead of sizing every search hit up front. An
    unknown size is a last resort, only filling what's left once the
    whole list is walked."""
    fitting: list[ModelCandidate] = []
    unknown_size: list[ModelCandidate] = []
    configs: dict[str, dict | None] = {}
    if max_results <= 0:
        return fitting
    for start in range(0, len(ranked), max_results):
        batch = ranked[start : start + max_results]
        missing_configs = sorted({config_repo_ids[c.repo_id] for c in batch} - configs.keys())
        lookups = [partial(_weight_bytes_or_none, c.repo_id) for c in batch] + [
            partial(_fetch_config, repo_id) for repo_id in missing_configs
        ]
        results = _map_concurrently(lambda lookup: lookup(), lookups)
        weights, fetched_configs = results[: len(batch)], results[len(batch) :]
        configs.update(zip(missing_configs, fetched_configs, strict=True))

        for candidate, weight_bytes in zip(batch, weights, strict=True):
            size = _estimated_vram_gb(
                weight_bytes,
                config=configs[config_repo_ids[candidate.repo_id]],
                params_billion=candidate.params_billion,
            )
            if size is None:
                if not exclude_unknown_size:
                    unknown_size.append(candidate)
            elif size <= max_size_gb:
                fitting.append(replace(candidate, estimated_vram_gb=size))
                if len(fitting) == max_results:
                    return fitting
    return fitting + unknown_size[: max_results - len(fitting)]


def _map_concurrently(fn: Callable[[_T], _R], items: Iterable[_T]) -> list[_R]:
    items = list(items)
    if not items:
        return []
    workers = min(len(items), _MAX_CONCURRENT_LOOKUPS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fn, items))


def _looks_like_repo_id(query: str) -> bool:
    """A pasted `org/repo` id doesn't tokenize the way `list_models`'s
    own free-text search expects - it's one unbroken string full of
    `/`, `-`, `.`, so a query shaped exactly like a real repo id often
    matches nothing there even when the repo plainly exists. Narrow on
    purpose (exactly one `/`, no spaces) so an ordinary multi-word query
    is never mistaken for one."""
    return query.count("/") == 1 and " " not in query


def _fetch_exact_repo_id(repo_id: str) -> ModelInfo | None:
    """A direct, authoritative lookup - bypasses the fuzzy search
    entirely, so it isn't at the mercy of how the Hub's search index
    happens to tokenize `repo_id`. `None` on anything from "doesn't
    exist" to a network hiccup: this is a supplementary fallback, not
    the primary path, so it degrades to whatever the fuzzy search
    already found rather than failing the whole command."""
    try:
        return model_info(repo_id, expand=_EXPAND)
    except (HfHubHTTPError, RequestException):
        return None


def _fetch_quantization(q: str, *, query: str | None, search_limit: int) -> list[ModelInfo]:
    """Consumed eagerly, not returned as `list_models`'s own lazy
    generator - a lazy result fires its HTTP requests on whichever
    thread iterates it, defeating the point of calling this inside a
    worker thread."""
    search_term = q if query is None else f"{q} {query}"
    results = list(list_models(search=search_term, expand=_EXPAND, limit=search_limit))
    if (
        query is not None
        and _looks_like_repo_id(query)
        and not any(info.id.lower() == query.lower() for info in results)
    ):
        direct = _fetch_exact_repo_id(query)
        if direct is not None:
            results.append(direct)
    return results


def _matches_quantization(repo_id: str, tags: list[str] | None, quantization: str) -> bool:
    tag_set = {tag.lower() for tag in (tags or [])}
    if quantization.lower() in tag_set:
        return True
    upper_id = repo_id.upper()
    upper_quantization = quantization.upper()
    return upper_id.endswith((f"-{upper_quantization}", f"_{upper_quantization}"))


def _is_gguf(repo_id: str, tags: list[str] | None) -> bool:
    """Excluded outright, not just deprioritized: GGUF-via-vLLM needs a
    separate out-of-tree plugin with no per-model compatibility
    guarantee, and carries no safetensors metadata to size in the first
    place."""
    tag_set = {tag.lower() for tag in (tags or [])}
    if "gguf" in tag_set:
        return True
    return repo_id.upper().endswith(("-GGUF", "_GGUF"))


def _base_model_id(tags: list[str] | None) -> str | None:
    """`None` for a merge (several bare base tags) as well as for no tag
    at all - there's no single original to take the count from."""
    bases = [
        tag.removeprefix(_BASE_MODEL_TAG_PREFIX)
        for tag in tags or []
        if tag.startswith(_BASE_MODEL_TAG_PREFIX)
        and ":" not in tag.removeprefix(_BASE_MODEL_TAG_PREFIX)
    ]
    return bases[0] if len(bases) == 1 else None


def _safetensors_total_billion(info: ModelInfo) -> float | None:
    if info.safetensors is None:
        return None
    return info.safetensors.total / 1_000_000_000


def _fetch_params_billion(repo_id: str) -> float | None:
    try:
        return _safetensors_total_billion(model_info(repo_id, expand=["safetensors"]))
    except (HfHubHTTPError, RequestException):
        return None


def _estimated_vram_gb(
    weight_bytes: int | None, *, config: dict | None, params_billion: float | None
) -> float | None:
    if weight_bytes is None:
        return None
    kv_bytes = _kv_cache_bytes(config) if config is not None else None
    if kv_bytes is None:
        if params_billion is None:
            return None
        kv_bytes = (
            _FALLBACK_KV_BYTES_PER_TOKEN_PER_BILLION_PARAMS
            * params_billion
            * _ASSUMED_MAX_MODEL_LEN
        )
    return (weight_bytes + kv_bytes) / (1024**3) + _MIN_RUNTIME_OVERHEAD_GB


def _kv_cache_bytes(config: dict) -> int | None:
    """Only attention layers hold a per-token KV cache: a linear-attention
    (Mamba/DeltaNet-style) layer keeps a small fixed state instead, and
    a sliding-window layer caches at most its window. MLA models cache
    one compressed latent per token rather than a K and V per head."""
    text_config = config.get("text_config") or config.get("llm_config") or config
    try:
        layer_count = int(text_config["num_hidden_layers"])
        layer_types = text_config.get("layer_types")
        interval = text_config.get("full_attention_interval")
        if layer_types:
            full_layers = sum(t == "full_attention" for t in layer_types)
            sliding_layers = sum(t == "sliding_attention" for t in layer_types)
        elif interval:
            full_layers, sliding_layers = layer_count // int(interval), 0
        else:
            full_layers, sliding_layers = layer_count, 0

        if text_config.get("kv_lora_rank"):
            elements_per_token_per_layer = int(text_config["kv_lora_rank"]) + int(
                text_config["qk_rope_head_dim"]
            )
        else:
            heads = int(text_config["num_attention_heads"])
            kv_heads = int(text_config.get("num_key_value_heads") or heads)
            head_dim = int(text_config.get("head_dim") or int(text_config["hidden_size"]) // heads)
            elements_per_token_per_layer = 2 * kv_heads * head_dim

        window = text_config.get("sliding_window") or _ASSUMED_MAX_MODEL_LEN
        cached_tokens = full_layers * _ASSUMED_MAX_MODEL_LEN + sliding_layers * min(
            int(window), _ASSUMED_MAX_MODEL_LEN
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return cached_tokens * elements_per_token_per_layer * _KV_CACHE_BYTES_PER_ELEMENT


def _weight_bytes_or_none(repo_id: str) -> int | None:
    try:
        return _weight_bytes(repo_id)
    except (HfHubHTTPError, RequestException, KeyError, ValueError):
        return None


def _weight_bytes(repo_id: str) -> int | None:
    info = model_info(repo_id, files_metadata=True)
    sizes = {
        sibling.rfilename: sibling.size
        for sibling in info.siblings or []
        if sibling.rfilename.endswith(".safetensors") and "/" not in sibling.rfilename
    }
    if not sizes:
        return None

    has_index = any(sibling.rfilename == _SAFETENSORS_INDEX for sibling in info.siblings or [])
    single_weight_set = all(_SHARDED_WEIGHTS_PATTERN.fullmatch(name) for name in sizes)
    if has_index and not single_weight_set:
        indexed = _indexed_weight_files(repo_id, revision=info.sha)
        sizes = {name: size for name, size in sizes.items() if name in indexed}

    if not sizes or any(size is None for size in sizes.values()):
        return None
    return sum(sizes.values())


def _indexed_weight_files(repo_id: str, *, revision: str | None) -> set[str]:
    return set(_fetch_json(repo_id, _SAFETENSORS_INDEX, revision=revision)["weight_map"].values())


def _fetch_config(repo_id: str) -> dict | None:
    try:
        config = _fetch_json(repo_id, "config.json")
    except (HfHubHTTPError, RequestException, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _fetch_json(repo_id: str, filename: str, *, revision: str | None = None):
    """Read straight off the Hub rather than via `hf_hub_download`, which
    would leave a partial repo in the HF cache for `model list` to show."""
    response = get_session().get(
        hf_hub_url(repo_id, filename, revision=revision), headers=build_hf_headers()
    )
    hf_raise_for_status(response)
    return response.json()


def _rank(
    candidates: list[ModelCandidate], *, max_params_billion: float | None
) -> list[ModelCandidate]:
    """Without `max_params_billion` there's no absolute params ceiling,
    so the largest candidate returned stands in for one."""
    known_params = [c.params_billion for c in candidates if c.params_billion is not None]
    params_ceiling = (
        max_params_billion if max_params_billion is not None else max(known_params, default=None)
    )

    def fit_score(c: ModelCandidate) -> float:
        if c.params_billion is None or not params_ceiling:
            return _UNKNOWN_PARAMS_SCORE
        return min(1.0, c.params_billion / params_ceiling)

    def score(c: ModelCandidate) -> float:
        return (
            _FIT_WEIGHT * fit_score(c)
            + _DOWNLOADS_ALL_TIME_WEIGHT
            * _downloads_score(c.downloads_all_time, _DOWNLOADS_ALL_TIME_LOG10_CEILING)
            + _DOWNLOADS_RECENT_WEIGHT
            * _downloads_score(c.downloads, _DOWNLOADS_RECENT_LOG10_CEILING)
        )

    return sorted(candidates, key=score, reverse=True)


def _downloads_score(downloads: int | None, log10_ceiling: float) -> float:
    return min(1.0, math.log10(1 + (downloads or 0)) / log10_ceiling)
