"""Searches the HF Hub for models within an estimated-VRAM ceiling,
ranked best-first - a coarse heuristic, not a benchmarked guarantee.
"""

from __future__ import annotations

import concurrent.futures
import math
import re
from dataclasses import dataclass
from datetime import datetime

from huggingface_hub import ModelInfo, list_models, model_info
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import RequestException

_MULTIPLIER_BILLION_PARAMS = {"T": 1000.0, "B": 1.0, "M": 0.001}
_SIZE_UNIT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)([TBM])", re.IGNORECASE)

# Byte width per dtype the safetensors format defines - fixed by the
# spec, so a packed 4-bit format still sizes correctly by reusing one
# of these container dtypes.
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

# `list_models(search=...)` fetches every matching page without a
# `limit`, even though only `max_results` survives - capped at a
# multiple of it instead, trading a small chance of missing a stray
# lower-ranked match for not paging an entire result set.
_SEARCH_LIMIT_PER_QUANTIZATION_MULTIPLIER = 10

# `expand` is exclusive, not additive - a field omitted here is None on
# every result. "safetensors" rides the same request and gives the
# Hub's own tensor-element count ("Model size" on the model page).
_EXPAND = ["tags", "downloads", "downloadsAllTime", "lastModified", "safetensors"]

# _FIT_WEIGHT is a budget split between VRAM-closeness and
# params-closeness when both are active (see `fit_score`), not two
# separate weights - so a params bound never out-weighs popularity on
# its own. No recency term: the recent download count already sinks a
# stale model, and `lastModified` moves on any commit, even a README edit.
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
_UNKNOWN_SIZE_SCORE = 0.0


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
    metadata to check against `max_size_gb` - only use it when
    `max_size_gb` is the caller's own explicit ask, since it can't be
    shown to satisfy a bound nobody asked this candidate to be measured
    against otherwise. `min_params_billion`/`max_params_billion` apply
    the same rule implicitly: unset means an undetermined param count
    still passes; given, it no longer does.
    """
    exclude_unknown_params = min_params_billion is not None or max_params_billion is not None
    effective_min_params = min_params_billion if min_params_billion is not None else 0.0

    found: list[ModelCandidate] = []
    seen_ids: set[str] = set()
    search_limit = max_results * _SEARCH_LIMIT_PER_QUANTIZATION_MULTIPLIER

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(quantizations), 1)) as executor:
        infos_per_quantization = list(
            executor.map(
                lambda q: _fetch_quantization(q, query=query, search_limit=search_limit),
                quantizations,
            )
        )

    for q, infos in zip(quantizations, infos_per_quantization, strict=True):
        for info in infos:
            if info.id in seen_ids:
                continue
            if not _matches_quantization(info.id, info.tags, q):
                continue
            if _is_gguf(info.id, info.tags):
                continue

            declared_params = _params_billion(info)
            if declared_params is None:
                if exclude_unknown_params:
                    continue
            elif declared_params < effective_min_params or (
                max_params_billion is not None and declared_params > max_params_billion
            ):
                continue

            estimated_vram = _estimated_vram_gb(info)
            if estimated_vram is None:
                if exclude_unknown_size:
                    continue
            elif estimated_vram > max_size_gb:
                continue

            seen_ids.add(info.id)
            found.append(
                ModelCandidate(
                    repo_id=info.id,
                    quantization=q,
                    params_billion=declared_params,
                    estimated_vram_gb=estimated_vram,
                    downloads=info.downloads,
                    downloads_all_time=info.downloads_all_time,
                    last_modified=info.last_modified,
                )
            )

    ranked = _rank(found, max_size_gb=max_size_gb, max_params_billion=max_params_billion)
    return ranked[:max_results]


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


def _params_billion(info: ModelInfo) -> float | None:
    """Falls back to guessing from the repo_id's naming convention only
    when the Hub has no safetensors metadata - a repo_id often carries
    more than one size-shaped number (a version, a total param count, an
    MoE active-param count), so even this guess stays a heuristic.
    """
    if info.safetensors is not None:
        return info.safetensors.total / 1_000_000_000

    match = _SIZE_UNIT_PATTERN.search(info.id)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2).upper()
    return value * _MULTIPLIER_BILLION_PARAMS[unit]


def _estimated_vram_gb(info: ModelInfo) -> float | None:
    """Summed directly from `safetensors.parameters` (a `{dtype:
    element_count}` breakdown) rather than `params_billion` and a
    per-quantization bytes-per-param guess - the dtype+count already
    gives the exact stored byte size regardless of packing. `None`
    (not a partial sum) when a dtype isn't one this module knows the
    byte width of.
    """
    if info.safetensors is None:
        return None
    try:
        total_bytes = sum(
            count * _SAFETENSORS_DTYPE_BYTES[dtype.upper()]
            for dtype, count in info.safetensors.parameters.items()
        )
    except KeyError:
        return None
    return total_bytes / (1024**3)


def _rank(
    candidates: list[ModelCandidate], *, max_size_gb: float, max_params_billion: float | None
) -> list[ModelCandidate]:
    def closeness(value: float | None, ceiling: float | None) -> float:
        if value is None or ceiling is None or ceiling <= 0:
            return _UNKNOWN_SIZE_SCORE
        return min(1.0, value / ceiling)

    def fit_score(c: ModelCandidate) -> float:
        size_closeness = closeness(c.estimated_vram_gb, max_size_gb)
        if max_params_billion is None:
            return size_closeness
        params_closeness = closeness(c.params_billion, max_params_billion)
        return 0.5 * size_closeness + 0.5 * params_closeness

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
