"""Searches the HF Hub for models within an estimated-VRAM ceiling,
ranked best-first - a coarse heuristic, not a benchmarked guarantee.
"""

from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass
from datetime import datetime

from huggingface_hub import ModelInfo, list_models

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
# params-closeness when both are active (see `_fit_score`), not two
# separate weights - so a params bound never out-weighs
# popularity/recency on its own.
_FIT_WEIGHT = 0.4
_DOWNLOADS_ALL_TIME_WEIGHT = 0.2
_DOWNLOADS_RECENT_WEIGHT = 0.2
_RECENCY_WEIGHT = 0.2

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


def _fetch_quantization(q: str, *, query: str | None, search_limit: int) -> list[ModelInfo]:
    """Consumed eagerly, not returned as `list_models`'s own lazy
    generator - a lazy result fires its HTTP requests on whichever
    thread iterates it, defeating the point of calling this inside a
    worker thread."""
    search_term = q if query is None else f"{q} {query}"
    return list(list_models(search=search_term, expand=_EXPAND, limit=search_limit))


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
    if not candidates:
        return []

    downloads_recent = _normalize([c.downloads or 0 for c in candidates])
    downloads_all_time = _normalize([c.downloads_all_time or 0 for c in candidates])
    recency = _normalize(
        [c.last_modified.timestamp() if c.last_modified else 0.0 for c in candidates]
    )

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

    weighted = [
        (
            _FIT_WEIGHT * fit_score(c)
            + _DOWNLOADS_ALL_TIME_WEIGHT * downloads_all_time[i]
            + _DOWNLOADS_RECENT_WEIGHT * downloads_recent[i]
            + _RECENCY_WEIGHT * recency[i],
            c,
        )
        for i, c in enumerate(candidates)
    ]
    weighted.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in weighted]


def _normalize(values: list[float]) -> list[float]:
    lowest, highest = min(values), max(values)
    if highest == lowest:
        return [0.5] * len(values)
    return [(value - lowest) / (highest - lowest) for value in values]
