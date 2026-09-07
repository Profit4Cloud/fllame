"""Searches the HF Hub for models plausibly servable within a given
per-quantization size ceiling, ranked best-first - a coarse pre-filter
and heuristic, not a benchmarked guarantee that a candidate actually
fits. See `fllame/models/sizing.py` for where the ceiling comes from.
A candidate's param count comes from the Hub's own safetensors metadata
where available (see `_params_billion`), falling back to a unit-aware
guess from the repo_id's naming convention only when it isn't.
`_estimated_vram_gb` derives a separate, independent minimum
weights-only VRAM figure straight from that same safetensors metadata's
per-dtype byte breakdown, for display alongside `params_billion` - not
a substitute for it, and not the deferred recipe-level VRAM estimator.
Adapted from Profit4Cloud's brainzz-documents admin UI, trimmed to
fllame's vLLM-only scope: no model-weight-format (GGUF/MLX) filtering,
since fllame only ever serves via vLLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from huggingface_hub import ModelInfo, list_models

_MULTIPLIER_BILLION_PARAMS = {"T": 1000.0, "B": 1.0, "M": 0.001}
_SIZE_UNIT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)([TBM])", re.IGNORECASE)

# Byte width of every dtype the safetensors format itself defines (a
# fixed spec, not something that grows with new quantization methods -
# a quantization scheme with no dtype of its own, e.g. 4-bit formats
# packing two values per byte, reuses one of these container dtypes for
# its tensors and is sized correctly here precisely because this counts
# physical stored bytes, not logical parameters).
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

# `list_models(search=...)` is a paginated Hub API call; without a
# `limit` it transparently fetches every matching page before this
# module gets to filter/rank anything, even though only `max_results`
# ever survives. One search per quantization is unavoidable - the Hub's
# search has no "match any of these tags" query - but each is capped at
# a multiple of `max_results`, trading a small chance of missing a
# stray lower-ranked match for not paging a search term's entire result
# set on every call.
_SEARCH_LIMIT_PER_QUANTIZATION_MULTIPLIER = 10

# Only properties in this list come back populated at all (verified
# against the installed huggingface_hub: `expand` is exclusive, not
# additive - omitting a field here means it's None on every result).
# "safetensors" rides along in the same search call (no extra request
# per candidate) and gives the real total tensor element count the Hub
# itself computes - the same figure shown as a repo's "Model size" - so
# it's preferred over guessing from the repo_id below.
_EXPAND = ["tags", "downloads", "downloadsAllTime", "lastModified", "safetensors"]

# Ranking weights - a rough heuristic: reward a candidate for being
# close to (but under) its quantization's size ceiling, recently
# updated, and widely downloaded. Downloads also stand in for
# reputation, which isn't otherwise derivable from Hub metadata.
_SIZE_CLOSENESS_WEIGHT = 0.4
_DOWNLOADS_ALL_TIME_WEIGHT = 0.2
_DOWNLOADS_RECENT_WEIGHT = 0.2
_RECENCY_WEIGHT = 0.2

# Size-closeness score for a candidate whose param count couldn't be
# determined (no safetensors metadata and no parseable repo_id).
# Deliberately 0.0, not a neutral average: missing evidence isn't a
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


@dataclass(frozen=True)
class _Scored:
    candidate: ModelCandidate
    ceiling_billion: float


def search_models(
    *,
    quantizations: list[str],
    ceiling_billion: dict[str, float],
    min_params_billion: float | None = None,
    max_params_billion: float | None = None,
    query: str | None = None,
    max_results: int = 20,
) -> list[ModelCandidate]:
    """One Hub search per entry in `quantizations`, ranked and capped at
    `max_results` overall. `ceiling_billion` is this search's
    per-quantization size ceiling (see `fllame/models/sizing.py`);
    `max_params_billion`, if given, tightens it further - explicit
    intent always wins over the estimate.

    Leaving both `min_params_billion` and `max_params_billion` unset
    means "no size constraint": a candidate whose param count couldn't
    be determined still passes, since nothing was asked of it. The
    moment either bound is given, an undetermined size no longer
    passes - it can't be shown to satisfy a range that was explicitly
    asked for.
    """
    exclude_unknown_size = min_params_billion is not None or max_params_billion is not None
    effective_min = min_params_billion if min_params_billion is not None else 0.0
    effective_max = {
        q: (ceiling if max_params_billion is None else min(ceiling, max_params_billion))
        for q, ceiling in ceiling_billion.items()
    }

    found: list[_Scored] = []
    seen_ids: set[str] = set()
    search_limit = max_results * _SEARCH_LIMIT_PER_QUANTIZATION_MULTIPLIER

    for q in quantizations:
        search_term = q if query is None else f"{q} {query}"
        for info in list_models(search=search_term, expand=_EXPAND, limit=search_limit):
            if info.id in seen_ids:
                continue
            if not _matches_quantization(info.id, info.tags, q):
                continue

            declared_size = _params_billion(info)
            if declared_size is None:
                if exclude_unknown_size:
                    continue
            elif declared_size < effective_min or declared_size > effective_max[q]:
                continue

            seen_ids.add(info.id)
            found.append(
                _Scored(
                    candidate=ModelCandidate(
                        repo_id=info.id,
                        quantization=q,
                        params_billion=declared_size,
                        estimated_vram_gb=_estimated_vram_gb(info),
                        downloads=info.downloads,
                        downloads_all_time=info.downloads_all_time,
                        last_modified=info.last_modified,
                    ),
                    ceiling_billion=effective_max[q],
                )
            )

    ranked = _rank(found)
    return [scored.candidate for scored in ranked[:max_results]]


def _matches_quantization(repo_id: str, tags: list[str] | None, quantization: str) -> bool:
    """A matching Hub tag where present, falling back to the repo_id
    naming convention this quantization method is commonly published
    under (e.g. "...-AWQ")."""
    tag_set = {tag.lower() for tag in (tags or [])}
    if quantization.lower() in tag_set:
        return True
    upper_id = repo_id.upper()
    upper_quantization = quantization.upper()
    return upper_id.endswith((f"-{upper_quantization}", f"_{upper_quantization}"))


def _params_billion(info: ModelInfo) -> float | None:
    """The candidate's real parameter count where the Hub can supply
    one, falling back to guessing from the repo_id's naming convention
    only when it can't.

    `info.safetensors.total` is the Hub's own tensor-element count for
    the repo (the same figure shown on the model page as "Model size"),
    populated by requesting `"safetensors"` in `_EXPAND` - authoritative
    where present, unlike anything derivable from the repo_id. It's
    unavailable for repos with no safetensors weights (e.g. GGUF-only
    exports), which is the only case the repo_id guess below exists for.
    A repo_id often carries more than one size-shaped number - a version
    (`Qwen3.8`), a total parameter count (`2.4T`), an active-parameter
    count for an MoE model (`A95B`) - so even the improved unit-aware
    guess stays a heuristic, not a guarantee.
    """
    if info.safetensors is not None:
        return info.safetensors.total / 1_000_000_000

    match = _SIZE_UNIT_PATTERN.search(info.id)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2).upper()
    return value * _MULTIPLIER_BILLION_PARAMS[unit]


def _estimated_vram_gb(info: ModelInfo) -> float | None:
    """A minimum weights-only VRAM estimate - the checkpoint's real
    on-disk byte size, summed directly from `safetensors.parameters`
    (a `{dtype: element_count}` breakdown) rather than going through
    `params_billion` and a per-quantization bytes-per-param guess.

    This sidesteps the ambiguity `_params_billion` can't fully resolve
    for packed quantization formats (see its docstring): a dtype tag
    and its element count together give the exact stored byte size for
    that group of tensors regardless of how many logical parameters
    happen to be packed into each stored element, so no per-quantization
    assumption is needed here at all.

    Deliberately not the recipe-level VRAM estimator tracked as
    still-deferred in CLAUDE.md: weights only, no KV cache/activations/
    concurrency, and it exists for `model scan`'s display only - `serve`
    doesn't consult it. `None` when there's no safetensors metadata to
    sum, or when a tensor's dtype isn't one this module knows the byte
    width of - a partial sum would silently understate the real size.
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


def _rank(scored: list[_Scored]) -> list[_Scored]:
    if not scored:
        return []

    downloads_recent = _normalize([s.candidate.downloads or 0 for s in scored])
    downloads_all_time = _normalize([s.candidate.downloads_all_time or 0 for s in scored])
    recency = _normalize(
        [
            s.candidate.last_modified.timestamp() if s.candidate.last_modified else 0.0
            for s in scored
        ]
    )

    def size_closeness(s: _Scored) -> float:
        if s.candidate.params_billion is None or s.ceiling_billion <= 0:
            return _UNKNOWN_SIZE_SCORE
        return min(1.0, s.candidate.params_billion / s.ceiling_billion)

    weighted = [
        (
            _SIZE_CLOSENESS_WEIGHT * size_closeness(s)
            + _DOWNLOADS_ALL_TIME_WEIGHT * downloads_all_time[i]
            + _DOWNLOADS_RECENT_WEIGHT * downloads_recent[i]
            + _RECENCY_WEIGHT * recency[i],
            s,
        )
        for i, s in enumerate(scored)
    ]
    weighted.sort(key=lambda pair: pair[0], reverse=True)
    return [s for _, s in weighted]


def _normalize(values: list[float]) -> list[float]:
    lowest, highest = min(values), max(values)
    if highest == lowest:
        return [0.5] * len(values)
    return [(value - lowest) / (highest - lowest) for value in values]
