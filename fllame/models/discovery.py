"""Searches the HF Hub for models plausibly servable within a given
estimated-VRAM ceiling, ranked best-first - a coarse pre-filter and
heuristic, not a benchmarked guarantee that a candidate actually fits.
See `fllame/models/sizing.py` for where the default ceiling comes from.

Two independent restrictions can narrow a search further: `max_size_gb`
(always given - explicit, or defaulted from a hardware scan by the
caller; see `exclude_unknown_size` for what "explicit" changes) caps
`_estimated_vram_gb`; `min_params_billion`/`max_params_billion` (only
active when given - no hardware-derived default) cap `_params_billion`.
Both are pass/fail filters; both also feed the ranking's fit score,
weighted evenly when both are active.

A candidate's param count comes from the Hub's own safetensors metadata
where available (see `_params_billion`), falling back to a unit-aware
guess from the repo_id's naming convention only when it isn't.
`_estimated_vram_gb` derives a separate, independent minimum
weights-only VRAM figure straight from that same safetensors metadata's
per-dtype byte breakdown - deliberately not derived from
`params_billion` via a per-quantization bytes-per-param guess, since a
declared param count doesn't tell you how a specific quantization
format packs its bits on disk, but the Hub's own per-dtype element
counts do, directly.

GGUF results are excluded outright (see `_is_gguf`): fllame is
vLLM-only, and GGUF-via-vLLM now needs a separate out-of-tree plugin
with no per-model compatibility guarantee, on top of carrying no
safetensors metadata for the size estimate above to work with anyway.
MLX isn't filtered - fllame has no MLX serving story at all, unlike
GGUF's (limited, unreliable) one, so it wasn't the case that motivated
this and hasn't been evaluated on its own merits.

Adapted from Profit4Cloud's brainzz-documents admin UI, trimmed to
fllame's vLLM-only scope: no training/LoRA headroom.
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
# close to (but under) the active size ceiling(s), recently updated, and
# widely downloaded. Downloads also stand in for reputation, which isn't
# otherwise derivable from Hub metadata. _FIT_WEIGHT is a budget split
# between VRAM-closeness and params-closeness when both are active (see
# `_fit_score`), not two separate weights - so the two together never
# out-weigh popularity/recency just because a params bound was given.
_FIT_WEIGHT = 0.4
_DOWNLOADS_ALL_TIME_WEIGHT = 0.2
_DOWNLOADS_RECENT_WEIGHT = 0.2
_RECENCY_WEIGHT = 0.2

# Closeness score for a candidate whose relevant size couldn't be
# determined (and wasn't already excluded by `exclude_unknown_size`/an
# explicit params bound - see `search_models`). Deliberately 0.0, not a
# neutral average: missing evidence isn't a known middling fit.
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
    """One Hub search per entry in `quantizations`, ranked and capped at
    `max_results` overall.

    `max_size_gb` always caps a candidate whose estimated VRAM is known.
    `exclude_unknown_size` decides what happens to one that isn't (no
    safetensors metadata at all): pass `True` when `max_size_gb` is the
    caller's own explicit ask - it can't be shown to satisfy a bound
    that was explicitly asked for. Leave it `False` (the default) when
    `max_size_gb` is merely a hardware-scan default nobody asked this
    candidate to be measured against - excluding it here as if it were
    an explicit ask would make repos with no safetensors metadata (e.g.
    GGUF-only exports) disappear from every default, unfiltered search.

    `min_params_billion`/`max_params_billion` are a separate, optional
    restriction with no implicit default and exactly the same rule:
    leaving both unset means "no params constraint" and an undetermined
    param count still passes; the moment either is given, it no longer
    does - it can't be shown to satisfy a range that was explicitly
    asked for.
    """
    exclude_unknown_params = min_params_billion is not None or max_params_billion is not None
    effective_min_params = min_params_billion if min_params_billion is not None else 0.0

    found: list[ModelCandidate] = []
    seen_ids: set[str] = set()
    search_limit = max_results * _SEARCH_LIMIT_PER_QUANTIZATION_MULTIPLIER

    for q in quantizations:
        search_term = q if query is None else f"{q} {query}"
        for info in list_models(search=search_term, expand=_EXPAND, limit=search_limit):
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


def _is_gguf(repo_id: str, tags: list[str] | None) -> bool:
    """A GGUF Hub tag where present, falling back to the repo_id's
    common "-GGUF" naming convention - same dual-check shape as
    `_matches_quantization`.

    Excluded outright rather than merely deprioritized: GGUF-via-vLLM
    now requires a separate out-of-tree plugin fllame doesn't manage,
    is documented by vLLM itself as experimental with no per-model
    compatibility guarantee even with that plugin installed, and (being
    what motivated this filter) has no safetensors metadata for
    `_params_billion`/`_estimated_vram_gb` to size in the first place.
    A repo offering both a GGUF and a real (safetensors) release isn't
    lost here - only the GGUF listing itself is excluded, the other one
    still matches the quantization search on its own tags/name.
    """
    tag_set = {tag.lower() for tag in (tags or [])}
    if "gguf" in tag_set:
        return True
    return repo_id.upper().endswith(("-GGUF", "_GGUF"))


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
