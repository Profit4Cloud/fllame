"""Searches the HF Hub for models plausibly servable within a given
per-quantization size ceiling, ranked best-first - a coarse pre-filter
and heuristic, not a benchmarked guarantee that a candidate actually
fits. See `fllame/models/sizing.py` for where the ceiling comes from.
Adapted from Profit4Cloud's brainzz-documents admin UI, trimmed to
fllame's vLLM-only scope: no model-weight-format (GGUF/MLX) filtering,
since fllame only ever serves via vLLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from huggingface_hub import list_models

_DECLARED_SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)B", re.IGNORECASE)

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
_EXPAND = ["tags", "downloads", "downloadsAllTime", "lastModified"]

# Ranking weights - a rough heuristic: reward a candidate for being
# close to (but under) its quantization's size ceiling, recently
# updated, and widely downloaded. Downloads also stand in for
# reputation, which isn't otherwise derivable from Hub metadata.
_SIZE_CLOSENESS_WEIGHT = 0.4
_DOWNLOADS_ALL_TIME_WEIGHT = 0.2
_DOWNLOADS_RECENT_WEIGHT = 0.2
_RECENCY_WEIGHT = 0.2

# Size-closeness score for a candidate whose declared size couldn't be
# parsed from its repo_id. Deliberately 0.0, not a neutral average: an
# unparseable size is missing evidence, not a known middling fit.
_UNKNOWN_SIZE_SCORE = 0.0


@dataclass(frozen=True)
class ModelCandidate:
    repo_id: str
    quantization: str
    params_billion: float | None
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
    means "no size constraint": a candidate whose size couldn't be
    parsed from its repo_id still passes, since nothing was asked of
    it. The moment either bound is given, an unparseable size no longer
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

            declared_size = _declared_size_billion_params(info.id)
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


def _declared_size_billion_params(repo_id: str) -> float | None:
    match = _DECLARED_SIZE_PATTERN.search(repo_id)
    return float(match.group(1)) if match else None


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
