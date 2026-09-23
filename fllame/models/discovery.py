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

from fllame.models.vram import RUNTIME_OVERHEAD_GB

# `list_models(search=...)` fetches every matching page without a
# `limit`, even though only `max_results` survives - capped at a
# multiple of it instead, trading a small chance of missing a stray
# lower-ranked match for not paging an entire result set.
_SEARCH_LIMIT_PER_QUANTIZATION_MULTIPLIER = 10

# `expand` is exclusive, not additive - a field omitted here is None on
# every result. "safetensors" rides the same request and gives the
# Hub's own tensor-element count ("Model size" on the model page).
_EXPAND = ["tags", "downloads", "downloadsAllTime", "lastModified", "safetensors"]

# The repo name is the parameter count's primary source, the way a
# person reads it: the Hub's own safetensors `total` is unreliable for a
# quantized repo (some count packed values, a 27B AWQ build reporting
# 11B), so it's only a fallback for a name without a size. The
# lookarounds keep "4bit", an MoE's active count ("A3B") and "8x7B"
# from reading as sizes.
_NAME_SIZE_PATTERN = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)([tbm])(?![a-z])", re.IGNORECASE)
_BILLION_PARAMS_PER_UNIT = {"t": 1000.0, "b": 1.0, "m": 0.001}

# Estimated weight bytes per parameter, for ranking and filtering
# without a request per repo. Quantized formats keep embeddings, the
# output head and norms in BF16, hence the margin over the nominal
# width; how much a repo leaves unquantized varies, so this is +-30%.
_WEIGHT_BYTES_PER_PARAM = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.05,
    "int8": 1.05,
    "nvfp4": 0.65,
    "fp4": 0.65,
    "mxfp4": 0.6,
    "awq": 0.6,
    "gptq": 0.6,
    "int4": 0.6,
}
# AWQ/GPTQ default to 4-bit; these name/tag markers mean 8-bit.
_EIGHT_BIT_MARKERS = ("int8", "8bit", "8-bit", "w8a16", "w8a8")
_EIGHT_BIT_WEIGHT_BYTES_PER_PARAM = 1.05

# EST. VRAM is the floor for one request at a time at this context
# length - vLLM's own default is the model's full context, often far
# longer. Dense GQA models land roughly between 4 and 16 KiB of BF16 KV
# cache per token per billion params; hybrid-attention and MLA models
# need much less, older models without GQA more.
_ASSUMED_MAX_MODEL_LEN = 32768
_KV_BYTES_PER_TOKEN_PER_BILLION_PARAMS = 8 * 1024

# Fit rewards parameter count, not VRAM: bigger is better, and anything
# over the VRAM or params bound is already filtered out. No recency
# term: the recent download count already sinks a stale model, and
# `lastModified` moves on any commit, even a README edit.
_FIT_WEIGHT = 0.3
_DOWNLOADS_ALL_TIME_WEIGHT = 0.3
_DOWNLOADS_RECENT_WEIGHT = 0.4

# Downloads span orders of magnitude, so they're scored on a log scale
# - linear let one huge outlier flatten everyone else to ~0. It runs
# from this floor (0.0) up to the most-downloaded result (1.0), so the
# download terms use their full range rather than the narrow band real
# counts occupy on a scale starting at zero downloads.
_DOWNLOADS_FLOOR = 100

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
    """`exclude_unknown_size=True` drops a candidate whose size can't be
    estimated (no parameter count, or an unknown quantization) - only
    use it when `max_size_gb` is the caller's own explicit ask, since it can't be
    shown to satisfy a bound nobody asked this candidate to be measured
    against otherwise. `min_params_billion`/`max_params_billion` apply
    the same rule implicitly: unset means an undetermined param count
    still passes; given, it no longer does.
    """
    exclude_unknown_params = min_params_billion is not None or max_params_billion is not None
    effective_min_params = min_params_billion if min_params_billion is not None else 0.0
    search_limit = max_results * _SEARCH_LIMIT_PER_QUANTIZATION_MULTIPLIER

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(quantizations), 1)) as executor:
        infos_per_quantization = list(
            executor.map(
                lambda q: _fetch_quantization(q, query=query, search_limit=search_limit),
                quantizations,
            )
        )

    candidates: list[ModelCandidate] = []
    seen_ids: set[str] = set()
    for q, infos in zip(quantizations, infos_per_quantization, strict=True):
        for info in infos:
            if info.id in seen_ids:
                continue
            if not _matches_quantization(info.id, info.tags, q):
                continue
            if _is_gguf(info.id, info.tags):
                continue

            params = _params_billion(info)
            if params is None:
                if exclude_unknown_params:
                    continue
            elif params < effective_min_params or (
                max_params_billion is not None and params > max_params_billion
            ):
                continue

            estimated_vram = _estimated_vram_gb(
                params, bytes_per_param=_weight_bytes_per_param(info.id, info.tags, q)
            )
            if estimated_vram is None:
                if exclude_unknown_size:
                    continue
            elif estimated_vram > max_size_gb:
                continue

            seen_ids.add(info.id)
            candidates.append(
                ModelCandidate(
                    repo_id=info.id,
                    quantization=q,
                    params_billion=params,
                    estimated_vram_gb=estimated_vram,
                    downloads=info.downloads,
                    downloads_all_time=info.downloads_all_time,
                    last_modified=info.last_modified,
                )
            )

    ranked = _rank(candidates, max_params_billion=max_params_billion)
    known_size = [c for c in ranked if c.estimated_vram_gb is not None]
    unknown_size = [c for c in ranked if c.estimated_vram_gb is None]
    return (known_size + unknown_size)[:max_results]


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
    match = _NAME_SIZE_PATTERN.search(info.id.split("/")[-1])
    if match:
        return float(match.group(1)) * _BILLION_PARAMS_PER_UNIT[match.group(2).lower()]
    if info.safetensors is not None:
        return info.safetensors.total / 1_000_000_000
    return None


def _weight_bytes_per_param(
    repo_id: str, tags: list[str] | None, quantization: str
) -> float | None:
    q = quantization.lower()
    if q in ("awq", "gptq"):
        markers = [repo_id.lower(), *(tag.lower() for tag in tags or [])]
        if any(marker in text for text in markers for marker in _EIGHT_BIT_MARKERS):
            return _EIGHT_BIT_WEIGHT_BYTES_PER_PARAM
    return _WEIGHT_BYTES_PER_PARAM.get(q)


def _estimated_vram_gb(
    params_billion: float | None, *, bytes_per_param: float | None
) -> float | None:
    if params_billion is None or bytes_per_param is None:
        return None
    return _vram_gb(params_billion * 1_000_000_000 * bytes_per_param, params_billion=params_billion)


def _vram_gb(weight_bytes: float, *, params_billion: float) -> float:
    kv_bytes = _KV_BYTES_PER_TOKEN_PER_BILLION_PARAMS * params_billion * _ASSUMED_MAX_MODEL_LEN
    return (weight_bytes + kv_bytes) / (1024**3) + RUNTIME_OVERHEAD_GB


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

    most_all_time = max((c.downloads_all_time or 0 for c in candidates), default=0)
    most_recent = max((c.downloads or 0 for c in candidates), default=0)

    def score(c: ModelCandidate) -> float:
        return (
            _FIT_WEIGHT * fit_score(c)
            + _DOWNLOADS_ALL_TIME_WEIGHT * _downloads_score(c.downloads_all_time, most_all_time)
            + _DOWNLOADS_RECENT_WEIGHT * _downloads_score(c.downloads, most_recent)
        )

    return sorted(candidates, key=score, reverse=True)


def _downloads_score(downloads: int | None, most_downloads: int) -> float:
    floor, ceiling = math.log10(1 + _DOWNLOADS_FLOOR), math.log10(1 + most_downloads)
    if ceiling <= floor:
        return 0.0
    return max(0.0, (math.log10(1 + (downloads or 0)) - floor) / (ceiling - floor))
