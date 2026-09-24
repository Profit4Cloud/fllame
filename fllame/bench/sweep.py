"""A `vllm bench serve` concurrency sweep: the per-level commands, and
the results table built from their saved result JSON."""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_CONCURRENCY = (1, 4, 8, 16, 32)
DEFAULT_INPUT_LEN = 10000
DEFAULT_OUTPUT_LEN = 1000

# Every level needs at least this many requests for its averages to mean anything.
_MIN_PROMPTS = 10
# Requests are always a whole number of waves: a trailing partial wave would run
# below the level's concurrency and skew its numbers.
_MIN_WAVES = 2


@dataclass(frozen=True)
class Level:
    concurrency: int
    num_prompts: int


@dataclass(frozen=True)
class Column:
    header: str
    help: str
    key: str | None = None
    inverted: bool = False


COLUMNS = (
    Column("CONC", "Maximum number of requests in flight at once."),
    Column("PROMPTS", "Number of requests sent at this concurrency level."),
    Column("FAILED", "Requests that errored instead of completing."),
    Column(
        "S/REQ",
        "Seconds per completed request, over the whole level.",
        "request_throughput",
        inverted=True,
    ),
    Column(
        "OUT TOK/S", "Generated tokens per second, summed over all requests.", "output_throughput"
    ),
    Column(
        "TOT TOK/S",
        "Prompt plus generated tokens per second, over all requests.",
        "total_token_throughput",
    ),
    Column("TTFT MS", "Mean time until a request's first token arrives.", "mean_ttft_ms"),
    Column("TPOT MS", "Mean time per generated token, excluding the first.", "mean_tpot_ms"),
)


class SweepError(ValueError):
    pass


def default_num_prompts(concurrency: int) -> int:
    return concurrency * max(_MIN_WAVES, math.ceil(_MIN_PROMPTS / concurrency))


def parse_int_list(flag: str, value: str) -> list[int]:
    try:
        numbers = [int(part) for part in value.split(",")]
    except ValueError as e:
        raise SweepError(f"{flag} must be comma-separated whole numbers, got {value!r}.") from e
    if any(n <= 0 for n in numbers):
        raise SweepError(f"{flag} values must be positive, got {value!r}.")
    return numbers


def build_levels(concurrency: list[int], num_prompts: list[int] | None) -> list[Level]:
    if num_prompts is None:
        return [Level(c, default_num_prompts(c)) for c in concurrency]
    if len(num_prompts) != len(concurrency):
        raise SweepError(
            f"--num-prompts has {len(num_prompts)} value(s) but --concurrency has "
            f"{len(concurrency)} - give one per concurrency level."
        )
    for c, n in zip(concurrency, num_prompts, strict=True):
        if n % c != 0:
            raise SweepError(f"--num-prompts {n} is not a multiple of concurrency {c}.")
    return [Level(c, n) for c, n in zip(concurrency, num_prompts, strict=True)]


def bench_command(
    *,
    model: str,
    base_url: str,
    level: Level,
    input_len: int,
    output_len: int,
    result_dir: str,
    result_filename: str,
) -> list[str]:
    command = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--base-url", base_url,
        "--model", model,
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--max-concurrency", str(level.concurrency),
        "--num-prompts", str(level.num_prompts),
        "--save-result",
        "--result-dir", result_dir,
        "--result-filename", result_filename,
    ]  # fmt: skip
    return command


def _format_metric(value: object) -> str:
    if not isinstance(value, int | float):
        return "-"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}" if value < 1000 else f"{value:.0f}"


def result_cells(level: Level, result: dict) -> list[str]:
    completed = result.get("completed")
    failed = result.get("failed")
    if not isinstance(failed, int) and isinstance(completed, int):
        failed = level.num_prompts - completed
    cells = [
        str(level.concurrency),
        str(level.num_prompts),
        str(failed) if isinstance(failed, int) else "-",
    ]
    return cells + [_format_metric(_metric(result, column)) for column in COLUMNS[3:]]


def _metric(result: dict, column: Column) -> object:
    value = result.get(column.key)
    if column.inverted and isinstance(value, int | float):
        return 1 / value if value > 0 else None
    return value


def column_widths() -> list[int]:
    """Fixed up front rather than fitted to the data, since rows are
    printed one by one as each level finishes."""
    return [max(len(column.header), 7) for column in COLUMNS]


def format_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.rjust(width) for cell, width in zip(cells, widths, strict=True))


def header_row(widths: list[int]) -> str:
    return format_row([column.header for column in COLUMNS], widths)


def columns_help() -> str:
    width = max(len(column.header) for column in COLUMNS)
    return "\n".join(f"{column.header.ljust(width)}  {column.help}" for column in COLUMNS)
