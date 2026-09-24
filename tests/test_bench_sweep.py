import pytest

from fllame.bench.sweep import (
    COLUMNS,
    Level,
    SweepError,
    bench_command,
    build_levels,
    column_widths,
    default_num_prompts,
    format_row,
    header_row,
    parse_int_list,
    result_cells,
)


def test_default_num_prompts_is_whole_waves_of_at_least_ten_requests():
    assert [default_num_prompts(c) for c in (1, 3, 4, 8, 16, 32)] == [10, 12, 12, 16, 32, 64]


def test_parse_int_list():
    assert parse_int_list("--concurrency", "1,4, 8") == [1, 4, 8]


@pytest.mark.parametrize("value", ["1,x", "0,4", "-1", ""])
def test_parse_int_list_rejects_bad_values(value):
    with pytest.raises(SweepError):
        parse_int_list("--concurrency", value)


def test_build_levels_defaults_num_prompts():
    assert build_levels([1, 8], None) == [Level(1, 10), Level(8, 16)]


def test_build_levels_uses_given_num_prompts():
    assert build_levels([1, 8], [5, 24]) == [Level(1, 5), Level(8, 24)]


def test_build_levels_rejects_length_mismatch():
    with pytest.raises(SweepError, match="one per concurrency level"):
        build_levels([1, 8], [10])


def test_build_levels_rejects_partial_wave():
    with pytest.raises(SweepError, match="not a multiple"):
        build_levels([8], [20])


def _command(tokenizer=None):
    return bench_command(
        model="org/demo",
        tokenizer=tokenizer,
        base_url="http://localhost:8000",
        level=Level(4, 16),
        input_len=100,
        output_len=10,
        result_dir="/tmp/x",
        result_filename="c4.json",
    )


def _flag(command, name):
    return command[command.index(name) + 1]


def test_bench_command():
    command = _command()
    assert command[:3] == ["vllm", "bench", "serve"]
    assert _flag(command, "--model") == "org/demo"
    assert _flag(command, "--dataset-name") == "random"
    assert _flag(command, "--max-concurrency") == "4"
    assert _flag(command, "--num-prompts") == "16"
    assert _flag(command, "--random-input-len") == "100"
    assert _flag(command, "--random-output-len") == "10"
    assert _flag(command, "--result-dir") == "/tmp/x"
    assert _flag(command, "--result-filename") == "c4.json"
    assert "--save-result" in command
    assert "--tokenizer" not in command


def test_bench_command_with_tokenizer():
    assert _flag(_command(tokenizer="org/repo"), "--tokenizer") == "org/repo"


def test_result_cells():
    result = {
        "completed": 15,
        "request_throughput": 0.51234,
        "output_throughput": 1234.56,
        "total_token_throughput": 12000.4,
        "mean_ttft_ms": 250.0,
        "p99_ttft_ms": 900.25,
        "mean_tpot_ms": 20.04,
        "mean_itl_ms": 19.96,
    }
    assert result_cells(Level(4, 16), result) == [
        "4", "16", "1", "0.51", "1235", "12000", "250.0", "900.2", "20.0", "20.0",
    ]  # fmt: skip


def test_result_cells_prefers_reported_failed_count_and_dashes_missing_metrics():
    cells = result_cells(Level(1, 10), {"completed": 10, "failed": 2})
    assert cells[:3] == ["1", "10", "2"]
    assert set(cells[3:]) == {"-"}


def test_rows_line_up_with_header():
    widths = column_widths()
    row = format_row(result_cells(Level(32, 64), {}), widths)
    assert len(row) == len(header_row(widths))
    assert len(widths) == len(COLUMNS)
