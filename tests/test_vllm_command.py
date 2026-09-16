from fllame.domain.vllm_command import extract_flag_value, extract_port, join_command_lines


def test_join_command_lines_strips_trailing_backslash():
    lines = ["vllm serve org/demo \\", "--tensor-parallel-size 1 \\", "--enable-auto-tool-choice"]

    assert (
        join_command_lines(lines)
        == "vllm serve org/demo --tensor-parallel-size 1 --enable-auto-tool-choice"
    )


def test_join_command_lines_tolerates_missing_trailing_backslash():
    lines = ["vllm serve org/demo", "--tensor-parallel-size 1", "--enable-auto-tool-choice"]

    assert (
        join_command_lines(lines)
        == "vllm serve org/demo --tensor-parallel-size 1 --enable-auto-tool-choice"
    )


def test_join_command_lines_strips_leading_and_trailing_whitespace():
    lines = ["  vllm serve org/demo \\  ", "\t--tensor-parallel-size 1\t"]

    assert join_command_lines(lines) == "vllm serve org/demo --tensor-parallel-size 1"


def test_join_command_lines_drops_blank_and_comment_lines():
    lines = ["vllm serve org/demo \\", "", "# a comment", "--tensor-parallel-size 1"]

    assert join_command_lines(lines) == "vllm serve org/demo --tensor-parallel-size 1"


def test_join_command_lines_empty_list():
    assert join_command_lines([]) == ""


def test_extract_flag_value_space_form():
    assert extract_flag_value(["--max-model-len", "8192"], "--max-model-len") == "8192"


def test_extract_flag_value_equals_form():
    assert extract_flag_value(["--max-model-len=8192"], "--max-model-len") == "8192"


def test_extract_flag_value_none_when_absent():
    assert extract_flag_value(["--port", "9000"], "--max-model-len") is None


def test_extract_flag_value_none_when_flag_is_last_token_with_no_value():
    assert extract_flag_value(["--max-model-len"], "--max-model-len") is None


def test_extract_port_space_form():
    assert extract_port(["--port", "9000"]) == 9000


def test_extract_port_equals_form():
    assert extract_port(["--port=9000"]) == 9000


def test_extract_port_defaults_when_absent():
    assert extract_port([]) == 8000
    assert extract_port(["--other-flag"], default=1234) == 1234
