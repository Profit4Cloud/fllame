from fllame.domain.vllm_command import join_command_lines


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
