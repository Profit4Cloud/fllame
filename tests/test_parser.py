import shlex

import pytest

from fllame.recipes.parser import RecipePasteError, parse_command_args, parse_env_line


def test_unquoted_args_join_into_one_command():
    command = parse_command_args(["vllm", "serve", "org/repo", "--port", "8000"])
    assert command == "vllm serve org/repo --port 8000"


def test_unquoted_arg_with_spaces_stays_one_token():
    command = parse_command_args(["vllm", "serve", "org/repo", "--chat-template", "a b"])
    assert shlex.split(command)[-1] == "a b"


def test_quoted_single_arg_is_taken_as_is():
    command = parse_command_args(["vllm serve org/repo --port 8000"])
    assert command == "vllm serve org/repo --port 8000"


def test_quoted_multi_line_with_backslashes():
    text = (
        "vllm serve Inferact/Qwen3.8-27B-NVFP4 \\\n"
        "  --tensor-parallel-size 1 \\\n"
        "  --tool-call-parser qwen3_coder\n"
    )

    command = parse_command_args([text])

    assert shlex.split(command) == [
        "vllm",
        "serve",
        "Inferact/Qwen3.8-27B-NVFP4",
        "--tensor-parallel-size",
        "1",
        "--tool-call-parser",
        "qwen3_coder",
    ]


def test_quoted_multi_line_without_backslashes():
    text = "vllm serve org/repo\n  --tensor-parallel-size 1\n  --enable-auto-tool-choice"
    command = parse_command_args([text])
    assert command == "vllm serve org/repo --tensor-parallel-size 1 --enable-auto-tool-choice"


def test_quoted_value_inside_quoted_command_is_kept():
    command = parse_command_args(["vllm serve org/repo --chat-template 'a b'"])
    assert shlex.split(command)[-1] == "a b"


def test_rejects_image_before_repo_id():
    with pytest.raises(RecipePasteError, match="without arguments"):
        parse_command_args(["vllm/vllm-openai:v0.28.0", "org/repo", "--port", "8000"])


def test_rejects_export_and_run_lines():
    with pytest.raises(RecipePasteError, match="without arguments"):
        parse_command_args(["export FOO=bar\nvllm serve org/repo"])
    with pytest.raises(RecipePasteError, match="without arguments"):
        parse_command_args(["RUN pip install foo\nvllm serve org/repo"])


def test_rejects_missing_repo_id():
    with pytest.raises(RecipePasteError, match="repo id"):
        parse_command_args(["vllm", "serve"])


def test_rejects_command_chaining():
    with pytest.raises(RecipePasteError, match="won't evaluate"):
        parse_command_args(["vllm serve org/repo; rm -rf /"])


def test_rejects_malformed_flag():
    with pytest.raises(RecipePasteError, match="malformed flag"):
        parse_command_args(["vllm", "serve", "org/repo", "-x"])


def test_parse_env_line_bare_key_value():
    assert parse_env_line("FOO=bar") == ("FOO", "bar")


def test_parse_env_line_quoted_value_with_spaces():
    assert parse_env_line('BAZ="quoted value"') == ("BAZ", "quoted value")


def test_parse_env_line_rejects_export_prefix():
    """The dialogue's env step collects bare KEY=VALUE lines - unlike
    the mixed paste grammar, `export ` isn't part of this shape."""
    with pytest.raises(RecipePasteError):
        parse_env_line("export FOO=bar")


def test_parse_env_line_rejects_multi_token_value():
    with pytest.raises(RecipePasteError, match="single token"):
        parse_env_line("FOO=bar baz")


def test_parse_env_line_rejects_shell_metacharacters():
    with pytest.raises(RecipePasteError, match="won't evaluate"):
        parse_env_line("FOO=$(cat /etc/passwd)")


def test_parse_env_line_rejects_non_kv_line():
    with pytest.raises(RecipePasteError):
        parse_env_line("not an env line")
