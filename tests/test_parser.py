import pytest

from fllame.recipes.parser import RecipePasteError, parse_pasted_recipe


def test_parses_env_and_command():
    text = """
    export FOO=bar
    export BAZ="quoted value"
    vllm serve org/repo --max-model-len 8192 --gpu-memory-utilization=0.9
    """

    parsed = parse_pasted_recipe(text)

    assert parsed.repo_id == "org/repo"
    assert parsed.env == {"FOO": "bar", "BAZ": "quoted value"}
    assert parsed.command == "vllm serve org/repo --max-model-len 8192 --gpu-memory-utilization=0.9"


def test_no_export_lines_is_fine():
    parsed = parse_pasted_recipe("vllm serve org/repo --port 8000")
    assert parsed.env == {}


def test_blank_lines_and_comments_ignored():
    text = """
    # a comment
    export FOO=bar

    vllm serve org/repo
    """
    parsed = parse_pasted_recipe(text)
    assert parsed.repo_id == "org/repo"
    assert parsed.env == {"FOO": "bar"}


def test_missing_vllm_serve_line_raises():
    with pytest.raises(RecipePasteError, match="vllm serve"):
        parse_pasted_recipe("export FOO=bar")


def test_missing_repo_id_raises():
    with pytest.raises(RecipePasteError, match="repo id"):
        parse_pasted_recipe("vllm serve")


def test_multiple_vllm_serve_lines_raises():
    text = "vllm serve org/a\nvllm serve org/b"
    with pytest.raises(RecipePasteError, match="exactly one"):
        parse_pasted_recipe(text)


def test_unrecognized_line_raises():
    text = "docker run vllm/vllm-openai:latest\nvllm serve org/repo"
    with pytest.raises(RecipePasteError, match="unrecognized line"):
        parse_pasted_recipe(text)


def test_rejects_shell_metacharacters_in_export_value():
    with pytest.raises(RecipePasteError, match="won't evaluate"):
        parse_pasted_recipe("export FOO=$(cat /etc/passwd)\nvllm serve org/repo")


def test_rejects_command_chaining_in_serve_line():
    text = "vllm serve org/repo; rm -rf /"
    with pytest.raises(RecipePasteError, match="won't evaluate"):
        parse_pasted_recipe(text)


def test_rejects_multi_token_env_value_without_quotes():
    with pytest.raises(RecipePasteError, match="single token"):
        parse_pasted_recipe("export FOO=bar baz\nvllm serve org/repo")


def test_rejects_malformed_flag():
    with pytest.raises(RecipePasteError, match="malformed flag"):
        parse_pasted_recipe("vllm serve org/repo -x")


def test_bare_value_tokens_allowed_after_a_flag():
    parsed = parse_pasted_recipe("vllm serve org/repo --port 8000")
    assert parsed.command == "vllm serve org/repo --port 8000"


def test_multiline_backslash_continued_command_joins_into_one_line():
    import shlex

    text = (
        "vllm serve Inferact/Qwen3.8-27B-NVFP4 \\\n"
        "  --tensor-parallel-size 1 \\\n"
        "  --enable-auto-tool-choice \\\n"
        "  --tool-call-parser qwen3_coder\n"
    )

    parsed = parse_pasted_recipe(text)

    assert parsed.repo_id == "Inferact/Qwen3.8-27B-NVFP4"
    assert "\\" not in parsed.command
    assert shlex.split(parsed.command) == [
        "vllm",
        "serve",
        "Inferact/Qwen3.8-27B-NVFP4",
        "--tensor-parallel-size",
        "1",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
    ]


def test_parses_run_lines_as_preinstall():
    text = """
    RUN uv pip install -U "transformers>=5.8.0"
    vllm serve org/repo
    """
    parsed = parse_pasted_recipe(text)
    assert parsed.preinstall == ['uv pip install -U "transformers>=5.8.0"']


def test_no_run_lines_is_fine():
    parsed = parse_pasted_recipe("vllm serve org/repo")
    assert parsed.preinstall == []


def test_multiple_run_lines_preserve_order():
    text = "RUN pip install foo\nRUN pip install bar\nvllm serve org/repo"
    parsed = parse_pasted_recipe(text)
    assert parsed.preinstall == ["pip install foo", "pip install bar"]


def test_run_line_shell_metacharacters_allowed():
    """Unlike an export value or a vllm serve flag, a RUN line is genuinely
    meant to be a shell command - chaining two installs with && is normal,
    not a smuggled command where a plain token was expected."""
    text = "RUN pip install foo && pip install bar\nvllm serve org/repo"
    parsed = parse_pasted_recipe(text)
    assert parsed.preinstall == ["pip install foo && pip install bar"]
