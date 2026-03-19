import os
import shutil
from pathlib import Path

import pytest

from ch_02_multi_tool_agent import (
    _DISPATCH,
    edit_file,
    glob_search,
    grep_search,
    perform_websearch,
    read_file,
    run_agent,
    run_bash_command,
    write_file,
)


@pytest.fixture
def workspace():
    ws = Path(f"test_workspace_{os.getpid()}").resolve()
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir()

    original_cwd = os.getcwd()
    os.chdir(ws)

    yield ws

    os.chdir(original_cwd)
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


def test_bash_echo():
    result = run_bash_command("echo hello")
    assert result.strip() == "hello"


def test_bash_stderr():
    result = run_bash_command("ls /nonexistent_path_xyz")
    assert "No such file" in result or "cannot access" in result


def test_bash_timeout():
    result = run_bash_command("sleep 60")
    assert "Error" in result


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file(workspace):
    (workspace / "hello.txt").write_text("line1\nline2\nline3\n")
    result = read_file("hello.txt")
    assert "line1" in result
    assert "line2" in result
    assert "line3" in result


def test_read_file_line_numbers(workspace):
    (workspace / "nums.txt").write_text("aaa\nbbb\n")
    result = read_file("nums.txt")
    assert "1" in result
    assert "2" in result


def test_read_file_with_offset_and_limit(workspace):
    (workspace / "nums.txt").write_text("a\nb\nc\nd\ne\n")
    result = read_file("nums.txt", offset=2, limit=2)
    assert "b" in result
    assert "c" in result
    assert "a" not in result
    assert "d" not in result


def test_read_file_empty(workspace):
    (workspace / "empty.txt").write_text("")
    result = read_file("empty.txt")
    assert "empty file" in result.lower()


def test_read_file_not_found(workspace):
    result = read_file("nonexistent.txt")
    assert "Error" in result


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def test_write_file(workspace):
    result = write_file("out.txt", "hello world")
    assert "Wrote" in result
    assert (workspace / "out.txt").read_text() == "hello world"


def test_write_file_creates_dirs(workspace):
    write_file("sub/dir/deep.txt", "nested")
    assert (workspace / "sub" / "dir" / "deep.txt").read_text() == "nested"


def test_write_file_overwrites(workspace):
    (workspace / "existing.txt").write_text("old content")
    write_file("existing.txt", "new content")
    assert (workspace / "existing.txt").read_text() == "new content"


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


def test_edit_file(workspace):
    (workspace / "code.py").write_text("x = 1\ny = 2\n")
    result = edit_file("code.py", "x = 1", "x = 42")
    assert "Edited" in result
    assert (workspace / "code.py").read_text() == "x = 42\ny = 2\n"


def test_edit_file_multiline(workspace):
    (workspace / "ml.txt").write_text("aaa\nbbb\nccc\n")
    result = edit_file("ml.txt", "aaa\nbbb", "xxx\nyyy")
    assert "Edited" in result
    assert (workspace / "ml.txt").read_text() == "xxx\nyyy\nccc\n"


def test_edit_file_not_found_string(workspace):
    (workspace / "code.py").write_text("hello")
    result = edit_file("code.py", "missing string", "replacement")
    assert "not found" in result


def test_edit_file_ambiguous(workspace):
    (workspace / "dup.py").write_text("x\nx\n")
    result = edit_file("dup.py", "x", "y")
    assert "2 times" in result


def test_edit_file_missing_file(workspace):
    result = edit_file("nope.txt", "a", "b")
    assert "Error" in result


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


def test_glob_search(workspace):
    (workspace / "a.py").write_text("")
    (workspace / "b.txt").write_text("")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "c.py").write_text("")
    result = glob_search("**/*.py")
    assert "a.py" in result
    assert "c.py" in result
    assert "b.txt" not in result


def test_glob_no_matches(workspace):
    result = glob_search("**/*.rs")
    assert "No matches" in result


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


def test_grep_search(workspace):
    (workspace / "sample.py").write_text("import json\nimport os\nprint('hello')\n")
    result = grep_search("import", path=".")
    assert "sample.py" in result
    assert "import json" in result


def test_grep_with_include_filter(workspace):
    (workspace / "a.py").write_text("match here\n")
    (workspace / "b.txt").write_text("match here\n")
    result = grep_search("match", path=".", include="*.py")
    assert "a.py" in result
    assert "b.txt" not in result


def test_grep_single_file(workspace):
    (workspace / "target.py").write_text("foo\nbar\nbaz\n")
    result = grep_search("bar", path="target.py")
    assert "target.py:2" in result
    assert "bar" in result


def test_grep_no_matches(workspace):
    (workspace / "empty_match.py").write_text("nothing relevant\n")
    result = grep_search("zzzzz", path=".")
    assert "No matches" in result


def test_grep_invalid_regex(workspace):
    result = grep_search("[invalid", path=".")
    assert "No matches" in result or "Error" in result


# ---------------------------------------------------------------------------
# websearch
# ---------------------------------------------------------------------------


def test_websearch_returns_results():
    result = perform_websearch("Python programming language")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def test_tool_dispatch_has_all_tools():
    expected = {"bash", "read_file", "write_file", "edit_file", "glob", "grep", "websearch"}
    assert set(_DISPATCH.keys()) == expected


# ---------------------------------------------------------------------------
# Integration tests (hit real LLM)
# ---------------------------------------------------------------------------


def test_agent_read_and_summarize(workspace):
    (workspace / "data.txt").write_text("The answer is 42.\n")
    run_agent(
        "Read data.txt and write a file called summary.txt containing only the number from data.txt",
        max_steps=3,
        enable_hitl=False,
    )
    assert (workspace / "summary.txt").exists()
    assert "42" in (workspace / "summary.txt").read_text()


def test_agent_edit_task(workspace):
    (workspace / "config.txt").write_text("mode = debug\nport = 8080\n")
    run_agent(
        "Edit config.txt: change 'mode = debug' to 'mode = production'",
        max_steps=3,
        enable_hitl=False,
    )
    content = (workspace / "config.txt").read_text()
    assert "mode = production" in content
    assert "port = 8080" in content
