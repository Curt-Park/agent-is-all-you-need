"""Unit tests for ch_01 bash agent functions (no LLM calls)."""

import subprocess
from types import SimpleNamespace

from ch_01_bash_agent import bash, execute_tool_call, gather_project_context, git_files, run_quiet, shell

# ---------------------------------------------------------------------------
# shell()
# ---------------------------------------------------------------------------


def test_shell_captures_stdout():
    """shell() should capture command stdout."""
    result = shell("echo hello")
    assert result.stdout.strip() == "hello"
    assert result.returncode == 0


def test_shell_captures_stderr():
    """shell() should capture stderr from failing commands."""
    result = shell("ls /nonexistent_path_xyz 2>&1")
    assert result.stdout.strip() or result.stderr.strip()


def test_shell_timeout():
    """shell() should raise on timeout."""
    try:
        shell("sleep 10", timeout=1)
        raise AssertionError("Expected TimeoutExpired")
    except subprocess.TimeoutExpired:
        pass


# ---------------------------------------------------------------------------
# run_quiet()
# ---------------------------------------------------------------------------


def test_run_quiet_returns_stripped_stdout():
    """run_quiet() should return stripped stdout on success."""
    assert run_quiet("echo hello") == "hello"


def test_run_quiet_returns_empty_on_failure():
    """run_quiet() should return '' when the command fails."""
    result = run_quiet("false")
    assert result == ""


def test_run_quiet_returns_empty_on_timeout():
    """run_quiet() should return '' when the command times out."""
    result = run_quiet("sleep 10", timeout=1)
    assert result == ""


# ---------------------------------------------------------------------------
# git_files()
# ---------------------------------------------------------------------------


def test_git_files_returns_list():
    """git_files() should return a sorted list of strings in a git repo."""
    files = git_files()
    assert isinstance(files, list)
    assert len(files) > 0
    assert all(isinstance(f, str) for f in files)
    assert files == sorted(files)


# ---------------------------------------------------------------------------
# bash()
# ---------------------------------------------------------------------------


def test_bash_returns_stdout():
    """bash() should return command output."""
    output = bash("echo hello")
    assert "hello" in output


def test_bash_returns_stderr_on_empty_stdout():
    """bash() should fall back to stderr when stdout is empty."""
    output = bash("python -c 'import sys; sys.stderr.write(\"err\")'")
    assert "err" in output


def test_bash_returns_no_output():
    """bash() should return '(no output)' for silent commands."""
    output = bash("true")
    assert output == "(no output)"


def test_bash_returns_error_on_exception(monkeypatch):
    """bash() should return 'Error: ...' when shell() raises."""
    import ch_01_bash_agent

    monkeypatch.setattr(ch_01_bash_agent, "shell", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    output = bash("anything")
    assert "Error" in output
    assert "boom" in output


# ---------------------------------------------------------------------------
# execute_tool_call()
# ---------------------------------------------------------------------------


def _make_tool_call(arguments: str):
    """Helper to create a mock tool call object."""
    return SimpleNamespace(function=SimpleNamespace(arguments=arguments))


def test_execute_tool_call_valid():
    """execute_tool_call() should run the command and return output."""
    tc = _make_tool_call('{"command": "echo ok"}')
    output = execute_tool_call(tc)
    assert "ok" in output


def test_execute_tool_call_bad_json():
    """execute_tool_call() should return error for malformed JSON."""
    tc = _make_tool_call("not json")
    output = execute_tool_call(tc)
    assert "Error parsing tool call" in output


def test_execute_tool_call_missing_key():
    """execute_tool_call() should return error when 'command' key is missing."""
    tc = _make_tool_call('{"cmd": "echo hi"}')
    output = execute_tool_call(tc)
    assert "Error parsing tool call" in output


# ---------------------------------------------------------------------------
# gather_project_context()
# ---------------------------------------------------------------------------


def test_gather_project_context_contains_cwd():
    """gather_project_context() should include the working directory."""
    ctx = gather_project_context()
    assert "Working directory" in ctx


def test_gather_project_context_contains_platform():
    """gather_project_context() should include platform info."""
    ctx = gather_project_context()
    assert "Platform" in ctx
