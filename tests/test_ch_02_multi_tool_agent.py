import json
import os
import shutil
from pathlib import Path

import pytest

from ch_02_multi_tool_agent import run_agent


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


def get_tool_calls(trajectory):
    """Helper to extract all tool calls made during a trajectory."""
    calls = []
    for msg in trajectory["messages"]:
        if isinstance(msg, dict) and "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                calls.append({"name": name, "args": args})
    return calls


def test_run_agent_read_file(workspace):
    (workspace / "test.txt").write_text("Hello World!")
    trajectory = run_agent("Read the file test.txt", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(c["name"] == "read_file" and c["args"].get("path") == "test.txt" for c in calls)
    # Check if agent read content (agent often outputs content in next turn's message)
    assert "Hello World!" in str(trajectory["messages"])


def test_run_agent_write_file(workspace):
    trajectory = run_agent("Create a file named hello.txt with content 'hello_world'", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(
        c["name"] == "write_file" and c["args"].get("path") == "hello.txt" and c["args"].get("content") == "hello_world"
        for c in calls
    )
    assert (workspace / "hello.txt").read_text() == "hello_world"


def test_run_agent_edit_file(workspace):
    (workspace / "file.txt").write_text("original content")
    trajectory = run_agent("Edit file.txt: replace 'original' with 'new'", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(c["name"] == "edit_file" and c["args"].get("path") == "file.txt" for c in calls)
    assert "new content" in (workspace / "file.txt").read_text()


def test_run_agent_glob(workspace):
    (workspace / "a.py").write_text("")
    trajectory = run_agent("List files matching pattern *.py", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(c["name"] == "glob" and "*.py" in c["args"].get("pattern", "") for c in calls)
    # Also verify outcome if possible
    assert "a.py" in str(trajectory["messages"])


def test_run_agent_grep(workspace):
    (workspace / "code.py").write_text("def find_me():\n    pass")
    trajectory = run_agent("Find the file containing 'def find_me'", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(c["name"] == "grep" and "def find_me" in c["args"].get("pattern", "") for c in calls)


def test_run_agent_websearch(workspace):
    # Just verify the tool is triggered for a search task
    trajectory = run_agent("Search the web for 'current date'", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(c["name"] == "websearch" for c in calls)


def test_run_agent_bash(workspace):
    trajectory = run_agent("Use bash to write 'bash_output' to file named bash.txt", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(c["name"] == "bash" for c in calls)
    assert (workspace / "bash.txt").read_text().strip() == "bash_output"
