import json

from ch_02_multi_tool_agent import SYSTEM_PROMPT, TOOLS, execute_tool_call, run_agent


def _run_agent(task: str, **kwargs) -> list[dict]:
    return run_agent(task=task, system_prompt=SYSTEM_PROMPT, tools=TOOLS, execute_tool_call=execute_tool_call, **kwargs)


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
    trajectory = _run_agent("Read the file test.txt", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(c["name"] == "read" and "test.txt" in c["args"].get("path") for c in calls)
    # Check if agent read content (agent often outputs content in next turn's message)
    assert "Hello World!" in str(trajectory["messages"])


def test_run_agent_write_file(workspace):
    trajectory = _run_agent("Create a file named hello.txt with content 'hello_world'", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(
        c["name"] == "write" and "hello.txt" in c["args"].get("path") and c["args"].get("content") == "hello_world"
        for c in calls
    )
    assert (workspace / "hello.txt").read_text().strip() == "hello_world"


def test_run_agent_edit_file(workspace):
    (workspace / "file.txt").write_text("original content")
    trajectory = _run_agent("Edit file.txt: replace 'original' with 'new'", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(c["name"] == "edit" and "file.txt" in c["args"].get("path") for c in calls)
    assert "new content" in (workspace / "file.txt").read_text()


def test_run_agent_glob(workspace):
    (workspace / "a.py").write_text("")
    trajectory = _run_agent("List files matching pattern *.py", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(c["name"] == "glob" and "*.py" in c["args"].get("pattern", "") for c in calls)
    # Also verify outcome if possible
    assert "a.py" in str(trajectory["messages"])


def test_run_agent_grep(workspace):
    (workspace / "code.py").write_text("def find_me():\n    pass")
    trajectory = _run_agent("Find the file containing 'def find_me'", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(c["name"] == "grep" and "def find_me" in c["args"].get("pattern", "") for c in calls)


def test_run_agent_websearch(workspace):
    # Just verify the tool is triggered for a search task
    trajectory = _run_agent("Search the web for 'current date'", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(c["name"] == "websearch" for c in calls)


def test_run_agent_bash(workspace):
    trajectory = _run_agent("Use bash to write 'bash_output' to file named bash.txt", max_steps=3, enable_hitl=False)

    calls = get_tool_calls(trajectory)
    assert any(c["name"] == "bash" for c in calls)
    assert (workspace / "bash.txt").read_text().strip() == "bash_output"
