"""Integration tests for ch_06 context compact (requires LLM API)."""

import pytest

from conftest import get_compact_calls, get_tool_calls

from ch_06_context_compact import run_agent


@pytest.mark.integration
def test_agent_completes_simple_task(workspace):
    """The compact agent should still complete a basic task."""
    trajectory = run_agent("Create a file called hello.txt with the content 'Hello World'", max_steps=10)
    calls = get_tool_calls(trajectory)
    assert len(calls) > 0
    assert (workspace / "hello.txt").exists()


@pytest.mark.integration
def test_micro_compact_triggers_in_long_session(workspace):
    """In a multi-step task, old tool results should be compacted."""
    trajectory = run_agent(
        "Create files a.txt, b.txt, c.txt, d.txt, e.txt each with unique content, "
        "then read each file to verify its content",
        max_steps=20,
    )
    # After enough turns, old tool results should have been compacted
    tool_msgs = [m for m in trajectory["messages"] if m.get("role") == "tool"]
    compacted = [m for m in tool_msgs if "[Previous:" in m.get("content", "")]
    assert len(compacted) > 0, "Expected some old tool results to be micro-compacted"


@pytest.mark.integration
def test_compact_tool_in_trajectory(workspace):
    """When asked to compact, the agent should call the compact tool."""
    # First do some work, then ask to compact
    trajectory = run_agent(
        "Create a file called test.py with a hello world function, "
        "then call compact to free up context space, "
        "then confirm the file still exists",
        max_steps=15,
    )
    calls = get_compact_calls(trajectory)
    assert len(calls) > 0, "Expected agent to call the compact tool"
