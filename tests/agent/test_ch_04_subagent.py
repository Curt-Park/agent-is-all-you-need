from ch_04_subagent import run_agent
from tests.agent.conftest import get_task_calls


def test_task_tool_is_used_for_delegation(workspace):
    """Basic delegation: parent delegates a file-write to a child agent."""
    trajectory = run_agent(
        "Delegate to a subagent: have it write 'hello from child' to child_output.txt",
        max_steps=10,
        enable_hitl=False,
    )

    task_calls = get_task_calls(trajectory)
    assert len(task_calls) >= 1, "Expected at least 1 task tool call"
    assert (workspace / "child_output.txt").exists(), "child_output.txt should exist"
    assert "hello from child" in (workspace / "child_output.txt").read_text()


def test_child_context_isolation(workspace):
    """Parent creates a file, then delegates an independent subtask to a child.

    The child should succeed despite having no knowledge of the parent's
    earlier file creation — it gets fresh context with only its own task.
    """
    trajectory = run_agent(
        "First, write 'parent data' to parent.txt yourself. "
        "Then delegate to a subagent: have it write 'child data' to child.txt.",
        max_steps=10,
        enable_hitl=False,
    )

    task_calls = get_task_calls(trajectory)
    assert len(task_calls) >= 1, "Expected at least 1 task delegation"
    assert (workspace / "parent.txt").exists(), "parent.txt should exist"
    assert "parent data" in (workspace / "parent.txt").read_text()
    assert (workspace / "child.txt").exists(), "child.txt should exist"
    assert "child data" in (workspace / "child.txt").read_text()


def test_multiple_delegations(workspace):
    """Parent delegates multiple independent subtasks sequentially."""
    trajectory = run_agent(
        "Delegate each of these to a separate subagent: "
        "1) write 'one' to first.txt, "
        "2) write 'two' to second.txt, "
        "3) write 'three' to third.txt.",
        max_steps=20,
        enable_hitl=False,
    )

    task_calls = get_task_calls(trajectory)
    assert len(task_calls) >= 2, "Expected at least 2 task delegations"

    assert (workspace / "first.txt").exists(), "first.txt should exist"
    assert (workspace / "first.txt").read_text().strip() == "one"
    assert (workspace / "second.txt").exists(), "second.txt should exist"
    assert (workspace / "second.txt").read_text().strip() == "two"
    assert (workspace / "third.txt").exists(), "third.txt should exist"
    assert (workspace / "third.txt").read_text().strip() == "three"
