from ch_05_skills_agent import run_agent
from tests.agent.conftest import get_skill_calls, get_task_calls


def test_load_skill_tool_is_used(workspace):
    """Agent should call load_skill when asked to do a specialized task."""
    trajectory = run_agent(
        "Load the code_review skill, then review this code and write "
        "your findings to review.txt:\n\n"
        "def add(a, b):\n    return a + b\n",
        max_steps=10,
        enable_hitl=False,
    )

    skill_calls = get_skill_calls(trajectory)
    assert len(skill_calls) >= 1, "Expected at least 1 load_skill tool call"
    assert any(c["args"]["name"] == "code_review" for c in skill_calls), "Expected code_review skill to be loaded"
    assert (workspace / "review.txt").exists(), "review.txt should exist"


def test_child_agent_uses_skill(workspace):
    """Delegated subtask should be able to load and use skills."""
    trajectory = run_agent(
        "Delegate to a subagent: have it load the doc_writer skill and write documentation for a Calculator class to docs.md",
        max_steps=15,
        enable_hitl=False,
    )

    task_calls = get_task_calls(trajectory)
    assert len(task_calls) >= 1, "Expected at least 1 task delegation"
    assert (workspace / "docs.md").exists(), "docs.md should exist"
