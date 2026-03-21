from conftest import get_todo_calls

from ch_03_planning_agent import run_agent


def test_planning_creates_pending_then_completes_all(workspace):
    trajectory = run_agent(
        "Write 'hello' to hello.txt, then write 'world' to world.txt. Use the todo tool to plan first.",
        max_steps=10,
        enable_hitl=False,
    )

    todo_calls = get_todo_calls(trajectory)
    assert len(todo_calls) >= 2, "Expected at least 2 todo calls (initial plan + final update)"

    # First todo call: all items should be pending
    first_items = todo_calls[0]["items"]
    assert len(first_items) > 0
    assert all(item["status"] in {"pending", "in_progress"} for item in first_items)

    # Last todo call: all items should be completed
    last_items = todo_calls[-1]["items"]
    assert all(item["status"] == "completed" for item in last_items)
