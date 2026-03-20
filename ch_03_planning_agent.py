"""
Chapter 03. Planning Agent
===========================

Extends the multi-tool agent with explicit planning via todo tools.
The agent tracks its own progress and can update todos during task execution.

What you'll learn:
------------------
    - Todo tools: add, update, remove, and list tasks for structured planning.
    - Planning workflow: create a plan, mark tasks in_progress, complete when done.
    - Replanning pattern: when a tool execution suggests the plan needs
      adjustment, the agent can revise its todo list.

What changed from Chapter 02:
-----------------------------
    1. Added global TODO dict for persistent task tracking.
    2. Added planning tools: todo_add, todo_update, todo_remove, todo_list.
    3. System prompt guides the LLM to plan before acting.

Usage:
------
    $ python ch_03_planning_agent.py "Build a REST API with auth and tests"
    $ python ch_03_planning_agent.py "Refactor all bash agents into a class" --max-steps 10
"""

import argparse

# reuse
from ch_01_bash_agent import Colors, _run_agent, gather_project_context
from ch_02_multi_tool_agent import execute_tool_call, tool, TOOLS as BASE_TOOLS, DISPATCH as BASE_DISPATCH


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a coding agent. Solve tasks using the provided tools.

# Safety
- Never run destructive commands (rm -rf, git push --force, git reset --hard)
without explicit user confirmation.
- Avoid commands that could expose secrets (e.g. printing .env files).

# Tool usage
- Prefer specialized tools (read, write, edit, glob, grep)
over bash for file operations. They are safer, produce structured output,
and avoid common shell pitfalls.
- Use bash only for commands that have no dedicated tool
(e.g. running tests, installing packages, git commands).
- Use websearch for information not available in the project.

# Planning
- For multi-step tasks, create a todo list first with todo_add.
- Mark tasks as in_progress before starting, completed when done.
- Update your todo list as the plan evolves.
- Always prefer tools over prose when responding.

""" + gather_project_context()


# ---------------------------------------------------------------------------
# tool implementations — using @tool with custom registry
# ---------------------------------------------------------------------------

TOOLS: list[dict] = BASE_TOOLS.copy()  # OpenAI function-calling schemas for planning
DISPATCH: dict[str, callable] = BASE_DISPATCH.copy()  # name -> handler(**kwargs)

VALID_STATUSES = {"pending", "in_progress", "completed"}
TODO: dict[int, tuple[str, str]] = {}


@tool(tools=TOOLS, dispatch=DISPATCH)
def todo_add(id: int, text: str, status: str = "pending") -> str:
    """Add a new task to the todo list.

    Args:
        id: Unique identifier for the task.
        text: Description of the task to add.
        status: Initial status (pending, in_progress, completed). Defaults to pending.
    """
    print(f"{Colors.MAGENTA}[todo_add] {id}: {text}{Colors.RESET}")
    if id in TODO:
        return f"Error: {id} already exists in TODO"
    if status not in VALID_STATUSES:
        return f"Error: status must be one of {VALID_STATUSES}"
    TODO[id] = (text, status)
    return f"Item {id}: {text} added"


@tool(tools=TOOLS, dispatch=DISPATCH)
def todo_update(id: int, status: str) -> str:
    """Update a task's status in the todo list.

    Args:
        id: The identifier of the task to update.
        status: The new status (pending, in_progress, completed).
    """
    print(f"{Colors.MAGENTA}[todo_update] #{id} -> {status}{Colors.RESET}")
    if id not in TODO:
        return f"Error: {id} doesn't exist in TODO"
    if status not in VALID_STATUSES:
        return f"Error: status must be one of {VALID_STATUSES}"
    text, _ = TODO[id]
    TODO[id] = text, status
    return f"Item {id}'s status updated: {status}"


@tool(tools=TOOLS, dispatch=DISPATCH)
def todo_remove(id: int) -> str:
    """Remove a task from the todo list.

    Args:
        id: The identifier of the task to remove.
    """
    print(f"{Colors.MAGENTA}[todo_remove] #{id}{Colors.RESET}")
    if id not in TODO:
        return f"Error: {id} doesn't exist in TODO"
    TODO.pop(id)
    return f"Item {id} removed from TODO"


@tool(tools=TOOLS, dispatch=DISPATCH)
def todo_list() -> str:
    """Show the current todo list with all tasks and their statuses."""
    print(f"{Colors.MAGENTA}[todo_list]{Colors.RESET}")
    if not TODO:
        return "TODO is empty."
    lines = ""
    for id, (text, status) in sorted(TODO.items()):
        if status == "pending":
            lines += "[ ] "
        elif status == "in_progress":
            lines += "[>] "
        elif status == "completed":
            lines += "[x] "
        else:
            return f"Error: invalid status {status} of item {id}: {text}"
        lines += text + "\n"
    return lines


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent(task: str, max_steps: int = 30, enable_hitl: bool = False) -> list[dict]:
    return _run_agent(
        task,
        system_prompt=SYSTEM_PROMPT,
        tools=TOOLS,
        execute_tool_call=lambda tool_call: execute_tool_call(tool_call, DISPATCH),
        max_steps=max_steps,
        enable_hitl=enable_hitl,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Task to perform")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--hitl", action="store_true", help="Enable human-in-the-loop")
    args = parser.parse_args()

    run_agent(args.task, args.max_steps, args.hitl)


if __name__ == "__main__":
    main()
