"""
Chapter 03. Planning Agent
===========================

Extends the multi-tool agent with explicit planning via todo tools.
The agent tracks its own progress and can update todos during task execution.

What you'll learn:
------------------
    - Single todo tool: update all tasks via a unified interface.
    - Planning workflow: create a plan, mark tasks in_progress, complete when done.
    - Replanning pattern: when a tool execution suggests the plan needs
      adjustment, the agent can revise its todo list.

What changed from Chapter 02:
-----------------------------
    1. Added global TODO list for persistent task tracking.
    2. Added single todo tool for add/update/remove operations.
    3. System prompt guides the LLM to plan before acting.

Usage:
------
    $ python ch_03_planning_agent.py "Plan Jeju Island travel for 3 days, step-by-step"
    $ python ch_03_planning_agent.py "Refactor all bash agents into a class" --max-steps 10
"""

import argparse
from typing import Literal, TypedDict

# reuse
from ch_01_bash_agent import Colors, _run_agent, gather_project_context
from ch_02_multi_tool_agent import DISPATCH as BASE_DISPATCH
from ch_02_multi_tool_agent import TOOLS as BASE_TOOLS
from ch_02_multi_tool_agent import execute_tool_call, tool

# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a coding agent. Solve tasks using the provided tools.

# Planning
- Use todo tool to plan For multi-step tasks.
- Mark tasks as in_progress before starting, completed when done.
- Update your todo list as the plan evolves.
- Always prefer tools over prose when responding.

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

""" + gather_project_context()


# ---------------------------------------------------------------------------
# Todo management
# ---------------------------------------------------------------------------

TOOLS: list[dict] = BASE_TOOLS.copy()  # OpenAI function-calling schemas for planning
DISPATCH: dict[str, callable] = BASE_DISPATCH.copy()  # name -> handler(**kwargs)

MARKER = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}


class TodoItem(TypedDict):
    id: int
    text: str
    status: Literal["pending", "in_progress", "completed"]


TODO: list[TodoItem] = []


def render() -> str:
    """Render the current todo list as a formatted string."""
    if not TODO:
        return "TODO is empty."
    lines = []
    for item in TODO:
        lines.append(f"{MARKER[item['status']]} #{item['id']}: {item['text']}")
    done = sum(1 for t in TODO if t["status"] == "completed")
    lines.append(f"\n({done}/{len(TODO)} completed)")
    return "\n".join(lines)


@tool(tools=TOOLS, dispatch=DISPATCH)
def todo(items: list[TodoItem]) -> str:
    """Update task list. Track progress on multi-step tasks.

    Args:
        items: List of todo items.
    """
    print(f"{Colors.MAGENTA}[todo] updating {len(items)} items{Colors.RESET}")
    TODO[:] = items
    return render()


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
