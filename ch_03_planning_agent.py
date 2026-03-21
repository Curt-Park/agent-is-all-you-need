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

# -- Reuse from earlier chapters --------------------------------------------
# This is a key design principle: each chapter builds on the previous one.
# We import the core agent loop from ch01, all the tools + dispatch from ch02,
# and only add the new planning-specific pieces here.
from ch_01_bash_agent import Colors, _run_agent, gather_project_context
from ch_02_multi_tool_agent import DISPATCH as BASE_DISPATCH
from ch_02_multi_tool_agent import TOOLS as BASE_TOOLS
from ch_02_multi_tool_agent import execute_tool_call, tool

# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------
#
# The system prompt now includes a "Planning" section that instructs the LLM
# to break tasks into steps and track them with the todo tool.  This is the
# simplest form of "plan-then-execute" — the agent creates a checklist,
# works through it item by item, and can revise the plan if things change.

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
#
# The Planning Pattern
# ~~~~~~~~~~~~~~~~~~~~
# Without planning, an agent with many tools tends to "wing it" — jumping
# straight into actions without thinking ahead.  For complex multi-step tasks
# this leads to missed steps, wrong ordering, and wasted iterations.
#
# The fix is surprisingly simple: give the agent a todo list tool and tell it
# (in the system prompt) to plan before acting.  The agent then:
#   1. Creates a todo list with all the steps it thinks it needs.
#   2. Marks each step "in_progress" as it starts working on it.
#   3. Marks it "completed" when done.
#   4. Can add/remove/reorder items if the plan needs to change mid-task.
#
# This gives us (the humans watching) visibility into what the agent is
# thinking, and gives the agent itself a structured "scratchpad" to track
# what's done and what's left.

# Start with all tools from ch02, then add the todo tool on top.
# We .copy() so that modifying these lists doesn't affect ch02's registries.
TOOLS: list[dict] = BASE_TOOLS.copy()
DISPATCH: dict[str, callable] = BASE_DISPATCH.copy()


class TodoItem(TypedDict):
    """Schema for a single todo item.

    The LLM sends a list of these every time it calls the todo tool.
    Using TypedDict (rather than a plain dict) lets the @tool decorator
    auto-generate a precise JSON Schema — the LLM sees exactly what
    fields and values are allowed (id, text, status with its 3 options).
    """

    id: int
    text: str
    status: Literal["pending", "in_progress", "completed"]


def render(items: list[TodoItem]) -> str:
    """Render the current todo list as a human-readable string.

    This output is sent back to the LLM as the tool result, so the agent
    can "see" its own plan and decide what to do next.
    """
    if not items:
        return "TODO is empty."
    marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
    lines = []
    for item in items:
        lines.append(f"{marker[item['status']]} #{item['id']}: {item['text']}")
    done = sum(1 for t in items if t["status"] == "completed")
    lines.append(f"\n({done}/{len(items)} completed)")
    return "\n".join(lines)


# The todo tool is stateless on our side — the LLM sends the *full* list
# every time it calls the tool.  This means the LLM's conversation history
# is the source of truth for the plan, not any server-side state.  Simple,
# and it means the plan is automatically included in trajectory logs.


@tool(tools=TOOLS, dispatch=DISPATCH)
def todo(items: list[TodoItem]) -> str:
    """Update task list. Track progress on multi-step tasks.

    Args:
        items: List of todo items.
    """
    print(f"{Colors.MAGENTA}[todo] {len(items)} items{Colors.RESET}")
    return render(items)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
#
# Same _run_agent() core from ch01 — the only difference is we pass in our
# extended TOOLS list (ch02 tools + todo) and the matching DISPATCH dict.
# The agent loop itself doesn't need to know about planning; the LLM handles
# it autonomously because the system prompt tells it to use the todo tool.


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
