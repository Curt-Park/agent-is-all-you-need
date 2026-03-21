"""
Chapter 04. Subagent
===========================

Extends the planning agent with subtask delegation via child agents.
The parent agent can spawn independent child agents that work with fresh
context and return only their final response.

What you'll learn:
------------------
    - Parent-child agent delegation: the "task" tool pattern.
    - Context isolation: each child gets a fresh messages[] list.
    - Information flow: only the child's final text returns to the parent.
    - Why self-contained task descriptions matter for effective delegation.

What changed from Chapter 03:
-----------------------------
    1. Added task tool for delegating subtasks to child agents.
    2. Child agents get fresh context (independent messages[]).
    3. Children have all tools except task (no recursive spawning).
    4. System prompt guides the LLM on when/how to delegate.

Usage:
------
    $ python ch_04_subagent.py "Research both Python dataclasses and Pydantic, \\
        then write a comparison summarizing pros and cons of each"
    $ python ch_04_subagent.py "Read every .py file in this project, \\
        summarize what each does, then create an INDEX.md" --max-steps 20
"""

import argparse
import contextlib
import io

# -- Reuse from earlier chapters --------------------------------------------
# Each chapter builds on the previous one.  We import the core agent loop
# from ch01, the tool decorator + dispatch from ch02, and all the tools
# (bash, read, write, edit, glob, grep, websearch, todo) from ch03.
# This chapter adds one new tool: task (for subtask delegation).
from ch_01_bash_agent import Colors, _run_agent, gather_project_context
from ch_02_multi_tool_agent import execute_tool_call, tool
from ch_03_planning_agent import DISPATCH as BASE_DISPATCH
from ch_03_planning_agent import TOOLS as BASE_TOOLS

# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------
#
# The parent system prompt adds a "Subagents" section on top of ch03's
# planning + safety + tool guidance.  This teaches the LLM *when* to
# delegate (self-contained, independent subtasks) and *how* (write clear
# descriptions because the child can't see the parent's conversation).

SYSTEM_PROMPT = """\
You are a coding agent. Solve tasks using the provided tools.

# Planning
- Use todo tool to plan for multi-step tasks.
- Mark tasks as in_progress before starting, completed when done.
- Update your todo list as the plan evolves.
- Always prefer tools over prose when responding.

# Subagents
- When a task requires applying the same operation to multiple items
  (e.g. summarize each file, research each topic, process each URL),
  you MUST delegate each item by calling the task tool — do NOT do them yourself.
- Call the task tool multiple times in a single response to run subtasks in parallel.
- Each child agent gets a fresh context — it cannot see your conversation,
  so write clear, self-contained task descriptions.

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


# The child gets the same prompt *minus* the Subagents section.  Since the
# child doesn't have the task tool, telling it about delegation would just
# waste context tokens and confuse it.  Everything else (planning, safety,
# tool guidance) still applies — a child is a fully capable agent.

CHILD_SYSTEM_PROMPT = """\
You are a coding agent. Solve tasks using the provided tools, then summarize the result.

# Planning
- Use todo tool to plan for multi-step tasks.
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
# Subagent delegation
# ---------------------------------------------------------------------------
#
# The Subagent Pattern
# ~~~~~~~~~~~~~~~~~~~~
# As tasks grow complex, a single agent's context window fills with tool
# outputs from many different subtasks — file contents, search results,
# command outputs.  This "context pollution" hurts the LLM's focus and
# increases cost (every token in the history is re-read on every turn).
#
# The fix: let the parent agent *delegate* self-contained subtasks to
# independent child agents, each with a fresh messages[] list.  This is
# the same principle as process isolation in operating systems — each
# child gets its own address space (context) so it can't be confused by
# the parent's unrelated work.
#
# The information bottleneck is intentional: only the child's final text
# response crosses the parent-child boundary.  If the parent needs
# intermediate results, it should break the task differently.  This
# constraint *forces* good decomposition.
#
# Cost implications: each child is a separate LLM conversation, so
# delegation trades context window usage for additional API calls.
# Delegate only when the subtask is genuinely independent — don't spawn
# a child just to run a single bash command.

# Start with all tools from ch03, then add the task tool on top.
# We .copy() so that modifying these lists doesn't affect ch03's registries.
TOOLS: list[dict] = BASE_TOOLS.copy()
DISPATCH: dict[str, callable] = BASE_DISPATCH.copy()


def _extract_final_response(trajectory: dict) -> str:
    """Get the last assistant text message from a trajectory.

    The child agent's trajectory contains the full conversation (system,
    user, assistant, tool messages).  We walk backwards to find the last
    assistant message with text content — that's the child's "answer"
    that gets returned to the parent.
    """
    for msg in reversed(trajectory["messages"]):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return "(Child agent produced no text response)"


# -- Child tool set ---------------------------------------------------------
#
# Preventing Recursion
# ~~~~~~~~~~~~~~~~~~~~
# If a child could call the task tool, it would spawn a grandchild, which
# could spawn a great-grandchild, and so on — an "agent fork bomb".  Even
# without infinite recursion, deep nesting makes debugging nearly impossible
# and costs explode exponentially.
#
# The simplest fix: strip the task tool from the child's tool set.  The
# child is still a fully capable agent (bash, read, write, edit, glob,
# grep, websearch, todo) — it just can't delegate further.  This gives us
# a clean, predictable depth of exactly 1 (parent → child).


def spawn_child(
    description: str,
    child_system_prompt: str,
    tools: list[dict],
    dispatch: dict[str, callable],
    max_steps: int = 15,
) -> str:
    """Spawn a child agent with fresh context and return its final response.

    This is the shared implementation behind the ``task`` tool.  Each chapter
    provides its own child system prompt and tool set; the spawning logic
    stays in one place.

    Args:
        description: Self-contained description of the subtask.
        child_system_prompt: System prompt for the child agent.
        tools: The parent's TOOLS list (task tool will be excluded for children).
        dispatch: The parent's DISPATCH dict (task will be excluded for children).
        max_steps: Maximum steps the child may take.
    """
    print(f"{Colors.CYAN}[task] Delegating: {description}{Colors.RESET}")

    # Build child tool sets at call time, so the task tool is guaranteed
    # to be in tools/dispatch and we can correctly exclude it.
    child_tools = [t for t in tools if t["function"]["name"] != "task"]
    child_dispatch = {k: v for k, v in dispatch.items() if k != "task"}

    # Spawn a child agent with fresh context.  The child gets:
    #   - Its own system prompt (no Subagents section)
    #   - Only the task description as the user message
    #   - All tools except task (no grandchildren)
    #   - Lower max_steps (subtasks should be focused)
    #   - No HITL (children run autonomously)
    #
    # We suppress the child's stdout so its step-by-step logs don't
    # clutter the parent's console.  Only the parent's "[task]" line
    # and the child's final response are visible.
    with contextlib.redirect_stdout(io.StringIO()):
        trajectory = _run_agent(
            task=description,
            system_prompt=child_system_prompt,
            tools=child_tools,
            execute_tool_call=lambda tc: execute_tool_call(tc, child_dispatch),
            max_steps=max_steps,
            enable_hitl=False,
        )

    return _extract_final_response(trajectory)


@tool(tools=TOOLS, dispatch=DISPATCH)
def task(description: str, max_steps: int = 15) -> str:
    """Spawn a subagent with fresh context. It shares the filesystem but not conversation history.

    Args:
        description: A clear, self-contained description of the subtask.
        max_steps: Maximum number of results to return. Defaults to 15.
    """
    return spawn_child(description, CHILD_SYSTEM_PROMPT, TOOLS, DISPATCH, max_steps)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
#
# Same _run_agent() core from ch01 — the only difference is we pass in our
# extended TOOLS list (ch03 tools + task) and the matching DISPATCH dict.
# The parent agent has access to the task tool for delegation; the LLM
# decides when to delegate based on the system prompt guidance.


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
