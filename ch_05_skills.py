"""
Chapter 05. Skills
========================

Extends the subagent system with on-demand domain expertise via SKILL.md files.
Instead of bloating the system prompt with every possible guideline, skills use
a two-layer injection pattern:

    Layer 1 (system prompt): skill names + one-line descriptions (~1 line each).
    Layer 2 (tool_result):   full skill body, loaded only when the agent asks.

What you'll learn:
------------------
    - Two-layer knowledge injection: cheap discovery + on-demand loading.
    - YAML frontmatter for skill metadata (name, description).
    - tool_result as context: SKILL.md content enters the conversation as a
      tool output — the agent loop stays unchanged.
    - Composable expertise: adding a new skill = creating one Markdown file.
    - Safety basics: validating user-controlled input (skill names).

What changed from Chapter 04:
-----------------------------
    1. Added skills/ directory with SKILL.md files for domain expertise.
    2. Added _discover_skills() + get_skill_descriptions() / load_skill_body()
       as plain functions (no class needed).
    3. Added load_skill tool — the agent calls it to inject a skill's full
       body into the conversation as a tool_result.
    4. Skill descriptions are embedded in both parent and child system prompts,
       so the LLM always knows what skills exist without a separate tool call.
    5. Task tool re-registered so children get the updated tools and prompts.

Usage:
------
    $ python ch_05_skills.py "Review ch_04_subagent.py for code quality"
    $ python ch_05_skills.py "Write documentation for the skill loader"
    $ python ch_05_skills.py "Generate tests for the todo tool in ch_03"
"""

import argparse
import re
from pathlib import Path

# -- Reuse from earlier chapters --------------------------------------------
# Each chapter builds on the previous one.  We import the core agent loop
# from ch01, the tool decorator + dispatch from ch02, and all tools from ch04
# (bash, read, write, edit, glob, grep, websearch, todo, task).
# This chapter adds one new tool: load_skill.
from ch_01_bash_agent import Colors, _run_agent, gather_project_context
from ch_02_multi_tool_agent import execute_tool_call, tool
from ch_04_subagent import DISPATCH as BASE_DISPATCH
from ch_04_subagent import TOOLS as BASE_TOOLS
from ch_04_subagent import _spawn_child

# ---------------------------------------------------------------------------
# Skill loading — two-layer knowledge injection
# ---------------------------------------------------------------------------
#
# The Two-Layer Pattern
# ~~~~~~~~~~~~~~~~~~~~~
# Large language models have limited context windows and every token counts.
# Stuffing every possible guideline into the system prompt wastes tokens on
# skills the agent may never need for a given task.
#
# The fix is a two-layer approach borrowed from learn-claude-code s05:
#
#   Layer 1 — CHEAP:  Embed skill *names + one-line descriptions* in the
#             system prompt.  This costs ~1 line per skill and lets the LLM
#             know what's available without reading the full body.
#
#   Layer 2 — ON DEMAND:  When the LLM decides it needs a skill, it calls
#             the load_skill tool.  The full SKILL.md body is returned as
#             a tool_result — it enters the conversation history like any
#             other tool output, so the agent loop stays untouched.
#
# This is "load knowledge when you need it, not upfront."
#
# Skill files use YAML frontmatter for metadata:
#
#   ---
#   name: code_review
#   description: Perform thorough code reviews with structured checklist.
#   ---
#
#   # Code Review Skill
#   (full body here)
#
# The frontmatter is parsed at startup; the body is served on demand.

SKILLS_DIR = Path(__file__).parent / "skills"


# ---------------------------------------------------------------------------
# Skill discovery — scan once at import time
# ---------------------------------------------------------------------------
#
# _SKILLS is a flat dict: { "code_review": {"description": "...", "body": "..."}, ... }
# Built once; consumed by get_skill_descriptions() and load_skill().


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from the Markdown body.

    Frontmatter is delimited by ``---`` on its own line::

        ---
        key: value
        ---
        Body starts here.

    Returns ``(metadata_dict, body_string)``.  If no frontmatter is found,
    returns ``({}, full_text)``.
    """
    match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).strip().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            meta[key.strip()] = val.strip()
    return meta, match.group(2).strip()


def _discover_skills(skills_dir: Path = SKILLS_DIR) -> dict[str, dict]:
    """Scan *skills_dir* for ``*/SKILL.md`` files and return a skills dict."""
    skills: dict[str, dict] = {}
    if not skills_dir.is_dir():
        return skills
    for skill_file in sorted(skills_dir.rglob("SKILL.md")):
        text = skill_file.read_text()
        meta, body = _parse_frontmatter(text)
        name = meta.get("name", skill_file.parent.name)
        skills[name] = {
            "description": meta.get("description", "No description"),
            "body": body,
        }
    return skills


_SKILLS = _discover_skills()


# -- Layer 1: system prompt descriptions ------------------------------------


def get_skill_descriptions(skills: dict[str, dict] = _SKILLS) -> str:
    """Return a compact one-liner per skill for the system prompt."""
    if not skills:
        return "(no skills available)"
    return "\n".join(f"  - {name}: {s['description']}" for name, s in skills.items())


# -- Layer 2: on-demand full body -------------------------------------------


def load_skill_body(name: str, skills: dict[str, dict] = _SKILLS) -> str:
    """Return the full skill body wrapped in ``<skill>`` tags.

    Returns an error string (not an exception) if the name is invalid or
    unknown — the agent loop treats it like any other tool output.
    """
    if ".." in name or "/" in name or "\\" in name:
        return f"Error: Invalid skill name {name!r} (path traversal not allowed). Available: {', '.join(skills)}"
    skill = skills.get(name)
    if not skill:
        return f"Error: Unknown skill {name!r}. Available: {', '.join(skills)}"
    return f'<skill name="{name}">\n{skill["body"]}\n</skill>'


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------
#
# Both system prompts embed Layer 1 via get_skill_descriptions().
# This way the LLM always knows what skills exist — no discovery tool needed.
#
# The parent prompt adds a "Skills" section on top of ch04's planning +
# subagents + safety + tool guidance.  The child prompt gets the same Skills
# section but omits the Subagents section (children can't delegate).

SYSTEM_PROMPT = (
    """\
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

# Skills
- Before starting a specialized task (code review, writing docs, generating
  tests), call load_skill to load the relevant guidelines.
- Skills provide domain expertise as context — read them carefully and follow
  their instructions.
- You can load multiple skills if a task spans domains.
- Available skills:
"""
    + get_skill_descriptions()
    + """

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

"""
    + gather_project_context()
)


CHILD_SYSTEM_PROMPT = (
    """\
You are a coding agent. Solve tasks using the provided tools, then summarize the result.

# Planning
- Use todo tool to plan for multi-step tasks.
- Mark tasks as in_progress before starting, completed when done.
- Update your todo list as the plan evolves.
- Always prefer tools over prose when responding.

# Skills
- Before starting a specialized task (code review, writing docs, generating
  tests), call load_skill to load the relevant guidelines.
- Skills provide domain expertise as context — read them carefully and follow
  their instructions.
- You can load multiple skills if a task spans domains.
- Available skills:
"""
    + get_skill_descriptions()
    + """

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

"""
    + gather_project_context()
)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------
#
# Start with all tools from ch04, then add load_skill on top.
# We .copy() so that modifying these lists doesn't affect ch04's registries.
# The task tool is re-registered below so children use this chapter's
# CHILD_SYSTEM_PROMPT (with Skills) and extended tool set.

TOOLS: list[dict] = BASE_TOOLS.copy()
DISPATCH: dict[str, callable] = BASE_DISPATCH.copy()


@tool(tools=TOOLS, dispatch=DISPATCH)
def load_skill(name: str) -> str:
    """Load domain expertise by name. The skill content will guide your approach.

    Args:
        name: Name of the skill to load (see system prompt for available skills).
    """
    print(f"{Colors.GREEN}[skill] Loading: {name}{Colors.RESET}")
    return load_skill_body(name)


# ---------------------------------------------------------------------------
# Subagent delegation — re-register with updated child config
# ---------------------------------------------------------------------------
#
# Why re-register?
# ~~~~~~~~~~~~~~~~
# The task tool inherited from ch04 spawns children with ch04's system prompt
# and tool set — neither includes load_skill or the Skills guidance.  We need
# children to be able to load skills independently (e.g. a child delegated
# to write docs should be able to load doc_writer on its own).
#
# The cleanest fix: remove ch04's task tool and re-register it here with
# this chapter's CHILD_SYSTEM_PROMPT and DISPATCH (which now includes
# load_skill).  make_task_tool() handles the spawning logic — we just
# inject this chapter's child config.

TOOLS[:] = [t for t in TOOLS if t["function"]["name"] != "task"]
del DISPATCH["task"]


@tool(tools=TOOLS, dispatch=DISPATCH)
def task(description: str, max_steps: int = 15) -> str:
    """Spawn a subagent with fresh context. It shares the filesystem but not conversation history.

    Args:
        description: A clear, self-contained description of the subtask.
        max_steps: Maximum number of results to return. Defaults to 15.
    """
    return _spawn_child(description, CHILD_SYSTEM_PROMPT, TOOLS, DISPATCH, max_steps)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent(task: str, max_steps: int = 30, enable_hitl: bool = False) -> dict:
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
