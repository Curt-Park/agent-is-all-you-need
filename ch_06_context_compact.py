"""
Chapter 06. Context Compact
===========================

Extends the skills agent with 3-layer context compaction for long sessions.
As conversations grow, the context window fills with stale tool outputs and
old exchanges.  This chapter adds automatic and manual compaction so the
agent can work indefinitely without hitting token limits.

What you'll learn:
------------------
    - Token estimation: cheap heuristics vs exact counting.
    - Micro-compaction: trimming stale tool outputs every turn.
    - Auto-compaction: LLM-based summarization at a token threshold.
    - Manual compaction: a tool the agent calls to free context space.
    - Transcript preservation: saving originals before summarization.

What changed from Chapter 05:
-----------------------------
    1. New agent loop _run_agent_compact() with compaction hooks.
    2. micro_compact() trims old tool_result messages each turn.
    3. auto_compact() summarizes the conversation when tokens exceed threshold.
    4. Added compact tool for agent-initiated compaction.
    5. Transcripts saved to .transcripts/ before any summarization.

Usage:
------
    $ python ch_06_context_compact.py "Refactor all Python files in this project"
    $ python ch_06_context_compact.py "Build a complex feature" --max-steps 50
"""

import argparse
import json
import os
import platform
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# -- Reuse from earlier chapters --------------------------------------------
from ch_01_bash_agent import Colors, gather_project_context
from ch_02_multi_tool_agent import execute_tool_call, tool
from ch_05_skills import DISPATCH as BASE_DISPATCH
from ch_05_skills import TOOLS as BASE_TOOLS
from ch_05_skills import _spawn_child, get_skill_descriptions

load_dotenv()


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
#
# Why estimate instead of count exactly?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Exact token counting requires a tokenizer (tiktoken for OpenAI models,
# different libraries for other providers).  This adds a dependency and
# couples us to a specific model family.
#
# For compaction decisions, we only need to answer "are we getting close to
# the limit?" — not "exactly how many tokens do we have?"  A rough estimate
# of ~4 characters per token is accurate enough for this purpose and works
# with any model.


def estimate_tokens(messages: list[dict]) -> int:
    """Estimate total tokens in the conversation using ~4 chars/token heuristic.

    Counts both message content AND tool_calls arguments, since assistant
    messages carry most of their cost in tool_calls (not content).
    """
    total_chars = 0
    for m in messages:
        total_chars += len(str(m.get("content", "") or ""))
        if m.get("tool_calls"):
            total_chars += len(json.dumps(m["tool_calls"]))
    return total_chars // 4


# ---------------------------------------------------------------------------
# Layer 1: Micro-compaction — trim stale tool outputs every turn
# ---------------------------------------------------------------------------
#
# The Problem
# ~~~~~~~~~~~
# Tool outputs (file contents, command output, search results) are often the
# largest messages in the conversation.  After the agent has processed a tool
# result and moved on, the raw output is dead weight — the LLM re-reads it
# on every turn but gains nothing new from it.
#
# The Fix
# ~~~~~~~
# Replace tool_result content from old turns with a one-line placeholder:
# "[Previous: used {tool_name}]".  This preserves the *fact* that the tool
# was called (which the LLM uses for planning) while removing the bulk.
#
# "Old" means more than MICRO_COMPACT_AGE turns ago.  This is conservative
# enough that the agent can still reference recent tool outputs.

MICRO_COMPACT_AGE = 3


def micro_compact(messages: list[dict]) -> None:
    """Replace stale tool_result messages with compact placeholders (in-place).

    A "turn" is defined by each assistant message with tool_calls.  Tool
    results from turns more than MICRO_COMPACT_AGE before the *last* turn
    in the message list are compacted.

    Staleness is computed purely from the message list (not an external step
    counter).  This ensures correct behavior even after auto_compact resets
    the messages list — the turn count restarts from the remaining messages.
    """
    # First pass: map each tool_call_id to its turn number and tool name.
    turn = -1
    call_info: dict[str, tuple[int, str]] = {}  # tool_call_id -> (turn, tool_name)
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            turn += 1
            for tc in msg["tool_calls"]:
                tc_id = tc["id"] if isinstance(tc, dict) else tc.id
                name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
                call_info[tc_id] = (turn, name)

    total_turns = turn  # Index of the last turn (-1 if no turns)

    # Second pass: compact old tool results.
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        tc_id = msg.get("tool_call_id", "")
        info = call_info.get(tc_id)
        if info is None:
            continue
        msg_turn, tool_name = info
        if total_turns - msg_turn >= MICRO_COMPACT_AGE:
            msg["content"] = f"[Previous: used {tool_name}]"


# ---------------------------------------------------------------------------
# Transcript saving — preserve originals before summarization
# ---------------------------------------------------------------------------

TRANSCRIPTS_DIR = Path(".transcripts")


def save_transcript(
    messages: list[dict],
    transcripts_dir: Path = TRANSCRIPTS_DIR,
) -> Path:
    """Save the current conversation to a timestamped JSON file.

    Called before auto_compact destroys the original messages.  The saved
    transcript is the complete, uncompressed conversation — useful for
    debugging, evaluation, and the trajectory analysis in later chapters.
    """
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    filename = f"transcript_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    path = transcripts_dir / filename
    path.write_text(json.dumps(messages, indent=2, ensure_ascii=False))
    print(f"{Colors.BLUE}[transcript] Saved: {path}{Colors.RESET}")
    return path


# ---------------------------------------------------------------------------
# Layer 2: Auto-compaction — LLM-based summarization at token threshold
# ---------------------------------------------------------------------------
#
# The Problem
# ~~~~~~~~~~~
# Micro-compaction handles individual tool results, but after many turns the
# conversation itself grows large — assistant reasoning, user follow-ups,
# planning text all accumulate.  Eventually even micro-compacted conversations
# hit the context limit.
#
# The Fix
# ~~~~~~~
# When estimated tokens exceed TOKEN_THRESHOLD, ask the LLM to summarize
# the entire conversation into a structured digest.  The original is saved
# to .transcripts/ first (we never destroy data silently).  Then the
# conversation is replaced with: [system_message, summary_message].

TOKEN_THRESHOLD = 50_000

SUMMARY_PROMPT = """\
Summarize this conversation for an AI agent that will continue the work.
Include:
1. The original task
2. Key decisions made
3. Files created or modified (with paths)
4. Current status and what remains to be done

Be concise but preserve all information needed to continue the task."""


def auto_compact(
    messages: list[dict],
    client: OpenAI,
    model: str,
    transcripts_dir: Path = TRANSCRIPTS_DIR,
) -> list[dict]:
    """Summarize the full conversation via LLM and return a compacted messages list.

    Steps:
      1. Save the original transcript.
      2. Ask the LLM to summarize the conversation.
      3. Return [system_message, summary_as_user_message].
    """
    # Preserve the original before we destroy it.
    save_transcript(messages, transcripts_dir)

    # Build a summarization request — we pass the full conversation as context
    # and ask for a structured summary.
    summary_messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": json.dumps(messages[1:], indent=2, ensure_ascii=False)},
    ]

    print(f"{Colors.YELLOW}[auto_compact] Summarizing conversation...{Colors.RESET}")
    response = client.chat.completions.create(model=model, messages=summary_messages)
    summary = response.choices[0].message.content

    # Rebuild: keep system prompt, replace everything else with the summary.
    return [
        messages[0],  # system message preserved
        {"role": "user", "content": f"[Context Summary]\n{summary}"},
    ]


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------
#
# Start with all tools from ch05, then add the compact tool on top.
# We .copy() so that modifying these lists doesn't affect ch05's registries.

TOOLS: list[dict] = BASE_TOOLS.copy()
DISPATCH: dict[str, callable] = BASE_DISPATCH.copy()


# ---------------------------------------------------------------------------
# Layer 3: Manual compaction — agent-invoked tool
# ---------------------------------------------------------------------------
#
# Sometimes the agent knows it's about to do something context-heavy (e.g.
# reading several large files).  The compact tool lets it proactively free
# up space rather than waiting for auto_compact to trigger.
#
# Implementation challenge: tool functions only receive their declared args,
# but compact needs access to the conversation state (messages, client, model).
# We solve this with a module-level state dict that _run_agent_compact()
# populates before each LLM call.

_COMPACT_STATE: dict = {}


@tool(tools=TOOLS, dispatch=DISPATCH)
def compact() -> str:
    """Trigger context compaction to free up space for complex operations."""
    if not _COMPACT_STATE:
        return "Error: compact can only be called during an agent run."

    messages = _COMPACT_STATE["messages"]
    client = _COMPACT_STATE["client"]
    model = _COMPACT_STATE["model"]

    before = estimate_tokens(messages)
    new_messages = auto_compact(messages, client, model)

    # Replace the messages list in-place so the loop sees the change.
    messages.clear()
    messages.extend(new_messages)

    after = estimate_tokens(messages)
    return f"Compacted: ~{before:,} → ~{after:,} tokens."


# ---------------------------------------------------------------------------
# Agent loop with compaction hooks
# ---------------------------------------------------------------------------
#
# Why a new loop instead of modifying ch_01's _run_agent()?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The "each chapter extends, never modifies earlier chapters" principle means
# we can't add hooks to _run_agent().  Compaction requires injection *inside*
# the loop body (after tool results, before each LLM call), which can't be
# done by wrapping the function.
#
# So we copy the loop structure and add two injection points.  The code
# duplication is small (~30 lines of loop logic) and well-documented.
#
#   ┌──────────────────────────────────────────────────────────┐
#   │  User task                                               │
#   │       ↓                                                  │
#   │  ┌──► [micro_compact] ──► [auto_compact?] ──► LLM call  │
#   │  │                                              │        │
#   │  │                                         tool call?    │
#   │  │                                          Yes / No     │
#   │  │                                           │     │     │
#   │  │    execute tool ◄─────────────────────────┘     │     │
#   │  │         │                                       │     │
#   │  │    append result                            text resp │
#   │  │         │                                       │     │
#   │  └─────────┘                                     done    │
#   └──────────────────────────────────────────────────────────┘


def _run_agent_compact(
    task: str,
    system_prompt: str,
    tools: list[dict],
    execute_tool_call: Callable,
    max_steps: int = 30,
    enable_hitl: bool = False,
    token_threshold: int = TOKEN_THRESHOLD,
) -> dict:
    """Agent loop with context compaction — extends ch_01's pattern.

    Structurally identical to _run_agent() from ch_01 but adds:
      1. micro_compact() after tool results are appended.
      2. auto_compact() before each LLM call if tokens exceed threshold.
    """
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
    model = os.getenv("LLM_MODEL_ID")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    # Expose state for the compact tool.
    _COMPACT_STATE.update({"messages": messages, "client": client, "model": model})

    for step in range(max_steps):
        print(f"\n--- Step {step + 1}/{max_steps} ---")

        # -- Compaction hook 1: trim stale tool outputs --------------------
        micro_compact(messages)

        # -- Compaction hook 2: full summarization if over threshold -------
        token_est = estimate_tokens(messages)
        if token_est > token_threshold:
            print(f"{Colors.YELLOW}[auto_compact] {token_est:,} tokens > {token_threshold:,} threshold{Colors.RESET}")
            messages[:] = auto_compact(messages, client, model)
            _COMPACT_STATE["messages"] = messages

        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools
        ).choices[0].message

        messages.append(response.model_dump(exclude_none=True))

        if response.content:
            print(f"{Colors.YELLOW}Agent:{Colors.RESET} {response.content}")

        if response.tool_calls:
            for tool_call in response.tool_calls:
                output = execute_tool_call(tool_call)
                print(output)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
            continue

        if enable_hitl:
            feedback = input(f"{Colors.MAGENTA}[HITL] Provide feedback (or press Enter to finish): {Colors.RESET}")
            if feedback.strip():
                messages.append({"role": "user", "content": feedback})
                continue

        print("Task marked complete.")
        break

    # Clean up compact state.
    _COMPACT_STATE.clear()

    # Trajectory logging (same as ch_01).
    trajectory = {
        "task": task,
        "model": model,
        "max_steps": max_steps,
        "steps_used": step + 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "messages": messages,
    }
    Path("logs").mkdir(exist_ok=True)
    log_file = Path("logs") / f"trajectory_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    log_file.write_text(json.dumps(trajectory, indent=2))
    print(f"\nTrajectory saved: {log_file}")

    return trajectory


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------

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

# Context Management
- Old tool outputs are automatically trimmed to save context space.  You may
  see "[Previous: used {tool_name}]" for stale outputs — this is normal.
- If a full context summary occurs, you'll see a "[Context Summary]" message
  with the key points from the conversation so far.
- Call compact to manually free context space before complex operations.

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

# Context Management
- Old tool outputs are automatically trimmed to save context space.
- Call compact to manually free context space before complex operations.

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
# Subagent delegation — re-register with updated child config
# ---------------------------------------------------------------------------
#
# Why children don't get compaction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# _spawn_child() (from ch_04) internally calls _run_agent() — the non-compact
# loop from ch_01.  This is intentional: children have fresh contexts and
# short lifetimes (max_steps=15), so they're unlikely to hit token limits.
# Adding compaction to children would require either modifying ch_04 (violating
# the "never modify earlier chapters" rule) or duplicating _spawn_child here.
# Neither is worth it for the negligible benefit.

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
    return _run_agent_compact(
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
