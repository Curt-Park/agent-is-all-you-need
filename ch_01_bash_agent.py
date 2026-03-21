"""
Chapter 01. Bash Agent
======================

A simple autonomous coding agent that interacts with a bash shell.

What you'll learn:
------------------
    - The core agent loop: LLM decides, tool executes, result feeds back.
    - OpenAI-compatible function calling with a single tool.
    - Human-in-the-loop (HITL) feedback.
    - Trajectory logging for later analysis.

Mechanism:
----------
    1. Agent receives a task.
    2. Agent decides: Need a tool?
       - Yes -> Execute bash, return output to agent, loop.
       - No  -> Task complete or seek human feedback.
    3. Loop until task done or max steps reached.

Usage:
------
    $ python ch_01_bash_agent.py "List current files"
    $ python ch_01_bash_agent.py "Create a README.md" --max-steps 5 --hitl
"""

import argparse
import json
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCallUnion

load_dotenv()


# ---------------------------------------------------------------------------
# Helper functions and dataclasses
# ---------------------------------------------------------------------------
#
# Small utilities used throughout the agent.  Nothing AI-specific here — just
# terminal colors, a thin subprocess wrapper, and git helpers.


@dataclass
class Colors:
    """ANSI escape codes for colored terminal output.

    Used purely for readability when the agent prints its actions.
    """

    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"


def shell(command: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a shell command and return the CompletedProcess result.

    This is the lowest-level execution primitive — every tool that touches the
    OS goes through here.  `capture_output=True` grabs both stdout and stderr
    so the agent can read the command's output.
    """
    return subprocess.run(command, shell=True, text=True, capture_output=True, timeout=timeout)


def run_quiet(cmd: str) -> str:
    """Run a shell command, return stdout stripped, or empty string on failure.

    A convenience wrapper for context-gathering calls where we don't want a
    failure (e.g. "not a git repo") to crash the startup.
    """
    try:
        return shell(cmd, timeout=5).stdout.strip()
    except Exception:
        return ""


def git_files() -> list[str]:
    """List all git-tracked + untracked-not-ignored files, or [] on failure.

    Combines two git commands to get a complete picture:
      - `git ls-files` — all tracked files.
      - `git ls-files --others --exclude-standard` — new untracked files
        (but not those in .gitignore for security and compact information).
    """
    try:

        def s(cmd):
            return run_quiet(cmd, timeout=5).stdout.strip()

        raw = s("git ls-files") + "\n" + s("git ls-files --others --exclude-standard")
        return sorted(set(filter(None, raw.splitlines())))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------
#
# The system prompt is the agent's "personality + knowledge".  We dynamically
# inject the current working directory, file tree, and git status so the LLM
# starts every session already knowing what project it's working in — no
# "run ls first" step needed.


def gather_project_context() -> str:
    """Build a project-aware system prompt from the current environment.

    Collects:
      - Working directory and OS platform.
      - File tree (git-aware when possible, simple ls fallback otherwise).
      - Git branch, status, and recent commits.
    """
    cwd = os.getcwd()

    # File tree — prefer git-aware listing, fall back to top-level ls.
    files = git_files() or sorted(p.name for p in Path(cwd).iterdir() if not p.name.startswith("."))

    # Git context with graceful fallback (returns "" for non-git dirs)
    branch = run_quiet("git branch --show-current")
    status = run_quiet("git status --short") or "(clean)"
    commits = run_quiet("git log --oneline -5")
    git_info = f"\n## Git\nBranch: {branch}\nStatus: {status}\nRecent commits:\n{commits}" if branch else ""

    return f"""\
# Environment
- Working directory: {cwd}
- Platform: {platform.system()} {platform.release()}

## File tree
{"\n".join(files) if files else "(empty)"}
{git_info}"""


# The system prompt is built once at import time.  It combines safety rules
# with the live project context gathered above.  Every message to the LLM
# starts with this — it's the first thing the model "reads".

SYSTEM_PROMPT = """\
You are a coding agent. You solve tasks by running bash commands.

# Safety
- Never run destructive commands (rm -rf, git push --force, git reset --hard)
without explicit user confirmation.
- Avoid commands that could expose secrets (e.g. printing .env files).

""" + gather_project_context()


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------
#
# OpenAI-compatible function calling in a nutshell:
#   1. You describe your tools as JSON schemas and pass them with every request.
#   2. The LLM reads the schemas and can decide to "call" a tool by returning a
#      structured JSON object with the tool name + arguments.
#   3. Your code executes the tool and feeds the output back to the LLM.
#
# Below is the simplest possible schema — a single "bash" tool with one
# required string parameter ("command").  Chapter 02 auto-generates these
# schemas with a decorator so you never hand-write JSON again.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------
#
# The handler that actually runs when the LLM calls the "bash" tool.
# It takes the command string, executes it, and returns the output as plain
# text that gets fed back into the conversation.


def bash(command: str) -> str:
    """Executes a shell command and returns stdout (or stderr if stdout is empty)."""
    print(f"{Colors.CYAN}$ {command}{Colors.RESET}")
    try:
        res = shell(command)
        return res.stdout or res.stderr or "(no output)"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
#
# This is the bridge between the LLM's structured output and our Python code.
# The LLM returns a tool_call object with a function name and JSON arguments;
# we parse that JSON, look up the right handler, and call it.


def execute_tool_call(tool_call: ChatCompletionMessageToolCallUnion) -> str:
    """Parse the LLM's tool call JSON and run the requested command.

    The LLM sometimes produces malformed JSON (especially smaller models),
    so we catch parse errors and return them as tool output — this lets the
    LLM see its mistake and retry, instead of crashing the whole loop.
    """
    try:
        return bash(json.loads(tool_call.function.arguments)["command"])
    except (json.JSONDecodeError, KeyError) as e:
        return f"Error parsing tool call: {e}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
#
# This is the heart of the agent — a simple while-loop that alternates
# between "ask the LLM what to do" and "execute what it asked for".
#
# The pattern:
#   ┌──────────────────────────────────────────┐
#   │  User task                               │
#   │       ↓                                  │
#   │  ┌──► LLM decides ──► tool call? ──Yes──►│──► execute tool ──┐
#   │  │                       │               │                   │
#   │  │                       No              │                   │
#   │  │                       ↓               │                   │
#   │  │                   text response       │                   │
#   │  │                       ↓               │                   │
#   │  │                   done (or HITL)      │                   │
#   │  │                                       │                   │
#   │  └───────────────────────────────────────┘◄──────────────────┘
#   └──────────────────────────────────────────┘
#
# _run_agent() is the reusable core; run_agent() is the chapter-specific
# wrapper that plugs in the right system prompt, tools, and dispatch logic.
# Later chapters reuse _run_agent with different tools.


def _run_agent(
    task: str,
    system_prompt: str,
    tools: list[dict],
    execute_tool_call: Callable[[ChatCompletionMessageToolCallUnion], str],
    max_steps: int = 30,
    enable_hitl: bool = False,
) -> dict:
    """Core agent loop: orchestrates the LLM turns and tool execution.

    Each iteration is one "turn":
      1. Send the conversation to the LLM (system prompt + history so far).
      2. If the LLM wants to call a tool  -> execute it, append result, next turn.
      3. If the LLM responds with text    -> task is done (or ask for human feedback).
    """
    # -- Setup: create an OpenAI client from environment variables ----------
    # These env vars let you point the agent at any OpenAI-compatible API
    # (OpenAI, Anthropic via proxy, local vLLM, etc.).
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
    model = os.getenv("LLM_MODEL_ID")

    # -- Conversation history = the agent's memory --------------------------
    # The conversation starts with the system prompt and the user's task.
    # Every LLM response and tool result gets appended here — this growing
    # list IS the agent's memory for this session.  The LLM is stateless;
    # it re-reads the entire history on every turn.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    # -- Main loop: one iteration = one LLM turn ---------------------------
    for step in range(max_steps):
        print(f"\n--- Step {step + 1}/{max_steps} ---")

        # Ask the LLM: "given this conversation so far, what do you want to do?"
        # We pass the tool schemas so the model knows what tools are available.
        response = client.chat.completions.create(model=model, messages=messages, tools=tools).choices[0].message

        # Append the LLM's response to the conversation history.
        # model_dump() converts the Pydantic response object to a plain dict.
        messages.append(response.model_dump(exclude_none=True))

        # Print any text the LLM produced (thinking out loud, final answer, etc.)
        if response.content:
            print(f"{Colors.YELLOW}Agent:{Colors.RESET} {response.content}")

        # -- Branch A: LLM wants to use tool(s) ----------------------------
        # The LLM can request multiple tool calls in one turn (parallel calls).
        if response.tool_calls:
            for tool_call in response.tool_calls:
                output = execute_tool_call(tool_call)
                print(output)
                # Feed the tool output back so the LLM can see what happened.
                # The tool_call_id links this result to the specific call.
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
            continue  # Go to next step — let the LLM decide what to do next.

        # -- Branch B: LLM responded with text (no tool calls) -------------
        # This usually means the agent thinks the task is done.
        # In HITL (human-in-the-loop) mode, give the human a chance to
        # provide feedback or corrections before we stop.
        if enable_hitl:
            feedback = input(f"{Colors.MAGENTA}[HITL] Provide feedback (or press Enter to finish): {Colors.RESET}")
            if feedback.strip():
                messages.append({"role": "user", "content": feedback})
                continue  # Feed the feedback back and let the agent continue.

        print("Task marked complete.")
        break

    # -- Trajectory logging -------------------------------------------------
    # Save the full conversation as a "trajectory" — the complete record of
    # what the agent did (every LLM response, tool call, and tool output).
    # This is invaluable for debugging and later chapters use trajectories
    # for evaluation and reinforcement learning.
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


# -- Public wrapper ---------------------------------------------------------
# Each chapter provides its own run_agent() that plugs chapter-specific tools
# and system prompt into the shared _run_agent() core.


def run_agent(task: str, max_steps: int = 30, enable_hitl: bool = False) -> list[dict]:
    return _run_agent(
        task,
        system_prompt=SYSTEM_PROMPT,
        tools=TOOLS,
        execute_tool_call=execute_tool_call,
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
