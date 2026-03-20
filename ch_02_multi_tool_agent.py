"""
Chapter 02. Multi-Tool Agent
============================

Extends the bash agent with developer-oriented tools that mirror a real coding
assistant (read, write, edit, glob, grep, websearch), plus bash as a fallback.

What you'll learn:
------------------
    - Specialized file tools vs raw bash — why constrained tools beat "do
      anything" for reliability and safety.
    - A @tool decorator that auto-builds OpenAI schemas + dispatch from a
      single declaration per tool (no more keeping two dicts in sync).
    - Graceful error handling when the LLM produces malformed tool calls.

What changed from Chapter 01:
-----------------------------
    1. Added developer tools: read_file, write_file, edit_file, glob, grep.
    2. Added websearch (DuckDuckGo) for external information.
    3. System prompt guides the LLM to prefer specialized tools over bash.
    4. @tool decorator replaces the verbose TOOLS list + TOOL_DISPATCH dict.

Usage:
------
    $ python ch_02_multi_tool_agent.py "Read ch_01_bash_agent.py and summarize it"
    $ python ch_02_multi_tool_agent.py "Find all Python files that import json"
    $ python ch_02_multi_tool_agent.py "Search the web for Python 3.13 new features"
"""

import argparse
import fnmatch
import inspect
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ddgs import DDGS
from docstring_parser import parse
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _shell(command: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a shell command and return the CompletedProcess result."""
    return subprocess.run(command, shell=True, text=True, capture_output=True, timeout=timeout)


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


def _git_files() -> list[str]:
    """List all git-tracked + untracked-not-ignored files, or [] on failure."""
    try:

        def s(cmd):
            return _shell(cmd, timeout=5).stdout.strip()

        raw = s("git ls-files") + "\n" + s("git ls-files --others --exclude-standard")
        return sorted(set(filter(None, raw.splitlines())))
    except Exception:
        return []


def gather_context() -> str:
    """Build a project-aware system prompt from the current environment."""
    cwd = os.getcwd()

    def s(cmd):
        return _shell(cmd, timeout=5).stdout.strip()

    files = _git_files() or sorted(p.name for p in Path(cwd).iterdir() if not p.name.startswith("."))

    try:
        git_info = (
            f"\n## Git\nBranch: {s('git branch --show-current')}"
            f"\nStatus: {s('git status --short') or '(clean)'}"
            f"\nRecent commits:\n{s('git log --oneline -5')}"
        )
    except Exception:
        git_info = ""

    return f"""\
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

# Environment
- Working directory: {cwd}
- Platform: {platform.system()} {platform.release()}

## File tree
{chr(10).join(files) if files else "(empty)"}
{git_info}"""


SYSTEM_PROMPT = gather_context()


# ---------------------------------------------------------------------------
# Tool registry — @tool decorator builds OpenAI schemas + dispatch in one place
# ---------------------------------------------------------------------------

TOOLS: list[dict] = []  # OpenAI function-calling schemas
DISPATCH: dict[str, callable] = {}  # name -> handler(**kwargs)


def tool(func):
    """Decorator: registers a function as an agent tool using its signature and docstring."""
    doc = parse(func.__doc__)
    sig = inspect.signature(func)

    properties = {}
    required = []

    # Map Python types to JSON schema types
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "string"}

    for param_name, param in sig.parameters.items():
        doc_param = next((p for p in doc.params if p.arg_name == param_name), None)

        # Get type
        t = param.annotation
        p_type = type_map.get(t, "string")

        properties[param_name] = {
            "type": p_type,
            "description": doc_param.description if doc_param else "",
        }

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.short_description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }

    TOOLS.append(schema)
    DISPATCH[func.__name__] = func
    return func


# ---------------------------------------------------------------------------
# Tool implementations — each @tool declaration is the single source of truth
# ---------------------------------------------------------------------------


@tool
def bash(command: str) -> str:
    """Run a shell command. Use for git, tests, installs, or anything without a dedicated tool."""
    print(f"\033[96m$ {command}\033[0m")
    try:
        res = _shell(command)
        return res.stdout or res.stderr or "(no output)"
    except Exception as e:
        return f"Error: {e}"


@tool
def read(path: str, offset: int = 1, limit: int | None = None) -> str:
    """Read a file with numbered lines. Use offset/limit for large files."""
    print(f"\033[94m[read] {path}" + (f" (lines {offset}-{offset + limit - 1})" if limit else "") + "\033[0m")
    try:
        lines = Path(path).read_text().splitlines()
        start = max(0, offset - 1)
        end = start + limit if limit else len(lines)
        numbered = [f"{i + 1:>6}\t{line}" for i, line in enumerate(lines[start:end], start=start)]
        return "\n".join(numbered) if numbered else "(empty file)"
    except Exception as e:
        return f"Error: {e}"


@tool
def write(path: str, content: str) -> str:
    """Create or overwrite a file with the given content."""
    print(f"\033[94m[write] {path}\033[0m")
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit(path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing an exact unique string match."""
    print(f"\033[94m[edit] {path}\033[0m")
    try:
        text = Path(path).read_text()
        count = text.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path}"
        if count > 1:
            return f"Error: old_string appears {count} times in {path} (must be unique)"
        Path(path).write_text(text.replace(old_string, new_string, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def glob(pattern: str) -> str:
    """Find files matching a glob pattern (e.g. '**/*.py'). Returns matching paths."""
    print(f"\033[94m[glob] {pattern}\033[0m")
    files = _git_files()
    if files:
        matches = [f for f in files if fnmatch.fnmatch(f, pattern)]
    else:
        matches = sorted(str(p) for p in Path(".").glob(pattern) if p.is_file())
    return "\n".join(matches) if matches else "No matches found."


@tool
def grep(pattern: str, path: str = ".", include: str | None = None) -> str:
    """Search file contents for a regex pattern. Returns file:line:content matches."""
    print(f"\033[94m[grep] /{pattern}/" + (f" in {path}" if path != "." else "") + "\033[0m")
    cmd = f"grep -rn -E {json.dumps(pattern)} {json.dumps(path)}"
    if include:
        cmd += f" --include={json.dumps(include)}"
    try:
        output = _shell(cmd, timeout=10).stdout.strip()
        # Limit output to 100 lines
        lines = output.splitlines()[:100]
        return "\n".join(lines) if lines else "No matches found."
    except Exception as e:
        return f"Error: {e}"


@tool
def websearch(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo. Use for external information not in the project."""
    print(f"\033[92m[websearch] {query}\033[0m")
    try:
        results = DDGS().text(query, max_results=max_results)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tool execution: modified for multiple tools
# ---------------------------------------------------------------------------


def execute_tool_call(tool_call) -> str:
    """Parse a tool call, dispatch to the right handler, and return the output."""
    try:
        name = tool_call.function.name
        kwargs = json.loads(tool_call.function.arguments)
        handler = DISPATCH.get(name)
        return f"Unknown tool: {name}" if handler is None else handler(**kwargs)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"Error parsing tool call: {e}"


# ---------------------------------------------------------------------------
# Agent loop: same as chapter 01, but returns a trajectory for tests
# ---------------------------------------------------------------------------


def run_agent(task: str, max_steps: int = 10, enable_hitl: bool = False) -> dict:
    """Core agent loop: orchestrates the LLM turns and tool execution."""
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
    model = os.getenv("LLM_MODEL_ID")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(max_steps):
        print(f"\n--- Step {step + 1}/{max_steps} ---")

        response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS).choices[0].message
        messages.append(response.model_dump(exclude_none=True))

        if response.content:
            print(f"\033[93mAgent:\033[0m {response.content}")

        if response.tool_calls:
            for tool_call in response.tool_calls:
                output = execute_tool_call(tool_call)
                print(output)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
            continue

        if enable_hitl:
            feedback = input("\033[95m[HITL] Provide feedback (or press Enter to finish): \033[0m")
            if feedback.strip():
                messages.append({"role": "user", "content": feedback})
                continue

        print("Task marked complete.")
        break

    # Save the trajectory as a JSON file
    trajectory = {
        "task": task,
        "model": model,
        "max_steps": max_steps,
        "steps_used": step + 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "messages": [m if isinstance(m, dict) else m.model_dump() for m in messages],
    }

    Path("logs").mkdir(exist_ok=True)
    log_file = Path("logs") / f"trajectory_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    log_file.write_text(json.dumps(trajectory, indent=2, default=str))
    print(f"\nTrajectory saved: {log_file}")

    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Task to perform")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--hitl", action="store_true", help="Enable human-in-the-loop")
    args = parser.parse_args()

    run_agent(args.task, args.max_steps, args.hitl)


if __name__ == "__main__":
    main()
