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

    # Git info with graceful fallback
    try:
        branch = s("git branch --show-current")
        status = s("git status --short") or "(clean)"
        commits = s("git log --oneline -5")
        git_info = f"\n## Git\nBranch: {branch}\nStatus: {status}\nRecent commits:\n{commits}"
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
{"\n".join(files) if files else "(empty)"}
{git_info}"""


SYSTEM_PROMPT = gather_context()


# ---------------------------------------------------------------------------
# Color codes for terminal output
# ---------------------------------------------------------------------------

CYAN = "\033[96m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


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
    """Run a shell command. Use for git, tests, installs, or anything without a dedicated tool.

    Args:
        command: The shell command to execute.
    """
    print(f"{CYAN}$ {command}{RESET}")
    try:
        res = _shell(command)
        return res.stdout or res.stderr or "(no output)"
    except Exception as e:
        return f"Error: {e}"


@tool
def read(path: str, offset: int = 1, limit: int | None = None) -> str:
    """Read a file with numbered lines. Use offset/limit for large files.

    Args:
        path: Path to the file to read.
        offset: Starting line number (1-indexed). Defaults to 1.
        limit: Maximum number of lines to read. Defaults to None (read all).
    """
    suffix = f" (lines {offset}-{offset + limit - 1})" if limit else ""
    print(f"{BLUE}[read] {path}{suffix}{RESET}")
    try:
        text = Path(path).read_text()
        if not text:
            return "(empty file)"
        all_lines = text.splitlines()
        start_idx = max(0, offset - 1)
        end_idx = start_idx + limit if limit else len(all_lines)
        slice_ = all_lines[start_idx:end_idx]
        return "\n".join(f"{i + 1:>6}\t{line}" for i, line in enumerate(slice_, start=start_idx))
    except Exception as e:
        return f"Error: {e}"


@tool
def write(path: str, content: str) -> str:
    """Create or overwrite a file with the given content.

    Args:
        path: Destination file path.
        content: The content to write to the file.
    """
    print(f"{BLUE}[write] {path}{RESET}")
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit(path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing an exact unique string match.

    Args:
        path: Path to the file to edit.
        old_string: The exact string to find and replace. Must be unique in the file.
        new_string: The replacement string.
    """
    print(f"{BLUE}[edit] {path}{RESET}")
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
    """Find files matching a glob pattern (e.g. '**/*.py'). Returns matching paths.

    Args:
        pattern: Glob pattern to match files against.
    """
    print(f"{BLUE}[glob] {pattern}{RESET}")
    files = _git_files()
    if files:
        matches = [f for f in files if fnmatch.fnmatch(f, pattern)]
    else:
        matches = sorted(str(p) for p in Path(".").glob(pattern) if p.is_file())
    return "\n".join(matches) if matches else "No matches found."


@tool
def grep(pattern: str, path: str = ".", include: str | None = None) -> str:
    """Search file contents for a regex pattern. Returns file:line:content matches.

    Args:
        pattern: Regular expression pattern to search for.
        path: Directory or file path to search in. Defaults to ".".
        include: Optional glob pattern to filter files (e.g. "*.py").
    """
    path_suffix = f" in {path}" if path != "." else ""
    print(f"{BLUE}[grep] /{pattern}/{path_suffix}{RESET}")
    cmd = f"grep -rn -E '{pattern}' '{path}'"
    if include:
        cmd += f" --include='{include}'"
    try:
        output = _shell(cmd, timeout=10).stdout.strip()
        # Limit output to 100 lines
        lines = output.splitlines()[:100]
        return "\n".join(lines) if lines else "No matches found."
    except Exception as e:
        return f"Error: {e}"


@tool
def websearch(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo. Use for external information not in the project.

    Args:
        query: The search query.
        max_results: Maximum number of results to return. Defaults to 3.
    """
    print(f"{GREEN}[websearch] {query}{RESET}")
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
            print(f"{YELLOW}Agent:{RESET} {response.content}")

        if response.tool_calls:
            for tool_call in response.tool_calls:
                output = execute_tool_call(tool_call)
                print(output)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
            continue

        if enable_hitl:
            feedback = input(f"{MAGENTA}[HITL] Provide feedback (or press Enter to finish): {RESET}")
            if feedback.strip():
                messages.append({"role": "user", "content": feedback})
                continue

        print("Task marked complete.")
        break

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
    log_file.write_text(json.dumps(trajectory, indent=2, default=str))
    print(f"\nTrajectory saved: {log_file}")

    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Task to perform")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--hitl", action="store_true", help="Enable human-in-the-loop")
    args = parser.parse_args()

    print(TOOLS)
    # run_agent(args.task, args.max_steps, args.hitl)


if __name__ == "__main__":
    main()
