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
from pathlib import Path

from ddgs import DDGS
from docstring_parser import parse
from openai.types.chat import ChatCompletionMessageToolCallUnion
from pydantic import TypeAdapter

# reuse
from ch_01_bash_agent import Colors, _run_agent, gather_project_context, git_files, shell

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

""" + gather_project_context()


# ---------------------------------------------------------------------------
# Tool registry — @tool decorator builds OpenAI schemas + dispatch in one place
# ---------------------------------------------------------------------------
#
# The Big Idea
# ~~~~~~~~~~~~
# An agent needs two things per tool:
#   1. A JSON **schema** so the LLM knows what arguments are available.
#   2. A **handler** function to actually run when the LLM calls the tool.
#
# Keeping those two in sync by hand is tedious and error-prone — you'd have to
# update both a JSON dict and a Python function every time you change a param.
#
# Our solution: a @tool decorator that inspects the function's type hints and
# docstring at import time, auto-generates the OpenAI schema, and registers
# the handler — all from a single function definition.
#
# The two registries below are populated automatically by @tool:

TOOLS: list[dict] = []  # OpenAI function-calling schemas (sent to the API)
DISPATCH: dict[str, callable] = {}  # name -> handler(**kwargs) (used at runtime)


# -- Schema helpers ----------------------------------------------------------
#
# OpenAI's function-calling API expects a JSON Schema for each tool's
# parameters.  We use Pydantic's TypeAdapter to convert Python type hints
# (str, int, int | None, etc.) into JSON Schema snippets automatically.


def _resolve_refs(schema: dict, defs: dict) -> dict:
    """Inline $ref references so the schema is self-contained.

    Pydantic sometimes emits schemas with $ref pointers to a "$defs" section
    (e.g. for union types like `int | None`).  OpenAI expects a flat,
    self-contained schema, so we recursively replace every $ref with the
    actual definition it points to.
    """
    if "$ref" in schema:
        ref_name = schema["$ref"].rsplit("/", 1)[-1]
        return _resolve_refs(defs[ref_name], defs)
    return {k: _resolve_refs(v, defs) if isinstance(v, dict) else v for k, v in schema.items()}


def _type_to_schema(t) -> dict:
    """Convert a Python type annotation to a JSON schema dict.

    Examples:
        str          -> {"type": "string"}
        int          -> {"type": "integer"}
        int | None   -> {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    """
    schema = TypeAdapter(t).json_schema()
    defs = schema.pop("$defs", None)
    return _resolve_refs(schema, defs) if defs else schema


def _build_schema(func: callable, doc: object, sig: inspect.Signature) -> dict:
    """Build an OpenAI function-calling schema from a function's signature and docstring.

    Walks each parameter in the function signature and:
      - converts its type annotation to a JSON Schema property,
      - pulls its description from the parsed docstring,
      - marks it as required if it has no default value.

    The result is a dict ready to pass in the `tools` list of a chat completion.
    """
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Match this param to its docstring description (if any)
        doc_param = next((p for p in doc.params if p.arg_name == param_name), None)

        # Turn the type hint into a JSON Schema property
        prop = _type_to_schema(param.annotation)
        if doc_param and doc_param.description:
            prop["description"] = doc_param.description

        properties[param_name] = prop

        # Parameters without a default value are required
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Assemble the final schema in the format OpenAI expects
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.short_description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


# -- The @tool decorator itself ----------------------------------------------


def tool(func=None, *, tools: list[dict], dispatch: dict):
    """Decorator: registers a function as an agent tool.

    Usage:
        @tool  # Registers to module's TOOLS/DISPATCH
        @tool(tools=MY_TOOLS, dispatch=MY_DISPATCH)  # Registers to custom registries

    Args:
        func: The function to register (when used without arguments).
        tools: Custom tools list to append the schema to.
        dispatch: Custom dispatch dict to register the handler in.
    """

    def decorator(f):
        # 1. Parse the docstring to extract the short description + param docs
        doc = parse(f.__doc__)
        # 2. Inspect the function signature (param names, types, defaults)
        sig = inspect.signature(f)
        # 3. Combine both to produce the OpenAI-compatible JSON schema
        schema = _build_schema(f, doc, sig)

        # 4. Register: append the schema for the API, map the name for dispatch
        tools.append(schema)
        dispatch[f.__name__] = f
        return f

    # Support both @tool and @tool(tools=..., dispatch=...) syntax
    if func is None:
        return decorator  # Called with arguments: @tool(tools=..., dispatch=...)
    return decorator(func)  # Called without arguments: @tool


# ---------------------------------------------------------------------------
# Tool implementations — each @tool declaration is the single source of truth
# ---------------------------------------------------------------------------


@tool(tools=TOOLS, dispatch=DISPATCH)
def bash(command: str) -> str:
    """Run a shell command. Use for git, tests, installs, or anything without a dedicated tool.

    Args:
        command: The shell command to execute.
    """
    print(f"{Colors.CYAN}$ {command}{Colors.RESET}")
    try:
        res = shell(command)
        return res.stdout or res.stderr or "(no output)"
    except Exception as e:
        return f"Error: {e}"


@tool(tools=TOOLS, dispatch=DISPATCH)
def read(path: str, offset: int = 1, limit: int | None = None) -> str:
    """Read a file with numbered lines. Use offset/limit for large files.

    Args:
        path: Path to the file to read.
        offset: Starting line number (1-indexed). Defaults to 1.
        limit: Maximum number of lines to read. Defaults to None (read all).
    """
    suffix = f" (lines {offset}-{offset + limit - 1})" if limit else ""
    print(f"{Colors.BLUE}[read] {path}{suffix}{Colors.RESET}")
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


@tool(tools=TOOLS, dispatch=DISPATCH)
def write(path: str, content: str) -> str:
    """Create or overwrite a file with the given content.

    Args:
        path: Destination file path.
        content: The content to write to the file.
    """
    print(f"{Colors.BLUE}[write] {path}{Colors.RESET}")
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool(tools=TOOLS, dispatch=DISPATCH)
def edit(path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing an exact unique string match.

    Args:
        path: Path to the file to edit.
        old_string: The exact string to find and replace. Must be unique in the file.
        new_string: The replacement string.
    """
    print(f"{Colors.BLUE}[edit] {path}{Colors.RESET}")
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


@tool(tools=TOOLS, dispatch=DISPATCH)
def glob(pattern: str) -> str:
    """Find files matching a glob pattern (e.g. '**/*.py'). Returns matching paths.

    Args:
        pattern: Glob pattern to match files against.
    """
    print(f"{Colors.BLUE}[glob] {pattern}{Colors.RESET}")
    files = git_files()
    if files:
        matches = [f for f in files if fnmatch.fnmatch(f, pattern)]
    else:
        matches = sorted(str(p) for p in Path(".").glob(pattern) if p.is_file())
    return "\n".join(matches) if matches else "No matches found."


@tool(tools=TOOLS, dispatch=DISPATCH)
def grep(pattern: str, path: str = ".", include: str | None = None) -> str:
    """Search file contents for a regex pattern. Returns file:line:content matches.

    Args:
        pattern: Regular expression pattern to search for.
        path: Directory or file path to search in. Defaults to ".".
        include: Optional glob pattern to filter files (e.g. "*.py").
    """
    path_suffix = f" in {path}" if path != "." else ""
    print(f"{Colors.BLUE}[grep] /{pattern}/{path_suffix}{Colors.RESET}")
    cmd = f"grep -rn -E '{pattern}' '{path}'"
    if include:
        cmd += f" --include='{include}'"
    try:
        output = shell(cmd, timeout=10).stdout.strip()
        # Limit output to 100 lines
        lines = output.splitlines()[:100]
        return "\n".join(lines) if lines else "No matches found."
    except Exception as e:
        return f"Error: {e}"


@tool(tools=TOOLS, dispatch=DISPATCH)
def websearch(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo. Use for external information not in the project.

    Args:
        query: The search query.
        max_results: Maximum number of results to return. Defaults to 3.
    """
    print(f"{Colors.GREEN}[websearch] {query}{Colors.RESET}")
    try:
        results = DDGS().text(query, max_results=max_results)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tool execution: modified for multiple tools
# ---------------------------------------------------------------------------


def execute_tool_call(tool_call: ChatCompletionMessageToolCallUnion, dispatch: dict) -> str:
    """Parse a tool call, dispatch to the right handler, and return the output."""
    try:
        name = tool_call.function.name
        kwargs = json.loads(tool_call.function.arguments)
        handler = dispatch.get(name)
        return f"Unknown tool: {name}" if handler is None else handler(**kwargs)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"Error parsing tool call: {e}"
    except Exception as e:
        return f"Error: {e}"


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
