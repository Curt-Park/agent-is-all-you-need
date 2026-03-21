"""Unit tests for ch_02 multi-tool agent functions (no LLM calls)."""

import inspect
from types import SimpleNamespace

from docstring_parser import parse

from ch_02_multi_tool_agent import (
    DISPATCH,
    TOOLS,
    _build_schema,
    _resolve_refs,
    _type_to_schema,
    edit,
    execute_tool_call,
    glob,
    grep,
    read,
    tool,
    write,
)

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def test_type_to_schema_str():
    """str should map to {"type": "string"}."""
    schema = _type_to_schema(str)
    assert schema["type"] == "string"


def test_type_to_schema_int():
    """int should map to {"type": "integer"}."""
    schema = _type_to_schema(int)
    assert schema["type"] == "integer"


def test_type_to_schema_optional_int():
    """int | None should produce an anyOf with integer and null."""
    schema = _type_to_schema(int | None)
    assert "anyOf" in schema
    types = {s["type"] for s in schema["anyOf"]}
    assert "integer" in types
    assert "null" in types


def test_resolve_refs_inlines_refs():
    """_resolve_refs should replace $ref pointers with actual definitions."""
    defs = {"MyType": {"type": "string"}}
    schema = {"$ref": "#/$defs/MyType"}
    resolved = _resolve_refs(schema, defs)
    assert resolved == {"type": "string"}


def test_resolve_refs_no_refs():
    """_resolve_refs should pass through schemas without $ref unchanged."""
    schema = {"type": "integer"}
    resolved = _resolve_refs(schema, {})
    assert resolved == {"type": "integer"}


# ---------------------------------------------------------------------------
# _build_schema()
# ---------------------------------------------------------------------------


def test_build_schema_includes_docstring_descriptions():
    """_build_schema should inject docstring param descriptions into the schema."""

    def sample(name: str, count: int = 5) -> str:
        """Do something.

        Args:
            name: The name to use.
            count: How many times.
        """

    doc = parse(sample.__doc__)
    sig = inspect.signature(sample)
    schema = _build_schema(sample, doc, sig)

    props = schema["function"]["parameters"]["properties"]
    assert props["name"]["description"] == "The name to use."
    assert props["count"]["description"] == "How many times."
    assert schema["function"]["description"] == "Do something."


def test_build_schema_all_required():
    """_build_schema should mark all params as required when none have defaults."""

    def sample(a: str, b: int) -> str:
        """Test func.

        Args:
            a: First.
            b: Second.
        """

    doc = parse(sample.__doc__)
    sig = inspect.signature(sample)
    schema = _build_schema(sample, doc, sig)

    assert schema["function"]["parameters"]["required"] == ["a", "b"]


def test_build_schema_no_required():
    """_build_schema should have empty required list when all params have defaults."""

    def sample(a: str = "x", b: int = 0) -> str:
        """Test func.

        Args:
            a: First.
            b: Second.
        """

    doc = parse(sample.__doc__)
    sig = inspect.signature(sample)
    schema = _build_schema(sample, doc, sig)

    assert schema["function"]["parameters"]["required"] == []


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


def test_tool_decorator_registers_schema_and_dispatch():
    """@tool should add a schema to tools and a handler to dispatch."""
    test_tools = []
    test_dispatch = {}

    @tool(tools=test_tools, dispatch=test_dispatch)
    def my_func(name: str, count: int = 5) -> str:
        """Do something useful.

        Args:
            name: The name to use.
            count: How many times. Defaults to 5.
        """
        return f"{name} x {count}"

    assert len(test_tools) == 1
    schema = test_tools[0]
    assert schema["function"]["name"] == "my_func"
    assert "name" in schema["function"]["parameters"]["properties"]
    assert "count" in schema["function"]["parameters"]["properties"]
    assert "name" in schema["function"]["parameters"]["required"]
    assert "count" not in schema["function"]["parameters"]["required"]
    assert test_dispatch["my_func"] is my_func


def test_tool_decorator_preserves_function():
    """@tool should return the original function unchanged."""
    test_tools = []
    test_dispatch = {}

    @tool(tools=test_tools, dispatch=test_dispatch)
    def add(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        return a + b

    assert add(2, 3) == 5


# ---------------------------------------------------------------------------
# Module-level TOOLS / DISPATCH registries
# ---------------------------------------------------------------------------


def test_all_tools_registered():
    """All expected tools should be in the module TOOLS list."""
    names = {t["function"]["name"] for t in TOOLS}
    assert names >= {"bash", "read", "write", "edit", "glob", "grep", "websearch"}


def test_dispatch_matches_tools():
    """Every tool schema should have a matching dispatch handler."""
    for t in TOOLS:
        name = t["function"]["name"]
        assert name in DISPATCH, f"{name} in TOOLS but not in DISPATCH"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def test_read_file(tmp_path):
    """read() should return numbered lines."""
    f = tmp_path / "hello.txt"
    f.write_text("line1\nline2\nline3\n")
    output = read(str(f))
    assert "line1" in output
    assert "line2" in output
    assert "line3" in output


def test_read_file_with_offset_and_limit(tmp_path):
    """read() should respect offset and limit."""
    f = tmp_path / "lines.txt"
    f.write_text("a\nb\nc\nd\ne\n")
    output = read(str(f), offset=2, limit=2)
    assert "b" in output
    assert "c" in output
    assert "a" not in output
    assert "d" not in output


def test_read_empty_file(tmp_path):
    """read() should return '(empty file)' for empty files."""
    f = tmp_path / "empty.txt"
    f.write_text("")
    output = read(str(f))
    assert output == "(empty file)"


def test_read_nonexistent_file():
    """read() should return an error for missing files."""
    output = read("/nonexistent/file.txt")
    assert "Error" in output


def test_write_creates_file(tmp_path):
    """write() should create a file with the given content."""
    f = tmp_path / "out.txt"
    output = write(str(f), "hello world")
    assert "Wrote" in output
    assert f.read_text() == "hello world"


def test_write_creates_parent_dirs(tmp_path):
    """write() should create parent directories as needed."""
    f = tmp_path / "sub" / "dir" / "file.txt"
    write(str(f), "nested")
    assert f.read_text() == "nested"


def test_edit_replaces_unique_match(tmp_path):
    """edit() should replace a unique string."""
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    return 1\n")
    output = edit(str(f), "return 1", "return 42")
    assert "Edited" in output
    assert "return 42" in f.read_text()


def test_edit_rejects_missing_string(tmp_path):
    """edit() should error when old_string is not found."""
    f = tmp_path / "code.py"
    f.write_text("hello")
    output = edit(str(f), "missing", "replacement")
    assert "Error" in output
    assert "not found" in output


def test_edit_rejects_non_unique_match(tmp_path):
    """edit() should error when old_string appears multiple times."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\nx = 1\n")
    output = edit(str(f), "x = 1", "x = 2")
    assert "Error" in output
    assert "2 times" in output


def test_glob_finds_files(tmp_path, monkeypatch):
    """glob() should find files matching a pattern."""
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    monkeypatch.chdir(tmp_path)
    output = glob("*.py")
    assert "a.py" in output
    assert "b.py" in output
    assert "c.txt" not in output


def test_glob_no_matches(tmp_path, monkeypatch):
    """glob() should report no matches."""
    monkeypatch.chdir(tmp_path)
    output = glob("*.xyz")
    assert "No matches" in output


def test_grep_finds_matches(tmp_path, monkeypatch):
    """grep() should find matching lines in files."""
    (tmp_path / "code.py").write_text("def hello():\n    pass\n")
    monkeypatch.chdir(tmp_path)
    output = grep("def hello", str(tmp_path))
    assert "def hello" in output


def test_grep_no_matches(tmp_path, monkeypatch):
    """grep() should report no matches."""
    (tmp_path / "code.py").write_text("nothing here\n")
    monkeypatch.chdir(tmp_path)
    output = grep("nonexistent_pattern_xyz", str(tmp_path))
    assert "No matches" in output


def test_grep_with_include_filter(tmp_path, monkeypatch):
    """grep() should filter files with the include parameter."""
    (tmp_path / "code.py").write_text("target_string\n")
    (tmp_path / "notes.txt").write_text("target_string\n")
    monkeypatch.chdir(tmp_path)
    output = grep("target_string", str(tmp_path), include="*.py")
    assert "code.py" in output
    assert "notes.txt" not in output


# ---------------------------------------------------------------------------
# execute_tool_call()
# ---------------------------------------------------------------------------


def _make_tool_call(name: str, arguments: str):
    """Helper to create a mock tool call object."""
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def test_execute_tool_call_valid():
    """execute_tool_call() should dispatch to the correct handler."""
    tc = _make_tool_call("bash", '{"command": "echo ok"}')
    output = execute_tool_call(tc, DISPATCH)
    assert "ok" in output


def test_execute_tool_call_unknown_tool():
    """execute_tool_call() should return error for unknown tool names."""
    tc = _make_tool_call("nonexistent_tool", "{}")
    output = execute_tool_call(tc, DISPATCH)
    assert "Unknown tool" in output


def test_execute_tool_call_bad_json():
    """execute_tool_call() should return error for malformed JSON."""
    tc = _make_tool_call("bash", "not json")
    output = execute_tool_call(tc, DISPATCH)
    assert "Error parsing tool call" in output
