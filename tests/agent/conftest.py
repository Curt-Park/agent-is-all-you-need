"""Shared pytest fixtures and helpers for agent integration tests."""

import json
import os
import shutil
from pathlib import Path

import pytest
from dotenv import load_dotenv

if os.path.exists(".env.test"):
    load_dotenv(".env.test", override=True)


@pytest.fixture
def workspace():
    """Create a temporary workspace directory for isolated testing."""
    ws = Path(f"test_workspace_{os.getpid()}").resolve()
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir()

    original_cwd = os.getcwd()
    os.chdir(ws)
    yield ws
    os.chdir(original_cwd)
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)


def get_tool_calls(trajectory):
    """Extract all tool calls from a trajectory."""
    calls = []
    for msg in trajectory["messages"]:
        if isinstance(msg, dict) and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                calls.append({"name": name, "args": args})
    return calls


def get_todo_calls(trajectory):
    """Extract parsed arguments from all todo tool calls in the trajectory."""
    return [c["args"] for c in get_tool_calls(trajectory) if c["name"] == "todo"]


def get_task_calls(trajectory):
    """Extract parsed arguments from all task tool calls in the trajectory."""
    return [c for c in get_tool_calls(trajectory) if c["name"] == "task"]


def get_skill_calls(trajectory):
    """Extract parsed arguments from all load_skill tool calls in the trajectory."""
    return [c for c in get_tool_calls(trajectory) if c["name"] == "load_skill"]


def get_compact_calls(trajectory):
    """Extract parsed arguments from all compact tool calls in the trajectory."""
    return [c for c in get_tool_calls(trajectory) if c["name"] == "compact"]
