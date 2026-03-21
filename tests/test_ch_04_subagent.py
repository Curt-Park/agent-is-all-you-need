"""Unit tests for ch_04 subagent functions (no LLM calls)."""

from ch_04_subagent import DISPATCH, TOOLS, _extract_final_response


def test_extract_final_response_finds_last_assistant():
    """Should return the last assistant message with content."""
    trajectory = {
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Do something."},
            {"role": "assistant", "content": "First response."},
            {"role": "assistant", "content": None, "tool_calls": [{}]},
            {"role": "tool", "tool_call_id": "1", "content": "tool output"},
            {"role": "assistant", "content": "Final answer."},
        ]
    }
    assert _extract_final_response(trajectory) == "Final answer."


def test_extract_final_response_skips_none_content():
    """Should skip assistant messages with no text content."""
    trajectory = {
        "messages": [
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": "The real answer."},
        ]
    }
    assert _extract_final_response(trajectory) == "The real answer."


def test_extract_final_response_skips_empty_string():
    """Should skip assistant messages with empty string content."""
    trajectory = {
        "messages": [
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "Real answer."},
        ]
    }
    assert _extract_final_response(trajectory) == "Real answer."


def test_extract_final_response_all_empty_returns_empty():
    """Should return empty string when only empty-string assistant messages exist."""
    trajectory = {
        "messages": [
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": None},
        ]
    }
    assert _extract_final_response(trajectory) == ""


def test_extract_final_response_no_assistant():
    """Should return fallback when no assistant message has content."""
    trajectory = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
    }
    assert "no text response" in _extract_final_response(trajectory).lower()


def test_task_tool_registered():
    """The task tool should be in TOOLS and DISPATCH."""
    names = {t["function"]["name"] for t in TOOLS}
    assert "task" in names
    assert "task" in DISPATCH


def test_tools_include_all_ch03_tools():
    """ch04 TOOLS should include all ch03 tools plus task."""
    names = {t["function"]["name"] for t in TOOLS}
    assert names >= {"bash", "read", "write", "edit", "glob", "grep", "websearch", "todo", "task"}
