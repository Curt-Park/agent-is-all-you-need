"""Unit tests for ch_03 planning agent functions (no LLM calls)."""

import pytest

from ch_03_planning_agent import render


def test_render_empty():
    """render() should handle an empty list."""
    assert render([]) == "TODO is empty."


def test_render_pending_items():
    """render() should show [ ] for pending items."""
    items = [{"id": 1, "text": "Do thing", "status": "pending"}]
    output = render(items)
    assert "[ ] #1: Do thing" in output
    assert "(0/1 completed)" in output


def test_render_in_progress():
    """render() should show [>] for in-progress items."""
    items = [{"id": 1, "text": "Working", "status": "in_progress"}]
    output = render(items)
    assert "[>] #1: Working" in output


def test_render_completed():
    """render() should show [x] for completed items."""
    items = [{"id": 1, "text": "Done", "status": "completed"}]
    output = render(items)
    assert "[x] #1: Done" in output
    assert "(1/1 completed)" in output


def test_render_mixed_statuses():
    """render() should show correct counts for mixed statuses."""
    items = [
        {"id": 1, "text": "A", "status": "completed"},
        {"id": 2, "text": "B", "status": "in_progress"},
        {"id": 3, "text": "C", "status": "pending"},
    ]
    output = render(items)
    assert "(1/3 completed)" in output


def test_render_invalid_status_raises():
    """render() should raise KeyError for an unrecognized status."""
    items = [{"id": 1, "text": "Bad", "status": "unknown"}]
    with pytest.raises(KeyError):
        render(items)
