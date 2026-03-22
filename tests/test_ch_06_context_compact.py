"""Unit tests for ch_06 context compact functions (no LLM calls)."""

from types import SimpleNamespace

from ch_06_context_compact import MICRO_COMPACT_AGE, estimate_tokens, micro_compact


# ---------------------------------------------------------------------------
# estimate_tokens()
# ---------------------------------------------------------------------------


def test_estimate_tokens_empty():
    """estimate_tokens() should return 0 for an empty message list."""
    assert estimate_tokens([]) == 0


def test_estimate_tokens_simple():
    """estimate_tokens() should return ~100 tokens for a message with 400 chars."""
    messages = [{"role": "user", "content": "a" * 400}]
    tokens = estimate_tokens(messages)
    assert 80 <= tokens <= 120


def test_estimate_tokens_multiple():
    """estimate_tokens() should return ~300 tokens for 2 messages with 1200 total chars."""
    messages = [
        {"role": "user", "content": "a" * 600},
        {"role": "assistant", "content": "b" * 600},
    ]
    tokens = estimate_tokens(messages)
    assert 250 <= tokens <= 350


def test_estimate_tokens_with_tool_calls():
    """estimate_tokens() should count assistant messages with tool_calls even without content."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "echo hello"}'},
                }
            ],
        }
    ]
    tokens = estimate_tokens(messages)
    assert tokens > 0


# ---------------------------------------------------------------------------
# micro_compact()
# ---------------------------------------------------------------------------


def _make_turn(tool_name: str, arguments: str, result: str, call_id: str):
    """Helper to build an assistant tool_call + tool result pair."""
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": arguments},
            }
        ],
    }
    tool_result_msg = {
        "role": "tool",
        "content": result,
        "tool_call_id": call_id,
    }
    return [assistant_msg, tool_result_msg]


def test_micro_compact_preserves_recent():
    """Within MICRO_COMPACT_AGE turns, no changes should be made."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    # Add recent tool turns (within age threshold)
    recent_turns = _make_turn("bash", '{"command": "echo 1"}', "1", "call_1")
    messages.extend(recent_turns)

    # micro_compact modifies in place and returns None
    micro_compact(messages)

    # Should be unchanged
    assert messages[-2]["tool_calls"][0]["function"]["arguments"] == '{"command": "echo 1"}'
    assert messages[-1]["content"] == "1"


def test_micro_compact_replaces_old_tool_results():
    """Tool results older than MICRO_COMPACT_AGE turns should be replaced with '[Previous: ...]'."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    # Add many old tool turns that exceed MICRO_COMPACT_AGE
    for i in range(MICRO_COMPACT_AGE + 3):
        old_turns = _make_turn(
            "bash",
            f'{{"command": "echo {i}"}}',
            f"result_{i}",
            f"call_{i}",
        )
        messages.extend(old_turns)

    micro_compact(messages)

    # The oldest tool results should be compacted (replaced with placeholder)
    # Find the tool result messages and check if some are replaced
    tool_result_msgs = [m for m in messages if m.get("role") == "tool"]
    compacted = [m for m in tool_result_msgs if m["content"].startswith("[Previous:")]
    assert len(compacted) > 0, "Expected some old tool results to be compacted"

    # The most recent ones should NOT be compacted
    recent = [m for m in tool_result_msgs if not m["content"].startswith("[Previous:")]
    assert len(recent) >= MICRO_COMPACT_AGE, "Expected recent tool results to be preserved"


def test_micro_compact_preserves_system_and_user():
    """System and user messages should never be modified."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello world"},
    ]
    # Add many old tool turns
    for i in range(MICRO_COMPACT_AGE + 1):
        old_turns = _make_turn(
            "bash",
            f'{{"command": "echo {i}"}}',
            f"result_{i}",
            f"call_{i}",
        )
        messages.extend(old_turns)

    micro_compact(messages)

    # System and user messages should be unchanged
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello world"


def test_micro_compact_preserves_tool_call_id():
    """The tool_call_id must survive compaction."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    # Add many old tool turns
    for i in range(MICRO_COMPACT_AGE + 1):
        old_turns = _make_turn(
            "bash",
            f'{{"command": "echo {i}"}}',
            f"result_{i}",
            f"call_{i}",
        )
        messages.extend(old_turns)

    micro_compact(messages)

    # All tool result messages should still have their tool_call_id
    tool_result_msgs = [m for m in messages if m.get("role") == "tool"]
    for msg in tool_result_msgs:
        assert "tool_call_id" in msg
        assert msg["tool_call_id"].startswith("call_")


def test_micro_compact_no_tool_calls():
    """Messages with no tool_calls should be a no-op."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    micro_compact(messages)
    assert messages[2]["content"] == "Hi there!"
    assert messages[3]["content"] == "How are you?"


# ---------------------------------------------------------------------------
# save_transcript()
# ---------------------------------------------------------------------------


def test_save_transcript_creates_file(tmp_path):
    """save_transcript should write a JSON file to the transcripts directory."""
    from ch_06_context_compact import save_transcript

    msgs = [{"role": "user", "content": "hello"}]
    path = save_transcript(msgs, transcripts_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".json"


def test_save_transcript_content(tmp_path):
    """The saved transcript should contain the original messages."""
    import json

    from ch_06_context_compact import save_transcript

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    path = save_transcript(msgs, transcripts_dir=tmp_path)
    saved = json.loads(path.read_text())
    assert saved == msgs


# ---------------------------------------------------------------------------
# auto_compact()
# ---------------------------------------------------------------------------


def test_auto_compact_preserves_system_message(tmp_path):
    """After auto_compact, the system message should be preserved."""
    from unittest.mock import MagicMock

    from ch_06_context_compact import auto_compact

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Summary of conversation."))
    ]

    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Do something."},
        {"role": "assistant", "content": "Done."},
    ]
    result = auto_compact(msgs, mock_client, "test-model", transcripts_dir=tmp_path)
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are a coding agent."


def test_auto_compact_produces_summary_message(tmp_path):
    """After auto_compact, there should be a [Context Summary] user message."""
    from unittest.mock import MagicMock

    from ch_06_context_compact import auto_compact

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Summary: did some work."))
    ]

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    result = auto_compact(msgs, mock_client, "test-model", transcripts_dir=tmp_path)
    assert len(result) == 2  # system + summary
    assert "[Context Summary]" in result[1]["content"]


def test_auto_compact_saves_transcript(tmp_path):
    """auto_compact should save a transcript before summarizing."""
    from unittest.mock import MagicMock

    from ch_06_context_compact import auto_compact

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Summary."))
    ]

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    auto_compact(msgs, mock_client, "test-model", transcripts_dir=tmp_path)
    transcripts = list(tmp_path.glob("transcript_*.json"))
    assert len(transcripts) == 1


# ---------------------------------------------------------------------------
# compact tool safety
# ---------------------------------------------------------------------------


def test_compact_outside_loop_returns_error():
    """Calling compact() outside an agent run should return a safe error."""
    from ch_06_context_compact import _COMPACT_STATE, compact

    _COMPACT_STATE.clear()
    result = compact()
    assert "Error" in result
