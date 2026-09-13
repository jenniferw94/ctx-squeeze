import pytest

from ctx_squeeze import parse_messages, prune_messages, to_dicts


def test_parse_messages_openai_shape_extracts_text_and_tool_ids():
    history = [
        {"role": "system", "content": "You are a careful build engineer."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "a.py"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "contents of a.py"},
    ]
    messages = parse_messages(history)
    assert messages[0].text == "You are a careful build engineer."
    assert messages[1].tool_use_ids == frozenset({"call_1"})
    assert messages[2].tool_result_ids == frozenset({"call_1"})


def test_parse_messages_anthropic_shape_extracts_blocks():
    history = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "let me check that"},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "contents of a.py"}],
        },
    ]
    messages = parse_messages(history)
    assert "let me check that" in messages[0].text
    assert messages[0].tool_use_ids == frozenset({"toolu_1"})
    assert messages[1].tool_result_ids == frozenset({"toolu_1"})
    assert "contents of a.py" in messages[1].text


def test_parse_messages_user_message_of_only_tool_results_is_not_a_turn():
    history = [
        {"role": "user", "content": "a real question"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
    ]
    messages = parse_messages(history)
    assert messages[0].is_user_turn is True
    assert messages[1].is_user_turn is False


def test_parse_messages_missing_role_raises():
    with pytest.raises(ValueError):
        parse_messages([{"content": "no role here"}])


def test_parse_messages_unsupported_content_type_raises():
    with pytest.raises(ValueError):
        parse_messages([{"role": "user", "content": 42}])


def test_prune_messages_empty_list_returns_empty_result():
    result = prune_messages([], budget=100)
    assert result.messages == []
    assert result.messages_in == 0
    assert result.messages_out == 0


def test_prune_messages_nonpositive_budget_drops_everything_but_counts_input():
    messages = parse_messages([{"role": "user", "content": "hello there"}])
    result = prune_messages(messages, budget=0)
    assert result.messages == []
    assert result.messages_in == 1
    assert result.original_tokens > 0


def test_prune_messages_generous_budget_keeps_everything():
    history = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    messages = parse_messages(history)
    result = prune_messages(messages, budget=1000, recent_turns=2)
    assert result.messages == messages
    assert result.messages_out == result.messages_in


def test_prune_messages_system_messages_always_survive():
    history = [
        {"role": "system", "content": "a long system prompt with plenty of words in it"},
        {"role": "user", "content": "hi"},
    ]
    messages = parse_messages(history)
    result = prune_messages(messages, budget=1, recent_turns=0)
    assert messages[0] in result.messages


def test_prune_messages_recent_turn_survives_whole_even_over_budget():
    history = [
        {"role": "user", "content": "kubernetes ingress misconfiguration caused the outage overnight"},
        {"role": "assistant", "content": "restarted the ingress controller and confirmed traffic recovered"},
    ]
    messages = parse_messages(history)
    result = prune_messages(messages, budget=1, recent_turns=1)
    assert result.messages == messages


def test_prune_messages_drops_older_turns_before_recent_ones():
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "an old unrelated question about widgets"},
        {"role": "assistant", "content": "an old unrelated answer about widgets"},
        {"role": "user", "content": "the current question"},
        {"role": "assistant", "content": "the current answer"},
    ]
    messages = parse_messages(history)
    result = prune_messages(messages, budget=1, recent_turns=1)
    assert messages[0] in result.messages
    assert messages[3] in result.messages
    assert messages[4] in result.messages
    assert messages[1] not in result.messages
    assert messages[2] not in result.messages
    assert "pruned 2 message" in result.notes[0]


def test_tool_call_and_result_are_never_split():
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "please read the file"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "a.py"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "contents of a.py, quite a bit of text here"},
        {"role": "assistant", "content": "here is a summary of the file"},
        {"role": "user", "content": "thanks, now do another unrelated thing"},
        {"role": "assistant", "content": "ok done"},
    ]
    messages = parse_messages(history)
    for budget in range(0, 40, 3):
        result = prune_messages(messages, budget=budget, recent_turns=1)
        call_kept = messages[2] in result.messages
        result_kept = messages[3] in result.messages
        assert call_kept == result_kept
        assert ("call_1" in result.pinned_tool_results) == call_kept


def test_to_dicts_round_trips_raw_dicts():
    history = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "hello"},
    ]
    messages = parse_messages(history)
    assert to_dicts(messages) == history
