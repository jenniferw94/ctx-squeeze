import pytest

from ctx_squeeze import estimate_tokens, squeeze


def test_generous_budget_keeps_everything_unchanged():
    text = "first paragraph\n\nsecond paragraph\n\nthird paragraph"
    result = squeeze(text, budget=1000)
    assert result.text == text
    assert result.segments_out == result.segments_in == 3
    assert result.final_tokens <= 1000
    assert result.notes == []


def test_empty_text_returns_empty_result():
    result = squeeze("", budget=100)
    assert result.text == ""
    assert result.original_tokens == 0
    assert result.final_tokens == 0
    assert result.segments_in == 0
    assert result.segments_out == 0


def test_zero_or_negative_budget_drops_everything():
    text = "some paragraph here"
    result = squeeze(text, budget=0)
    assert result.text == ""
    assert result.segments_out == 0
    result = squeeze(text, budget=-5)
    assert result.text == ""


def test_score_strategy_never_exceeds_budget_and_marks_gaps():
    text = "\n\n".join(
        [
            "kubernetes ingress misconfiguration caused the outage",
            "the the the the the the the the the",
            "segfault traceback core dump on the worker node",
            "the the the the the the the the the the",
        ]
    )
    result = squeeze(text, budget=12, strategy="score")
    assert result.final_tokens <= 12
    assert result.segments_out < result.segments_in
    assert "[" in result.text and "elided]" in result.text


def test_marker_can_be_suppressed():
    text = "\n\n".join(["alpha kubernetes", "the the the the the the", "beta segfault"])
    result = squeeze(text, budget=6, strategy="score", marker=False)
    assert "elided" not in result.text


def test_dedupe_stage_drops_near_duplicates_and_notes_it():
    # only the leading timestamp differs, so all but one 5-word shingle
    # still overlaps between the two paragraphs.
    text = "\n\n".join(
        [
            "10:01am deploy failed with a timeout after retrying the request twice",
            "10:02am deploy failed with a timeout after retrying the request twice",
            "an unrelated paragraph about something else entirely",
        ]
    )
    result = squeeze(text, budget=1000, strategy="dedupe", jaccard_threshold=0.6)
    assert result.segments_out == 2
    assert any("dedupe dropped" in note for note in result.notes)


def test_dedupe_then_score_pipeline_composes_left_to_right():
    text = "\n\n".join(
        [
            "10:01am deploy failed with a timeout after retrying the request twice",
            "10:02am deploy failed with a timeout after retrying the request twice",
            "kubernetes ingress misconfiguration caused the outage",
        ]
    )
    result = squeeze(text, budget=1000, strategy="dedupe,score", jaccard_threshold=0.6)
    assert result.segments_out == 2
    assert "10:0" in result.text
    assert "kubernetes" in result.text


def test_head_tail_strategy_keeps_both_ends():
    text = "\n\n".join(["start of the document", "middle filler content", "end of the document"])
    result = squeeze(text, budget=1000, strategy="head-tail")
    assert result.text.startswith("start of the document")
    assert result.text.endswith("end of the document")


def test_head_tail_respects_budget_split():
    paragraphs = [f"paragraph number {i} with some filler words in it" for i in range(10)]
    text = "\n\n".join(paragraphs)
    result = squeeze(text, budget=20, strategy="head-tail", head_ratio=0.5)
    assert result.final_tokens <= 20
    assert result.text.splitlines()[0] == paragraphs[0]


def test_code_fence_is_never_split_across_a_truncation():
    text = "intro paragraph\n\n```python\ndef f():\n    return 1\n```\n\noutro paragraph"
    result = squeeze(text, budget=1000, strategy="score")
    assert "```python\ndef f():\n    return 1\n```" in result.text


def test_unknown_strategy_stage_raises():
    with pytest.raises(ValueError):
        squeeze("some text", budget=10, strategy="bogus")


def test_final_tokens_never_exceed_budget_even_with_marker_overhead():
    paragraphs = [f"paragraph {i} " + "the " * 8 for i in range(30)]
    text = "\n\n".join(paragraphs)
    budget = 15
    result = squeeze(text, budget=budget, strategy="score")
    assert result.final_tokens <= budget
    assert estimate_tokens(result.text) == result.final_tokens
