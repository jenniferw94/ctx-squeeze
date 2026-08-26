from ctx_squeeze import split_segments


def test_empty_text_gives_no_segments():
    assert split_segments("") == []


def test_only_blank_lines_gives_no_segments():
    assert split_segments("\n\n   \n\t\n") == []


def test_single_paragraph_no_trailing_newline():
    segs = split_segments("one line\nanother line")
    assert len(segs) == 1
    seg = segs[0]
    assert seg.kind == "text"
    assert seg.text == "one line\nanother line"
    assert (seg.start_line, seg.end_line) == (1, 2)


def test_two_paragraphs_separated_by_a_blank_line():
    text = "first paragraph\nstill first\n\nsecond paragraph"
    segs = split_segments(text)
    assert [s.text for s in segs] == ["first paragraph\nstill first", "second paragraph"]
    assert [(s.start_line, s.end_line) for s in segs] == [(1, 2), (4, 4)]
    assert all(s.kind == "text" for s in segs)


def test_multiple_blank_lines_collapse_without_empty_segments():
    text = "para one\n\n\n\npara two"
    segs = split_segments(text)
    assert [s.text for s in segs] == ["para one", "para two"]
    assert [(s.start_line, s.end_line) for s in segs] == [(1, 1), (5, 5)]


def test_leading_blank_lines_are_skipped():
    text = "\n\nfirst real paragraph"
    segs = split_segments(text)
    assert len(segs) == 1
    assert segs[0].text == "first real paragraph"
    assert (segs[0].start_line, segs[0].end_line) == (3, 3)


def test_fenced_code_block_is_a_single_segment():
    text = "before\n\n```python\ndef f():\n    return 1\n```\n\nafter"
    segs = split_segments(text)
    kinds = [s.kind for s in segs]
    assert kinds == ["text", "code", "text"]
    code = segs[1]
    assert code.text == "```python\ndef f():\n    return 1\n```"
    assert (code.start_line, code.end_line) == (3, 6)


def test_blank_lines_inside_a_fence_do_not_split_it():
    text = "```\nfirst\n\nsecond\n```"
    segs = split_segments(text)
    assert len(segs) == 1
    assert segs[0].kind == "code"
    assert segs[0].text == text
    assert (segs[0].start_line, segs[0].end_line) == (1, 5)


def test_unterminated_fence_runs_to_end_of_text():
    text = "para\n\n```\nunterminated\nstill open"
    segs = split_segments(text)
    assert [s.kind for s in segs] == ["text", "code"]
    code = segs[1]
    assert code.text == "```\nunterminated\nstill open"
    assert (code.start_line, code.end_line) == (3, 5)


def test_tilde_fence_is_supported_and_needs_matching_char_to_close():
    text = "~~~\ncode with a ``` inside\n~~~"
    segs = split_segments(text)
    assert len(segs) == 1
    assert segs[0].kind == "code"
    assert segs[0].text == text


def test_closing_fence_must_be_at_least_as_long_as_opener():
    text = "````\ncontent\n```\nstill inside\n````"
    segs = split_segments(text)
    assert len(segs) == 1
    assert segs[0].kind == "code"
    assert segs[0].text == text
    assert (segs[0].start_line, segs[0].end_line) == (1, 5)


def test_fence_immediately_after_paragraph_without_blank_line():
    text = "intro line\n```\ncode\n```"
    segs = split_segments(text)
    assert [s.kind for s in segs] == ["text", "code"]
    assert segs[0].text == "intro line"
    assert (segs[0].start_line, segs[0].end_line) == (1, 1)
    assert (segs[1].start_line, segs[1].end_line) == (2, 4)


def test_trailing_newline_does_not_create_an_extra_segment():
    text = "only paragraph\n"
    segs = split_segments(text)
    assert len(segs) == 1
    assert segs[0].text == "only paragraph"
    assert (segs[0].start_line, segs[0].end_line) == (1, 1)
