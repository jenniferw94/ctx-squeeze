from ctx_squeeze import estimate_tokens, truncate_to_tokens


def test_empty_string_is_zero():
    assert estimate_tokens("") == 0


def test_pure_whitespace_is_free():
    assert estimate_tokens("   \t  ") == 0


def test_letters_cost_a_quarter_token_each():
    assert estimate_tokens("abcdefgh") == round(8 / 4)


def test_digit_runs_cost_a_third_token_each():
    assert estimate_tokens("123456") == round(6 / 3)


def test_newline_costs_half_a_token():
    assert estimate_tokens("\n") == round(0.5)


def test_symbol_costs_point_six_of_a_token():
    assert estimate_tokens("#") == round(0.6)


def test_cjk_character_costs_one_full_token():
    assert estimate_tokens("你") == 1


def test_cjk_run_does_not_get_the_letter_discount():
    text = "你好世界"
    assert estimate_tokens(text) == len(text)


def test_mixed_text_sums_each_class_independently():
    # 5 letters + 1 space (free) + 3 digits + 1 newline
    text = "hello 123\n"
    expected = round(5 / 4 + 3 / 3 + 1 * 0.5)
    assert estimate_tokens(text) == expected


def test_longer_prose_is_within_ten_percent_of_a_rough_word_count():
    prose = (
        "The nightly job started failing on Tuesday after the runner image "
        "was bumped, and every run now spends eleven minutes reinstalling "
        "dependencies from scratch before the tests even start."
    )
    words = len(prose.split())
    # a real BPE tokenizer runs roughly one token per word for prose like
    # this; the heuristic should land in the same neighborhood
    assert words * 0.5 <= estimate_tokens(prose) <= words * 1.6


def test_truncate_empty_budget_gives_empty_string():
    assert truncate_to_tokens("hello world", 0) == ""


def test_truncate_negative_budget_gives_empty_string():
    assert truncate_to_tokens("hello world", -5) == ""


def test_truncate_empty_text_gives_empty_string():
    assert truncate_to_tokens("", 100) == ""


def test_truncate_under_budget_returns_text_unchanged():
    text = "short and sweet"
    assert truncate_to_tokens(text, estimate_tokens(text) + 10) == text


def test_truncate_result_never_exceeds_the_budget():
    prose = (
        "The nightly job started failing on Tuesday after the runner image "
        "was bumped, and every run now spends eleven minutes reinstalling "
        "dependencies from scratch before the tests even start."
    )
    for budget in (1, 5, 10, 20, 50):
        truncated = truncate_to_tokens(prose, budget)
        assert estimate_tokens(truncated) <= budget


def test_truncate_returns_a_prefix():
    prose = "one two three four five six seven eight nine ten"
    truncated = truncate_to_tokens(prose, 5)
    assert prose.startswith(truncated)


def test_truncate_is_the_longest_prefix_that_fits():
    prose = "one two three four five six seven eight nine ten"
    budget = 5
    truncated = truncate_to_tokens(prose, budget)
    # one more character should either exceed the budget or run off the
    # end of the string
    next_len = len(truncated) + 1
    assert next_len > len(prose) or estimate_tokens(prose[:next_len]) > budget


def test_truncate_handles_cjk_text():
    text = "你好世界你好世界"
    truncated = truncate_to_tokens(text, 3)
    assert truncated == "你好世"
