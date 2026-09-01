from ctx_squeeze import jaccard, shingles


def test_empty_text_gives_no_shingles():
    assert shingles("") == frozenset()
    assert shingles("   \n\t  ") == frozenset()


def test_short_text_yields_a_single_shingle():
    shingled = shingles("one two three", size=5)
    assert shingled == frozenset({("one", "two", "three")})


def test_shingles_are_lowercased():
    assert shingles("One Two Three Four Five", size=5) == shingles(
        "one two three four five", size=5
    )


def test_shingles_slide_over_words():
    shingled = shingles("a b c d e f", size=5)
    assert shingled == frozenset(
        {("a", "b", "c", "d", "e"), ("b", "c", "d", "e", "f")}
    )


def test_default_shingle_size_is_five():
    words = "one two three four five six seven"
    assert shingles(words) == shingles(words, size=5)


def test_jaccard_of_identical_shingle_sets_is_one():
    s = shingles("the quick brown fox jumps over the lazy dog")
    assert jaccard(s, s) == 1.0


def test_jaccard_of_disjoint_shingle_sets_is_zero():
    a = shingles("alpha beta gamma delta epsilon")
    b = shingles("zulu yankee xray whiskey victor")
    assert jaccard(a, b) == 0.0


def test_jaccard_of_two_empty_sets_is_one():
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_jaccard_of_one_empty_set_is_zero():
    s = shingles("some words here to shingle")
    assert jaccard(s, frozenset()) == 0.0
    assert jaccard(frozenset(), s) == 0.0


def test_jaccard_of_partially_overlapping_shingle_sets():
    a = frozenset({("a",), ("b",), ("c",)})
    b = frozenset({("b",), ("c",), ("d",)})
    assert jaccard(a, b) == 2 / 4


def test_near_duplicate_traceback_scores_high_similarity():
    first = (
        "Traceback at 10:00:01\n"
        "FileNotFoundError: config.yaml not found\n"
        "Retrying in 5 seconds\n"
    )
    second = (
        "Traceback at 10:00:47\n"
        "FileNotFoundError: config.yaml not found\n"
        "Retrying in 5 seconds\n"
    )
    similarity = jaccard(shingles(first, size=3), shingles(second, size=3))
    assert similarity > 0.4


def test_unrelated_paragraphs_score_low_similarity():
    a = shingles("the nightly build failed after the runner image was bumped")
    b = shingles("please review the pull request for the login page redesign")
    assert jaccard(a, b) < 0.2
