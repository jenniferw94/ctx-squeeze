from ctx_squeeze import Segment, score_segments, select_by_score


def _seg(text, kind="text"):
    return Segment(text, 1, text.count("\n") + 1, kind)


def test_score_segments_of_empty_list_is_empty():
    assert score_segments([]) == []


def test_wordless_segment_scores_zero():
    segs = [_seg("hello world"), _seg("!!! ---")]
    scores = score_segments(segs)
    assert scores[1] == 0.0
    assert scores[0] > 0.0


def test_word_unique_to_one_segment_raises_its_score():
    # "common" appears in every segment, so it carries almost no weight;
    # "unique" appears in only one and should pull that segment's average up.
    segs = [
        _seg("common common common"),
        _seg("common common common"),
        _seg("common common unique"),
    ]
    scores = score_segments(segs)
    assert scores[2] > scores[0]
    assert scores[2] > scores[1]


def test_score_is_length_normalized_not_raw_word_count():
    # both segments are built entirely from a word that appears in both of
    # them, so it carries the same weight everywhere - repeating it ten
    # times shouldn't change the per-word average.
    segs = [_seg("widget"), _seg(" ".join(["widget"] * 10))]
    scores = score_segments(segs)
    assert scores[0] == scores[1]


def test_select_by_score_keeps_everything_under_a_generous_budget():
    segs = [_seg("alpha bravo"), _seg("charlie delta"), _seg("echo foxtrot")]
    kept = select_by_score(segs, budget=1000)
    assert kept == segs


def test_select_by_score_returns_nothing_for_zero_or_negative_budget():
    segs = [_seg("alpha bravo")]
    assert select_by_score(segs, budget=0) == []
    assert select_by_score(segs, budget=-5) == []


def test_select_by_score_returns_nothing_for_empty_segments():
    assert select_by_score([], budget=100) == []


def test_select_by_score_prefers_the_highest_scoring_segment():
    segs = [
        _seg("the the the the the the the the"),
        _seg("kubernetes ingress misconfiguration caused the outage"),
    ]
    kept = select_by_score(segs, budget=12)
    assert kept == [segs[1]]


def test_select_by_score_preserves_original_document_order():
    segs = [
        _seg("kubernetes ingress misconfiguration"),
        _seg("the the the the"),
        _seg("segfault traceback core dump"),
    ]
    kept = select_by_score(segs, budget=1000)
    assert [s.text for s in kept] == [s.text for s in segs]


def test_select_by_score_skips_a_pricier_segment_to_fit_a_cheaper_one():
    cheap = _seg("segfault")
    expensive = _seg("the the the the the the the the the the")
    segs = [expensive, cheap]
    kept = select_by_score(segs, budget=3)
    assert kept == [cheap]


def test_select_by_score_accepts_precomputed_scores():
    segs = [_seg("alpha"), _seg("beta")]
    # force the second segment to win even though score_segments would
    # otherwise call it a tie.
    kept = select_by_score(segs, budget=1, scores=[0.0, 1.0])
    assert kept == [segs[1]]
