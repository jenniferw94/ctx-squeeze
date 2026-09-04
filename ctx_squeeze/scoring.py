# Extractive selection: keep the segments that carry the most distinctive
# vocabulary, drop the rest. A word that shows up in every segment (boilerplate,
# common English) tells you nothing about which segment matters, so it's
# weighted down the same way document frequency works in search ranking.

import math
import re

from .tokens import estimate_tokens

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text):
    return _WORD_RE.findall(text.lower())


def score_segments(segments):
    """Score each segment by its average TF-IDF-weighted word frequency.

    A word's weight is its frequency within the segment times an inverse
    document frequency computed over all segments (words that appear in
    most segments count for less). The per-word weights are averaged over
    the segment's word count, so a long segment padded with common words
    doesn't automatically outscore a short, dense one. Segments with no
    words (empty text, or a code block made entirely of punctuation) score
    0.0. Returns a list of floats parallel to `segments`.
    """
    if not segments:
        return []

    doc_words = [_words(seg.text) for seg in segments]
    n_docs = len(segments)

    document_freq = {}
    for words in doc_words:
        for word in set(words):
            document_freq[word] = document_freq.get(word, 0) + 1

    idf = {
        word: math.log((n_docs + 1) / (count + 1)) + 1
        for word, count in document_freq.items()
    }

    scores = []
    for words in doc_words:
        if not words:
            scores.append(0.0)
            continue
        term_freq = {}
        for word in words:
            term_freq[word] = term_freq.get(word, 0) + 1
        weight = sum((count / len(words)) * idf[word] for word, count in term_freq.items())
        scores.append(weight)
    return scores


def select_by_score(segments, budget, scores=None):
    """Return the subset of `segments` that fits `budget` estimated tokens.

    Segments are ranked by `score_segments` (or a caller-supplied parallel
    list of `scores`, so a pipeline that already scored the segments doesn't
    pay for it twice) and added to the kept set highest-first, skipping any
    segment that would push the running total over budget - a later, cheaper
    segment can still be picked up after an earlier, pricier one is skipped.
    The result is returned in original document order, not score order, so
    callers can render it back into readable text.
    """
    if not segments or budget <= 0:
        return []

    if scores is None:
        scores = score_segments(segments)

    order = sorted(range(len(segments)), key=lambda i: scores[i], reverse=True)

    kept = set()
    remaining = budget
    for i in order:
        cost = estimate_tokens(segments[i].text)
        if cost <= remaining:
            kept.add(i)
            remaining -= cost

    return [segments[i] for i in sorted(kept)]
