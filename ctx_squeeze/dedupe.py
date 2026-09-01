# Near-duplicate detection for things like a retried tool call whose traceback
# is identical except for a timestamp. Word shingles absorb that kind of small
# edit; Jaccard similarity over the shingle sets gives a 0..1 score for how
# much two pieces of text overlap.


def shingles(text, size=5):
    """Return the set of `size`-word shingles in `text`.

    Words are lowercased before shingling so a duplicate that only differs by
    case still matches. A text with fewer than `size` words yields a single
    shingle covering all of it, so short segments can still be compared
    instead of always producing an empty set. An empty (or all-whitespace)
    text yields an empty set.
    """
    words = text.lower().split()
    if not words:
        return frozenset()
    if len(words) < size:
        return frozenset({tuple(words)})
    return frozenset(
        tuple(words[i : i + size]) for i in range(len(words) - size + 1)
    )


def jaccard(a, b):
    """Return the Jaccard similarity of two shingle sets, from 0.0 to 1.0.

    Two empty sets are treated as identical (1.0): they both came from
    empty/all-whitespace text, so there's nothing to distinguish them. One
    empty and one non-empty set has no overlap at all (0.0).
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
