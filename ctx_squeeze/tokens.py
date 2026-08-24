# Character-class heuristic: no vocabulary file, no downloads, close enough
# to a real BPE tokenizer to budget against. Weights are chosen so English
# prose lands within ~10% of GPT/Claude-style tokenizers; see README.

_CJK_RANGES = (
    (0x3040, 0x30FF),  # hiragana, katakana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xAC00, 0xD7A3),  # hangul syllables
)


def _is_cjk(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def estimate_tokens(text):
    """Estimate the token count of `text` without a tokenizer.

    Runs of letters cost 1/4 token per character, digit runs 1/3, each CJK
    character is its own token, newlines are half a token (they carry
    structure a tokenizer usually spends a token on), and everything else
    that isn't whitespace (punctuation, symbols) costs 0.6. Plain whitespace
    is free.
    """
    if not text:
        return 0

    letters = digits = cjk = newlines = symbols = 0
    for ch in text:
        if ch == "\n":
            newlines += 1
        elif _is_cjk(ch):
            cjk += 1
        elif ch.isalpha():
            letters += 1
        elif ch.isdigit():
            digits += 1
        elif ch.isspace():
            continue
        else:
            symbols += 1

    total = letters / 4 + digits / 3 + cjk + newlines * 0.5 + symbols * 0.6
    return round(total)


def truncate_to_tokens(text, n):
    """Return the longest prefix of `text` that estimates at or under `n` tokens.

    Every character estimate_tokens looks at adds a non-negative amount to
    the running total, so the estimate is monotonic in prefix length and a
    binary search over character offsets finds the cut point in O(log len)
    calls instead of scanning one character at a time.
    """
    if n <= 0 or not text:
        return ""
    if estimate_tokens(text) <= n:
        return text

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= n:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]
