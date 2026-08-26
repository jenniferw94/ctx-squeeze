import re
from dataclasses import dataclass

# CommonMark-ish fence rule: an opening fence is 3+ backticks or tildes (an
# optional info string may follow). A closing fence is a line that is nothing
# but the same fence character, at least as long as the opener. Backtick and
# tilde fences don't close each other, which lets a fenced block quote ``` in
# its own body using ~~~ and vice versa.
_FENCE_OPEN_RE = re.compile(r"^(`{3,}|~{3,})")


@dataclass(frozen=True)
class Segment:
    text: str
    start_line: int
    end_line: int
    kind: str  # "text" or "code"


def _opening_fence(line):
    match = _FENCE_OPEN_RE.match(line.strip())
    return match.group(1) if match else None


def _is_closing_fence(line, fence_char, min_len):
    stripped = line.strip()
    return len(stripped) >= min_len and set(stripped) == {fence_char}


def split_segments(text):
    """Split `text` into paragraph and fenced-code-block segments.

    Blank lines separate ordinary paragraphs. A fenced code block (``` or
    ~~~, however long the fence and whatever the info string) is captured as
    a single "code" segment from its opening fence through its closing fence
    or, if it's never closed, through the end of the text - it is never
    split, so downstream stages that budget by segment can't truncate it
    in the middle. Returned segments carry 1-based, inclusive line numbers
    into the original text.
    """
    if not text:
        return []

    lines = text.split("\n")
    n = len(lines)
    segments = []
    para_lines = []
    para_start = None

    def flush_para(end_line):
        nonlocal para_start
        if para_lines:
            segments.append(
                Segment("\n".join(para_lines), para_start, end_line, "text")
            )
            del para_lines[:]
            para_start = None

    i = 0
    while i < n:
        line = lines[i]
        lineno = i + 1
        fence = _opening_fence(line)
        if fence is not None:
            flush_para(lineno - 1)
            code_lines = [line]
            fence_char, fence_len = fence[0], len(fence)
            j = i + 1
            while j < n and not _is_closing_fence(lines[j], fence_char, fence_len):
                code_lines.append(lines[j])
                j += 1
            if j < n:
                code_lines.append(lines[j])
                j += 1
            segments.append(Segment("\n".join(code_lines), lineno, j, "code"))
            i = j
            continue

        if line.strip() == "":
            flush_para(lineno - 1)
            i += 1
            continue

        if para_start is None:
            para_start = lineno
        para_lines.append(line)
        i += 1

    flush_para(n)
    return segments
