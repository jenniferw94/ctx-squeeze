# Ties the standalone pieces together into the one call most callers want:
# split into segments, run them through a small pipeline of stages, and
# render the survivors back into text with markers where something was cut.

from dataclasses import dataclass, field

from .dedupe import jaccard, shingles
from .scoring import score_segments, select_by_score
from .segments import split_segments
from .tokens import estimate_tokens, truncate_to_tokens

_VALID_STAGES = {"head-tail", "score", "dedupe"}


@dataclass(frozen=True)
class SqueezeResult:
    text: str
    original_tokens: int
    final_tokens: int
    segments_in: int
    segments_out: int
    notes: list = field(default_factory=list)


def _dedupe_stage(segments, indices, threshold, shingle_size):
    kept = []
    kept_shingles = []
    dropped = 0
    for idx in indices:
        sh = shingles(segments[idx].text, shingle_size)
        if any(jaccard(sh, other) >= threshold for other in kept_shingles):
            dropped += 1
            continue
        kept.append(idx)
        kept_shingles.append(sh)
    return kept, dropped


def _score_stage(segments, indices, budget):
    subset = [segments[i] for i in indices]
    scores = score_segments(subset)
    selected = select_by_score(subset, budget, scores)
    selected_ids = {id(seg) for seg in selected}
    return [idx for idx, seg in zip(indices, subset) if id(seg) in selected_ids]


def _head_tail_stage(segments, indices, budget, head_ratio):
    if not indices or budget <= 0:
        return []

    head_budget = round(budget * head_ratio)
    head = []
    used = 0
    pos = 0
    while pos < len(indices):
        cost = estimate_tokens(segments[indices[pos]].text)
        if used + cost > head_budget:
            break
        head.append(indices[pos])
        used += cost
        pos += 1

    remaining = budget - used
    tail = []
    for idx in reversed(indices[pos:]):
        cost = estimate_tokens(segments[idx].text)
        if cost <= remaining:
            tail.append(idx)
            remaining -= cost
    tail.reverse()

    return head + tail


def _render(segments, kept, marker):
    n = len(segments)
    parts = []
    i = 0
    while i < n:
        if i in kept:
            parts.append(segments[i].text)
            i += 1
            continue
        j = i
        while j < n and j not in kept:
            j += 1
        if marker:
            parts.append(f"[{j - i} segments elided]")
        i = j
    return "\n\n".join(parts)


def squeeze(
    text,
    budget,
    strategy="score",
    head_ratio=0.5,
    jaccard_threshold=0.8,
    shingle_size=5,
    marker=True,
):
    """Compact `text` to fit within `budget` estimated tokens.

    `strategy` is a comma-separated pipeline of stage names applied left to
    right, each one narrowing the surviving set of segments the next stage
    sees: `dedupe` drops near-duplicates (by shingle/Jaccard similarity),
    `score` keeps the highest TF-IDF-scoring segments that fit the budget,
    and `head-tail` keeps a budget-share from the start and the rest from
    the end. Surviving segments are rendered back in original order, with
    runs of dropped segments replaced by a "[N segments elided]" marker
    (unless `marker` is False). If the rendered text (markers included)
    still comes in over budget, it's hard-truncated as a last resort, so
    `final_tokens` never exceeds `budget`.
    """
    original_tokens = estimate_tokens(text)
    segments = split_segments(text)
    if not segments or budget <= 0:
        return SqueezeResult("", original_tokens, 0, len(segments), 0, [])

    stages = [s.strip() for s in strategy.split(",") if s.strip()]
    for stage in stages:
        if stage not in _VALID_STAGES:
            raise ValueError(f"unknown strategy stage: {stage!r}")

    indices = list(range(len(segments)))
    notes = []
    for stage in stages:
        if stage == "dedupe":
            indices, dropped = _dedupe_stage(segments, indices, jaccard_threshold, shingle_size)
            if dropped:
                notes.append(f"dedupe dropped {dropped} near-duplicate segment(s)")
        elif stage == "score":
            indices = _score_stage(segments, indices, budget)
        elif stage == "head-tail":
            indices = _head_tail_stage(segments, indices, budget, head_ratio)

    kept = set(indices)
    final_text = _render(segments, kept, marker)
    final_tokens = estimate_tokens(final_text)
    if final_tokens > budget:
        final_text = truncate_to_tokens(final_text, budget)
        final_tokens = estimate_tokens(final_text)
        notes.append("hard-truncated to fit budget after assembling segments")

    return SqueezeResult(
        text=final_text,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        segments_in=len(segments),
        segments_out=len(indices),
        notes=notes,
    )
