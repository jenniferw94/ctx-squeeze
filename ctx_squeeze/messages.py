# Message-level pruning for chat transcripts. Segments carry no notion of
# who said what or which call a result belongs to, so a document strategy
# can't be reused here: the unit that has to survive or go together is a
# whole message, and a tool call/result pair has to move as one or the
# structure sent back to the provider is invalid.

from dataclasses import dataclass, field

from .scoring import score_segments
from .segments import Segment
from .tokens import estimate_tokens


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    tokens: int
    is_user_turn: bool
    tool_use_ids: frozenset
    tool_result_ids: frozenset
    raw: dict


@dataclass(frozen=True)
class PruneResult:
    messages: list
    original_tokens: int
    final_tokens: int
    messages_in: int
    messages_out: int
    pinned_tool_results: frozenset
    notes: list = field(default_factory=list)


def _flatten_tool_result(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def parse_messages(history):
    """Parse a JSON chat transcript into a list of `Message`.

    Accepts the OpenAI shape (`content` a string, tool calls in a top-level
    `tool_calls` list, results in `tool` messages carrying a `tool_call_id`)
    and the Anthropic shape (`content` a list of blocks, with `tool_use` and
    `tool_result` block types). A message counts as opening a new user turn
    only if it carries real user text - a `user`-role message that is
    nothing but Anthropic tool-result blocks doesn't count, since it belongs
    to the turn that issued the call.
    """
    messages = []
    for raw in history:
        role = raw.get("role")
        if role is None:
            raise ValueError("message is missing 'role'")

        text_parts = []
        tool_use_ids = set()
        tool_result_ids = set()
        has_own_text = False

        for call in raw.get("tool_calls") or []:
            call_id = call.get("id")
            if call_id:
                tool_use_ids.add(call_id)
            fn = call.get("function") or {}
            text_parts.append(f"{fn.get('name', '')} {fn.get('arguments', '')}")

        tool_call_id = raw.get("tool_call_id")
        if tool_call_id:
            tool_result_ids.add(tool_call_id)

        content = raw.get("content")
        if isinstance(content, str):
            text_parts.append(content)
            has_own_text = True
        elif isinstance(content, list):
            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                    has_own_text = True
                elif block_type == "tool_use":
                    tool_id = block.get("id")
                    if tool_id:
                        tool_use_ids.add(tool_id)
                    text_parts.append(block.get("name", ""))
                elif block_type == "tool_result":
                    tool_id = block.get("tool_use_id")
                    if tool_id:
                        tool_result_ids.add(tool_id)
                    text_parts.append(_flatten_tool_result(block.get("content", "")))
        elif content is not None:
            raise ValueError(f"unsupported content type for message: {type(content).__name__}")

        text = "\n".join(part for part in text_parts if part)
        messages.append(
            Message(
                role=role,
                text=text,
                tokens=estimate_tokens(text),
                is_user_turn=(role == "user" and has_own_text),
                tool_use_ids=frozenset(tool_use_ids),
                tool_result_ids=frozenset(tool_result_ids),
                raw=raw,
            )
        )
    return messages


def _linked_groups(messages):
    # union-find over shared tool ids, so a call and every result that
    # references it (and anything chained off those) end up in one group
    parent = list(range(len(messages)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_id = {}
    for i, m in enumerate(messages):
        for tool_id in m.tool_use_ids | m.tool_result_ids:
            by_id.setdefault(tool_id, []).append(i)

    for indices in by_id.values():
        for other in indices[1:]:
            union(indices[0], other)

    groups = {}
    for i in range(len(messages)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _select_units(messages, units, budget):
    segments = []
    costs = []
    for members in units:
        segments.append(Segment("\n".join(messages[i].text for i in members), 0, 0, "message"))
        costs.append(sum(messages[i].tokens for i in members))

    scores = score_segments(segments)
    order = sorted(range(len(units)), key=lambda k: scores[k], reverse=True)

    kept = set()
    pinned = set()
    remaining = budget
    for k in order:
        if costs[k] <= remaining:
            for i in units[k]:
                kept.add(i)
                pinned.update(messages[i].tool_use_ids | messages[i].tool_result_ids)
            remaining -= costs[k]
    return kept, pinned


def prune_messages(messages, budget, recent_turns=2):
    """Prune a parsed transcript to fit `budget` estimated tokens.

    System messages always survive. The last `recent_turns` user turns
    (a turn is a user message and everything up to the next real user
    message) survive whole and are never truncated or dropped, even if
    that alone exceeds the budget. Everything older is a candidate: a
    tool call is never kept without its result or vice versa - they, and
    anything chained to them through a shared id, move together - and
    candidates are then picked by TF-IDF score, richest first, until the
    remaining budget runs out. `pinned_tool_results` reports the ids of
    every tool call/result pair that survived from outside the recent
    window, whether because pruning selected it or because it was linked
    to something already protected.
    """
    original_tokens = sum(m.tokens for m in messages)
    if not messages:
        return PruneResult([], 0, 0, 0, 0, frozenset(), [])
    if budget <= 0:
        return PruneResult([], original_tokens, 0, len(messages), 0, frozenset(), [])

    user_turns = [i for i, m in enumerate(messages) if m.is_user_turn]
    if recent_turns <= 0:
        boundary = len(messages)
    elif len(user_turns) <= recent_turns:
        boundary = 0
    else:
        boundary = user_turns[-recent_turns]

    protected = {i for i, m in enumerate(messages) if m.role == "system"}
    protected.update(range(boundary, len(messages)))

    kept = set(protected)
    pinned = set()
    candidate_units = []
    for members in _linked_groups(messages):
        if any(i in protected for i in members):
            for i in members:
                if i not in protected:
                    kept.add(i)
                    pinned.update(messages[i].tool_use_ids | messages[i].tool_result_ids)
        else:
            candidate_units.append(members)

    remaining = budget - sum(messages[i].tokens for i in kept)
    if remaining > 0 and candidate_units:
        selected, selected_pinned = _select_units(messages, candidate_units, remaining)
        kept |= selected
        pinned |= selected_pinned

    kept_messages = [messages[i] for i in sorted(kept)]
    notes = []
    if len(kept) < len(messages):
        notes.append(f"pruned {len(messages) - len(kept)} message(s) outside the last {recent_turns} turn(s)")

    return PruneResult(
        messages=kept_messages,
        original_tokens=original_tokens,
        final_tokens=sum(m.tokens for m in kept_messages),
        messages_in=len(messages),
        messages_out=len(kept_messages),
        pinned_tool_results=frozenset(pinned),
        notes=notes,
    )


def to_dicts(messages):
    """Convert `Message` objects back to the plain dicts they were parsed from."""
    return [m.raw for m in messages]
