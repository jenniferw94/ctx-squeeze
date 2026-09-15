# Thin argument-parsing shell around squeeze()/prune_messages(). Kept
# separate from pipeline.py and messages.py so the library has no dependency
# on argparse, sys, or any other CLI-only machinery.

import argparse
import json
import sys

from .messages import parse_messages, prune_messages
from .pipeline import squeeze


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="ctx-squeeze",
        description="Fit a document or chat transcript into an estimated-token budget.",
    )
    parser.add_argument("input", help="path to the input file, or - to read stdin")
    parser.add_argument("--budget", type=int, required=True, help="target size in estimated tokens")
    parser.add_argument(
        "--strategy",
        default="score",
        help="comma-separated pipeline: head-tail, score, dedupe (default: score)",
    )
    parser.add_argument("--head-ratio", type=float, default=0.5, dest="head_ratio")
    parser.add_argument("--jaccard", type=float, default=0.8, dest="jaccard")
    parser.add_argument("--shingle-size", type=int, default=5, dest="shingle_size")
    parser.add_argument("--messages", action="store_true", help="treat the input as a JSON chat transcript")
    parser.add_argument("--recent-turns", type=int, default=2, dest="recent_turns")
    parser.add_argument("--no-marker", action="store_true", dest="no_marker")
    parser.add_argument("--stats", action="store_true", help="print a token summary to stderr")
    parser.add_argument("--json", action="store_true", dest="json_output", help="emit a JSON report")
    parser.add_argument("-o", dest="output", default=None, help="write the result to PATH instead of stdout")
    return parser


def _read_input(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def _emit(args, text):
    if not text.endswith("\n"):
        text += "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


def _run_document(args, text):
    result = squeeze(
        text,
        budget=args.budget,
        strategy=args.strategy,
        head_ratio=args.head_ratio,
        jaccard_threshold=args.jaccard,
        shingle_size=args.shingle_size,
        marker=not args.no_marker,
    )
    if args.stats:
        print(
            f"kept {result.segments_out} of {result.segments_in} segments | "
            f"{result.original_tokens} -> {result.final_tokens} tokens (budget {args.budget})",
            file=sys.stderr,
        )
    if args.json_output:
        payload = {
            "text": result.text,
            "original_tokens": result.original_tokens,
            "final_tokens": result.final_tokens,
            "segments_in": result.segments_in,
            "segments_out": result.segments_out,
            "notes": result.notes,
        }
        _emit(args, json.dumps(payload, indent=2))
    else:
        _emit(args, result.text)
    return 0


def _render_messages(parsed, pruned, marker):
    # Same run-length-encode-the-gaps approach as pipeline._render, but the
    # unit is a whole message and the elided placeholder is itself a message.
    kept_ids = {id(m) for m in pruned.messages}
    rendered = []
    i = 0
    n = len(parsed)
    while i < n:
        if id(parsed[i]) in kept_ids:
            rendered.append(parsed[i].raw)
            i += 1
            continue
        j = i
        while j < n and id(parsed[j]) not in kept_ids:
            j += 1
        if marker:
            rendered.append({"role": "system", "content": f"[{j - i} earlier messages elided]"})
        i = j
    return rendered


def _run_messages(args, text):
    try:
        history = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"ctx-squeeze: invalid JSON input: {exc}", file=sys.stderr)
        return 1
    if not isinstance(history, list):
        print("ctx-squeeze: --messages input must be a JSON array", file=sys.stderr)
        return 1

    parsed = parse_messages(history)
    pruned = prune_messages(parsed, budget=args.budget, recent_turns=args.recent_turns)
    rendered = _render_messages(parsed, pruned, marker=not args.no_marker)

    if args.stats:
        print(
            f"kept {pruned.messages_out} of {pruned.messages_in} messages | "
            f"{pruned.original_tokens} -> {pruned.final_tokens} tokens (budget {args.budget})",
            file=sys.stderr,
        )
    if args.json_output:
        payload = {
            "messages": rendered,
            "original_tokens": pruned.original_tokens,
            "final_tokens": pruned.final_tokens,
            "messages_in": pruned.messages_in,
            "messages_out": pruned.messages_out,
            "notes": pruned.notes,
        }
        _emit(args, json.dumps(payload, indent=2))
    else:
        _emit(args, json.dumps(rendered, indent=2))
    return 0


def main(argv=None):
    args = _build_parser().parse_args(argv)

    try:
        text = _read_input(args.input)
    except OSError as exc:
        print(f"ctx-squeeze: {exc}", file=sys.stderr)
        return 1

    try:
        if args.messages:
            return _run_messages(args, text)
        return _run_document(args, text)
    except ValueError as exc:
        print(f"ctx-squeeze: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
