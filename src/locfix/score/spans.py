"""y*₊ token spans (spec 06 §3): the tokens of the lines a verified-patch edit adds.

For an `edit_file` call, the added lines are the lines of `new_string` that are not lines of its
`old_string`; for `write_file`, every line of `content`. The Gemma 4 chat template writes a string
argument verbatim between `<|"|>` delimiters (`key:<|"|>value<|"|>`, no escaping), so each value is
found as a substring of the rendered tool-call turn, and each added line's characters map to the
tokens that overlap them (offset mapping of the tokenizer; a token that straddles a line boundary,
e.g. a newline merged with indentation, is included).
"""
from __future__ import annotations

from typing import Any

QUOTE = '<|"|>'


class SpanError(ValueError):
    pass


def value_span(target: str, key: str, value: str, base: int = 0) -> tuple[int, int]:
    """Character span of a string argument's value inside a rendered tool-call turn."""
    needle = f"{key}:{QUOTE}{value}{QUOTE}"
    n = target.count(needle)
    if n != 1:
        raise SpanError(f"argument {key!r} found {n} times in the rendered turn (expected once)")
    i = target.index(needle) + len(key) + 1 + len(QUOTE)
    return base + i, base + i + len(value)


def added_line_offsets(tool: str, args: dict[str, Any]) -> list[tuple[int, int]]:
    """(start, end) character offsets, inside the new text, of the added non-empty lines."""
    if tool == "write_file":
        text, old = args["content"], set()
    elif tool == "edit_file":
        text, old = args["new_string"], set(args["old_string"].split("\n"))
    else:
        raise SpanError(f"not an edit call: {tool}")
    out, pos = [], 0
    for line in text.split("\n"):
        if line not in old and line.strip():
            out.append((pos, pos + len(line)))
        pos += len(line) + 1
    return out


def added_char_spans(target: str, base: int, tool: str, args: dict[str, Any]) -> list[tuple[int, int]]:
    """Absolute character spans (in the full rendered text) of the added lines of one call;
    `target` is the rendered turn, starting at `base` in the full text."""
    key = "content" if tool == "write_file" else "new_string"
    s, _ = value_span(target, key, args[key], base)
    return [(s + a, s + b) for a, b in added_line_offsets(tool, args)]


def tokens_overlapping(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    """Token indices whose character range overlaps any span (sorted, unique)."""
    out = []
    spans = sorted(spans)
    j = 0
    for t, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        k = j
        while k < len(spans) and spans[k][0] < b:
            if a < spans[k][1] and b > spans[k][0]:
                out.append(t)
                break
            k += 1
    return out
