"""Offline observations for synthetic tool calls (spec 04 §7).

Synthetic `read_file` outputs are produced by the harness's own `read_file` on a local copy of the
task's files at the base commit (`LocalWorkspace`), so caps, truncation and JSON layout are the
harness's by construction. Synthetic `grep` outputs are never generated (we do not have the whole
repository); only the teacher's recorded ones are reused.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..harness.local_sandbox import LocalWorkspace
from .locations import Location, normalize_path
from .traces import Trace

WORKSPACE = "/workspace/"
MAX_READ_LINES = 150
MAX_READ_CHARS = 10000


class UnseenCode(ValueError):
    """A synthetic read would show lines whose text we do not know (code reconstructed from the
    trace, `code_from_trace`)."""


class ObsRenderer:
    """`read_file` on the base-commit files of one task. `known` (optional): per file, the line
    numbers whose text is known; reads touching other lines raise `UnseenCode`."""

    def __init__(self, files: dict[str, str], known: dict[str, set[int]] | None = None):
        self.files = files
        self.known = known
        self.ws = LocalWorkspace(files)

    def check_known(self, path: str, a: int, b: int) -> None:
        if self.known is None:
            return
        k = self.known.get(path, set())
        missing = [n for n in range(a, b + 1) if n not in k]
        if missing:
            raise UnseenCode(f"{path}: {len(missing)} unseen line(s) in {a}-{b}")

    def __enter__(self) -> ObsRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.ws.close()

    def read_file(self, filepath: str, start_line: int | None = None, end_line: int | None = None) -> str:
        return self.ws.read_file(filepath, start_line, end_line)


@dataclass(frozen=True)
class ReadCall:
    filepath: str                          # as passed to the tool (/workspace/...)
    start_line: int
    end_line: int

    @property
    def args(self) -> dict:
        return {"filepath": self.filepath, "start_line": self.start_line, "end_line": self.end_line}


def merge_ranges(ranges: list[tuple[int, int]], gap: int = 2) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + gap + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def chunk_range(lines: list[str], a: int, b: int, max_lines: int = MAX_READ_LINES,
                max_chars: int = MAX_READ_CHARS) -> list[tuple[int, int]]:
    """Split [a, b] into reads the harness returns whole: at most `max_lines` lines and at most
    `max_chars` characters of joined content. (The harness cuts content at the character cap in
    the middle of a line and still reports the line-capped `end_line`; avoid that.) A single line
    longer than the cap gets a read of its own."""
    out: list[tuple[int, int]] = []
    start, size = a, -1
    for n in range(a, b + 1):
        add = len(lines[n - 1]) + 1
        if n > start and (n - start + 1 > max_lines or size + add > max_chars):
            out.append((start, n - 1))
            start, size = n, -1
        size += add
    out.append((start, b))
    return out


def read_calls_for(locations: list[Location], files: dict[str, str], renderer: ObsRenderer | None = None,
                   max_lines: int = MAX_READ_LINES, max_chars: int = MAX_READ_CHARS
                   ) -> list[tuple[ReadCall, str]]:
    """`read_file` calls that show every location, with their outputs.

    Files in first-mention order; within a file, overlapping or nearly adjacent spans are merged and
    split into reads the harness returns whole (`chunk_range`). Locations in files that do not exist
    at the base commit (new files) are skipped."""
    order: list[str] = []
    spans: dict[str, list[tuple[int, int]]] = {}
    for loc in locations:
        p = normalize_path(loc.path)
        text = files.get(p)
        if text is None:
            continue
        n = len(text.splitlines())
        a, b = max(1, loc.start), min(loc.end, n)
        if loc.symbol == "<file>":
            a, b = 1, n
        if a > b:
            continue
        if p not in spans:
            order.append(p)
            spans[p] = []
        spans[p].append((a, b))
    own = renderer is None
    r = renderer or ObsRenderer(files)
    out: list[tuple[ReadCall, str]] = []
    try:
        for p in order:
            lines = files[p].splitlines()
            for a, b in merge_ranges(spans[p]):
                for s, e in chunk_range(lines, a, b, max_lines, max_chars):
                    r.check_known(p, s, e)
                    call = ReadCall(WORKSPACE + p, s, e)
                    out.append((call, r.read_file(call.filepath, call.start_line, call.end_line)))
    finally:
        if own:
            r.close()
    return out


def check_recorded_reads(trace: Trace, files: dict[str, str], before_first_edit: bool = True
                         ) -> tuple[int, list[dict]]:
    """Re-render every recorded `read_file` (before the first edit) from `files`.
    Returns (number reproduced, mismatches). Reads of files missing from `files` are mismatches
    with reason `missing_file`."""
    calls = trace.calls_before_first_edit() if before_first_edit else list(trace.tool_calls())
    ok, bad = 0, []
    with ObsRenderer(files) as r:
        for c in calls:
            if c.tool != "read_file":
                continue
            p = normalize_path(c.args["filepath"])
            if p not in files:
                bad.append({"entry": c.entry_index, "path": p, "reason": "missing_file"})
                continue
            got = r.read_file(c.args["filepath"], c.args.get("start_line"), c.args.get("end_line"))
            if got == c.result:
                ok += 1
            else:
                bad.append({"entry": c.entry_index, "path": p, "reason": "differs"})
    return ok, bad
