"""Base-commit code reconstructed from what the teacher itself saw (fallback for `code/<id>.json`).

The data pipeline's `code/<id>.json` holds the full files at the base commit. When a task has no
such file, the lines the teacher printed before its first edit are the only trustworthy source:
complete lines of `read_file` outputs (with `total_lines`), and the stdout of plain
`sed -n 'a,bp' FILE` / `cat FILE` commands. Lines nobody printed are unknown: they are left empty in
the reconstructed text and listed as unknown, so that synthetic reads covering them are refused
(`render_obs.UnseenCode`) instead of showing invented code.

Only files whose length is known (from a `read_file` response or a complete `cat`) are returned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .locations import normalize_path
from .traces import ToolCall, Trace

_PREFIX = r"^(?:cd\s+/workspace\s*(?:&&|;)\s*)?"
_SED = re.compile(_PREFIX + r"sed\s+-n\s+['\"]?(\d+)(?:,(\d+))?p['\"]?\s+(\S+)\s*$")
_CAT = re.compile(_PREFIX + r"cat\s+(\S+)\s*$")


@dataclass
class SeenCode:
    lines: dict[str, dict[int, str]] = field(default_factory=dict)
    total: dict[str, int] = field(default_factory=dict)

    def add(self, path: str, start: int, texts: list[str]) -> None:
        d = self.lines.setdefault(path, {})
        for i, t in enumerate(texts):
            d.setdefault(start + i, t)

    def files(self) -> dict[str, str]:
        out = {}
        for p, n in self.total.items():
            d = self.lines.get(p, {})
            out[p] = "".join(d.get(i, "") + "\n" for i in range(1, n + 1))
        return out

    def known(self) -> dict[str, set[int]]:
        return {p: {i for i in self.lines.get(p, {}) if i <= n} for p, n in self.total.items()}

    def coverage(self) -> dict[str, float]:
        return {p: round(len(k) / max(self.total[p], 1), 3) for p, k in self.known().items()}


def _stdout_lines(out: str, max_chars: int) -> list[str]:
    """Complete lines of a stdout that the harness may have cut at `max_chars`."""
    parts = out.split("\n")
    if len(out) >= max_chars:
        return parts[:-1]                 # the last piece may be cut
    return parts[:-1] if out.endswith("\n") else parts


def _from_call(seen: SeenCode, c: ToolCall, max_file_chars: int, max_stdout_chars: int) -> None:
    if c.tool == "read_file":
        r = c.result_json()
        if r.get("status") != "ok" or "content" not in r:
            return
        p = normalize_path(c.args["filepath"])
        seen.total[p] = int(r["total_lines"])
        texts = r["content"].split("\n")[: int(r["end_line"]) - int(r["start_line"]) + 1]
        if len(r["content"]) >= max_file_chars:
            texts = texts[:-1]            # cut at the character cap, possibly mid-line
        seen.add(p, int(r["start_line"]), texts)
        return
    if c.tool != "run_command":
        return
    cmd = c.args.get("command", "").strip()
    r = c.result_json()
    if r.get("status") != "ok" or not isinstance(r.get("stdout"), str):
        return
    m = _SED.match(cmd)
    if m:
        seen.add(normalize_path(m.group(3)), int(m.group(1)), _stdout_lines(r["stdout"], max_stdout_chars))
        return
    m = _CAT.match(cmd)
    if m:
        p = normalize_path(m.group(1))
        lines = _stdout_lines(r["stdout"], max_stdout_chars)
        seen.add(p, 1, lines)
        if len(r["stdout"]) < max_stdout_chars and r["stdout"].endswith("\n"):
            seen.total.setdefault(p, len(lines))


def seen_code(trace: Trace, max_file_chars: int = 10000, max_stdout_chars: int = 5000) -> SeenCode:
    seen = SeenCode()
    for c in trace.calls_before_first_edit():
        _from_call(seen, c, max_file_chars, max_stdout_chars)
    return seen
