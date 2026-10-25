"""Unified diffs: parsing, strict application, and conversion to harness `edit_file` calls.

`patch_to_edit_calls(files_before, patch)` returns the `edit_file` / `write_file` calls that turn
the original files into the patched ones. Each `old_string` is the smallest block around a change
that is unique in the file at that moment (extended with context lines until unique). The result is
validated by running the calls through the harness's own `edit_file` on a local copy
(`LocalWorkspace`) and requiring every edit to match with the `exact` strategy and the final files
to equal the patched files.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(ValueError):
    pass


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[str]                      # with leading ' ', '-', '+' (and '\\' markers)

    def changed_old_lines(self) -> list[int]:
        """1-based old-file line numbers that are removed, plus the anchor line of pure insertions."""
        out, ln, pending_insert = [], self.old_start, False
        for line in self.lines:
            tag = line[:1]
            if tag == " ":
                if pending_insert:
                    out.append(max(ln - 1, 1))
                    pending_insert = False
                ln += 1
            elif tag == "-":
                out.append(ln)
                ln += 1
            elif tag == "+":
                pending_insert = True
        if pending_insert:
            out.append(max(ln - 1, 1))
        return sorted(set(out))


@dataclass
class FilePatch:
    old_path: str | None                   # None for new files
    new_path: str | None                   # None for deleted files
    hunks: list[Hunk] = field(default_factory=list)
    binary: bool = False

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    @property
    def is_new(self) -> bool:
        return self.old_path is None

    @property
    def is_deleted(self) -> bool:
        return self.new_path is None

    def changed_old_lines(self) -> list[int]:
        return sorted({n for h in self.hunks for n in h.changed_old_lines()})


def _strip_prefix(p: str) -> str | None:
    p = p.split("\t")[0].strip()
    if p == "/dev/null":
        return None
    return p[2:] if p.startswith(("a/", "b/")) else p


def parse_patch(patch: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    cur: FilePatch | None = None
    lines = patch.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            cur = FilePatch(old_path=None, new_path=None)
            m = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if m:
                cur.old_path, cur.new_path = m.group(1), m.group(2)
            files.append(cur)
        elif line.startswith("new file mode") and cur is not None:
            cur.old_path = None
        elif line.startswith("deleted file mode") and cur is not None:
            cur.new_path = None
        elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
            if cur is not None:
                cur.binary = True
        elif line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            if cur is None or cur.hunks:
                cur = FilePatch(old_path=None, new_path=None)
                files.append(cur)
            cur.old_path = _strip_prefix(line[4:])
            cur.new_path = _strip_prefix(lines[i + 1][4:])
            i += 1
        elif line.startswith("@@"):
            m = _HUNK.match(line)
            if not m or cur is None:
                raise PatchError(f"bad hunk header: {line!r}")
            old_len = int(m.group(2)) if m.group(2) is not None else 1
            new_len = int(m.group(4)) if m.group(4) is not None else 1
            h = Hunk(int(m.group(1)), old_len, int(m.group(3)), new_len, [])
            i += 1
            seen_old = seen_new = 0
            while i < len(lines) and (seen_old < old_len or seen_new < new_len or
                                      (i < len(lines) and lines[i].startswith("\\"))):
                hl = lines[i]
                if hl.startswith("\\"):
                    h.lines.append(hl)
                elif hl[:1] in (" ", "-", "+") or hl == "":
                    hl = hl if hl else " "
                    h.lines.append(hl)
                    if hl[0] in (" ", "-"):
                        seen_old += 1
                    if hl[0] in (" ", "+"):
                        seen_new += 1
                else:
                    break
                i += 1
            cur.hunks.append(h)
            continue
        i += 1
    return files


def apply_file_patch(old_text: str | None, fp: FilePatch) -> str | None:
    """Apply one file's hunks strictly (context must match). Returns None for deletions."""
    if fp.binary:
        raise PatchError(f"{fp.path}: binary patch not supported")
    if fp.is_deleted:
        return None
    old_lines = [] if old_text is None else old_text.splitlines(keepends=True)
    out: list[str] = []
    pos = 0                                          # 0-based index into old_lines
    for h in fp.hunks:
        start = h.old_start - 1 if h.old_len > 0 else h.old_start
        if start < pos:
            raise PatchError(f"{fp.path}: overlapping hunks")
        out.extend(old_lines[pos:start])
        pos = start
        no_newline_after: str | None = None
        for hl in h.lines:
            tag, body = hl[:1], hl[1:]
            if tag == "\\":
                # "\ No newline at end of file" applies to the previous line
                if out and no_newline_after == "+":
                    out[-1] = out[-1].rstrip("\n")
                continue
            if tag in (" ", "-"):
                if pos >= len(old_lines) or old_lines[pos].rstrip("\r\n") != body:
                    got = old_lines[pos].rstrip("\n") if pos < len(old_lines) else "<EOF>"
                    raise PatchError(f"{fp.path}: context mismatch at old line {pos + 1}: "
                                     f"expected {body!r}, got {got!r}")
                if tag == " ":
                    out.append(old_lines[pos])
                pos += 1
            elif tag == "+":
                out.append(body + "\n")
            no_newline_after = tag
    out.extend(old_lines[pos:])
    return "".join(out)


def apply_patch(files_before: dict[str, str], patch: str) -> dict[str, str | None]:
    """{path: text} after applying `patch`; deleted files map to None. Only touched files returned."""
    after: dict[str, str | None] = {}
    for fp in parse_patch(patch):
        src = fp.old_path
        old = None if fp.is_new else files_before.get(src or "")
        if not fp.is_new and old is None:
            raise PatchError(f"{src}: file not available")
        after[fp.path] = apply_file_patch(old, fp)
    return after


def make_unified_diff(path: str, old: str, new: str) -> str:
    diff = difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                fromfile=f"a/{path}", tofile=f"b/{path}")
    return f"diff --git a/{path} b/{path}\n" + "".join(diff)


# --------------------------------------------------------------------------- diff -> edits
@dataclass
class EditCall:
    tool: str                              # "edit_file" | "write_file"
    args: dict[str, str]
    added_lines: list[str] = field(default_factory=list)   # lines of new_string not in old_string

    def to_json(self) -> dict:
        return {"tool": self.tool, "args": self.args}


def _change_groups(a: list[str], b: list[str], merge_gap: int = 2) -> list[tuple[int, int, int, int]]:
    ops = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
    groups: list[list[int]] = []
    for _, i1, i2, j1, j2 in ops:
        if groups and i1 - groups[-1][1] <= merge_gap:
            groups[-1][1], groups[-1][3] = i2, j2
        else:
            groups.append([i1, i2, j1, j2])
    return [tuple(g) for g in groups]  # type: ignore[misc]


def file_edit_calls(path: str, old: str, new: str, filepath_prefix: str = "") -> list[EditCall]:
    """edit_file calls turning `old` into `new`, top to bottom, each with a unique old_string."""
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    cur = list(a)
    offset = 0
    calls: list[EditCall] = []
    for i1, i2, j1, j2 in _change_groups(a, b):
        ci1, ci2 = i1 + offset, i2 + offset
        repl = b[j1:j2]
        for k in range(0, len(cur) + 1):
            s, e = max(0, ci1 - k), min(len(cur), ci2 + k)
            old_block = "".join(cur[s:e])
            if not old_block.strip():
                continue
            if "".join(cur).count(old_block) == 1:
                break
        else:
            raise PatchError(f"{path}: cannot find a unique old_string around lines {i1 + 1}-{i2}")
        new_block = "".join(cur[s:ci1] + repl + cur[ci2:e])
        old_s, new_s = old_block, new_block
        if old_s.endswith("\n") and new_s.endswith("\n"):
            old_s, new_s = old_s[:-1], new_s[:-1]
        old_set = {ln.rstrip("\n") for ln in cur[s:e]}
        added = [ln.rstrip("\n") for ln in repl if ln.rstrip("\n") not in old_set]
        calls.append(EditCall("edit_file", {"filepath": filepath_prefix + path,
                                            "old_string": old_s, "new_string": new_s}, added))
        cur[s:e] = cur[s:ci1] + repl + cur[ci2:e]
        offset += (j2 - j1) - (i2 - i1)
    if "".join(cur) != new:
        raise PatchError(f"{path}: internal error, edits do not reproduce the new text")
    return calls


def patch_to_edit_calls(files_before: dict[str, str], patch: str, filepath_prefix: str = "",
                        validate: bool = True) -> list[EditCall]:
    after = apply_patch(files_before, patch)
    calls: list[EditCall] = []
    for path, new in after.items():
        if new is None:
            raise PatchError(f"{path}: file deletion is not supported")
        old = files_before.get(path)
        if old is None or old == "":
            calls.append(EditCall("write_file", {"filepath": filepath_prefix + path, "content": new},
                                  [ln for ln in new.splitlines()]))
        else:
            calls.extend(file_edit_calls(path, old, new, filepath_prefix))
    if validate:
        validate_edit_calls(files_before, calls, after)
    return calls


def validate_edit_calls(files_before: dict[str, str], calls: list[EditCall],
                        expected_after: dict[str, str | None]) -> None:
    """Run the calls with the harness's own tools; require exact matches and the expected result."""
    from ..harness.local_sandbox import LocalWorkspace

    with LocalWorkspace(files_before) as ws:
        for c in calls:
            out = json.loads(ws.call(c.tool, c.args))
            if out.get("status") != "ok":
                raise PatchError(f"harness rejected {c.tool} on {c.args['filepath']}: {out.get('error_message')}")
            if c.tool == "edit_file" and out.get("strategy") != "exact":
                raise PatchError(f"{c.args['filepath']}: edit matched with strategy {out.get('strategy')!r}")
        for path, text in expected_after.items():
            if text is not None and ws.read_text(path) != text:
                raise PatchError(f"{path}: harness result differs from the patched file")
