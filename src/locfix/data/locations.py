"""Code locations: AST symbol index, L_edit (from a patch) and L_view (from a trajectory).

A location is (path, symbol, start, end). `symbol` is a qualified name (`Class.method`,
`function`) or `<module>` for top-level code outside any definition, or `<file>` for non-Python
files. Lines are 1-based and inclusive; decorator lines belong to their definition.
"""
from __future__ import annotations

import ast
import re
import shlex
from dataclasses import asdict, dataclass

from .edits import parse_patch
from .traces import ToolCall, Trace


@dataclass(frozen=True)
class Symbol:
    path: str
    name: str
    start: int
    end: int
    depth: int                             # 0 = top level

    def overlaps(self, a: int, b: int) -> bool:
        return self.start <= b and a <= self.end


@dataclass(frozen=True)
class Location:
    path: str
    symbol: str
    start: int
    end: int
    role: str = "context"                  # "edit" | "context"

    def to_json(self) -> dict:
        d = asdict(self)
        d["lines"] = [d.pop("start"), d.pop("end")]
        return d


def normalize_path(path: str) -> str:
    p = path.strip()
    if p.startswith("/workspace/"):
        p = p[len("/workspace/"):]
    elif p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


class SymbolIndex:
    """Definitions of one file (Python via `ast`; other files have only `<file>`).

    `known`: for code rebuilt from a trace (`code_from_trace`), the line numbers whose text is
    known. Such files usually do not parse; their definitions are then found by scanning the known
    lines (`def`/`class` headers and indentation), and each span stops at the end of its known block.
    """

    def __init__(self, path: str, text: str, known: set[int] | None = None):
        self.path = normalize_path(path)
        self.text = text
        self.n_lines = len(text.splitlines())
        self.symbols: list[Symbol] = []
        self.parsed = False
        if self.path.endswith((".py", ".pyi")):
            try:
                tree = ast.parse(text)
                self.parsed = True
            except SyntaxError:
                tree = None
            if tree is not None:
                self._walk(tree.body, prefix="", depth=0)
            elif known:
                self._scan_partial(known)
                self.parsed = True
        self.symbols.sort(key=lambda s: (s.start, -s.end))

    _HEADER = re.compile(r"^(\s*)(?:async\s+def|def|class)\s+(\w+)")

    def _scan_partial(self, known: set[int]) -> None:
        lines = self.text.splitlines()

        def indent(s: str) -> int:
            return len(s) - len(s.lstrip())

        stack: list[tuple[int, str, int]] = []          # (indent, qualified name, start)
        opened: list[tuple[int, str, int, int]] = []     # (start, name, indent, depth)
        for n in sorted(k for k in known if 1 <= k <= len(lines)):
            text = lines[n - 1]
            m = self._HEADER.match(text)
            if m:
                ind = indent(text)
                while stack and stack[-1][0] >= ind:
                    stack.pop()
                name = (stack[-1][1] + "." if stack else "") + m.group(2)
                stack.append((ind, name, n))
                opened.append((n, name, ind, len(stack) - 1))
        for start, name, ind, depth in opened:
            end = start
            n = start + 1
            while n in known and n <= len(lines):
                t = lines[n - 1]
                if t.strip() and indent(t) <= ind and not t.lstrip().startswith((")", "]", "}")):
                    break
                end = n
                n += 1
            while end > start and not lines[end - 1].strip():
                end -= 1
            first = start                                 # decorators belong to the definition
            while first - 1 in known and lines[first - 2].strip().startswith("@") \
                    and indent(lines[first - 2]) == ind:
                first -= 1
            self.symbols.append(Symbol(self.path, name, first, end, depth))

    def _walk(self, body, prefix: str, depth: int) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                name = f"{prefix}{node.name}"
                self.symbols.append(Symbol(self.path, name, start, node.end_lineno or node.lineno, depth))
                self._walk(node.body, prefix=name + ".", depth=depth + 1)

    def innermost(self, line: int) -> Symbol | None:
        best = None
        for s in self.symbols:
            if s.start <= line <= s.end and (best is None or s.depth > best.depth):
                best = s
        return best

    def location_for_lines(self, lines: list[int], role: str) -> list[Location]:
        """Innermost symbol per line; lines outside definitions grouped as `<module>` spans."""
        if not self.parsed:
            if not lines:
                return []
            return [Location(self.path, "<file>", min(lines), max(lines), role)]
        out: dict[tuple, Location] = {}
        module_lines = []
        for ln in lines:
            s = self.innermost(ln)
            if s is None:
                module_lines.append(ln)
            else:
                out[(s.name, s.start)] = Location(self.path, s.name, s.start, s.end, role)
        for a, b in _runs(sorted(set(module_lines))):
            out[("<module>", a)] = Location(self.path, "<module>", a, b, role)
        return sorted(out.values(), key=lambda loc: (loc.start, loc.end))

    def symbols_overlapping(self, a: int, b: int, role: str = "context") -> list[Location]:
        """Symbols a viewed range [a, b] showed: the innermost symbols overlapping it (a class is
        replaced by the methods of it that overlap), plus `<module>` spans for viewed lines outside
        any definition. A range covering the whole file is a single `<file>` entry."""
        if not self.parsed:
            return [Location(self.path, "<file>", a, b, role)]
        if a <= 1 and b >= self.n_lines:
            return [Location(self.path, "<file>", 1, self.n_lines, role)]
        over = [s for s in self.symbols if s.overlaps(a, b)]
        leaves = [s for s in over if not any(o is not s and o.depth > s.depth and s.start <= o.start
                                             and o.end <= s.end for o in over)]
        out = [Location(self.path, s.name, s.start, s.end, role) for s in leaves]
        covered = set()
        for s in over:
            covered.update(range(max(a, s.start), min(b, s.end) + 1))
        lines = self.text.splitlines()
        free = [n for n in range(a, b + 1) if n not in covered and n - 1 < len(lines)]
        for x, y in _runs(free):
            while x <= y and not lines[x - 1].strip():
                x += 1
            while y >= x and not lines[y - 1].strip():
                y -= 1
            if x <= y:
                out.append(Location(self.path, "<module>", x, y, role))
        return sorted(out, key=lambda loc: (loc.start, loc.end))


def _runs(nums: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for n in nums:
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


# --------------------------------------------------------------------------- L_edit
def edit_locations(files_before: dict[str, str], patch: str,
                   known: dict[str, set[int]] | None = None) -> list[Location]:
    """Symbols touched by `patch` (old-file line numbers). New files are file-level entries."""
    out: list[Location] = []
    for fp in parse_patch(patch):
        if fp.is_new:
            out.append(Location(normalize_path(fp.path), "<file>", 1, 1, "edit"))
            continue
        text = files_before.get(fp.old_path or "")
        if text is None:
            raise KeyError(f"code for {fp.old_path} not available")
        idx = SymbolIndex(fp.old_path or "", text, (known or {}).get(normalize_path(fp.old_path or "")))
        out.extend(idx.location_for_lines(fp.changed_old_lines(), role="edit"))
    return out


# --------------------------------------------------------------------------- L_view
@dataclass(frozen=True)
class View:
    path: str
    start: int
    end: int | None                        # None = to end of file
    call_entry: int
    via: str                               # read_file | sed | cat | head | tail


_SED_RANGE = re.compile(r"^(\d+)(?:,(\d+))?p$")


def views_of_call(call: ToolCall) -> list[View]:
    """File ranges a single tool call showed to the model."""
    if call.tool == "read_file":
        res = call.result_json()
        if res.get("status") != "ok":
            return []
        return [View(normalize_path(call.args["filepath"]), int(res["start_line"]), int(res["end_line"]),
                     call.entry_index, "read_file")]
    if call.tool != "run_command":
        return []
    cmd = call.args.get("command", "")
    views: list[View] = []
    for seg in re.split(r"&&|;|\|\||\n", cmd):
        seg = seg.split("|")[0].strip()          # only the producer of a pipeline shows a file
        try:
            toks = shlex.split(seg)
        except ValueError:
            continue
        if toks[:1] == ["cd"]:
            continue
        if not toks:
            continue
        prog, args = toks[0], toks[1:]
        files = [a for a in args
                 if not a.startswith("-") and not _SED_RANGE.match(a) and not re.fullmatch(r"\d+", a)
                 and ("/" in a or "." in a) and not a.startswith(("'", '"'))]
        if prog == "sed" and "-n" in args:
            rng = next((a for a in args if _SED_RANGE.match(a)), None)
            if rng:
                m = _SED_RANGE.match(rng)
                a_, b_ = int(m.group(1)), int(m.group(2) or m.group(1))
                shown = _shown_lines(call)
                if shown is not None and len(files) == 1:
                    b_ = min(b_, a_ + shown - 1)        # output truncated at 5,000 characters
                for f in files:
                    views.append(View(normalize_path(f), a_, b_, call.entry_index, "sed"))
        elif prog == "cat" and files:
            for f in files:
                views.append(View(normalize_path(f), 1, None, call.entry_index, "cat"))
        elif prog in ("head", "tail") and files:
            n = 10
            for i, a in enumerate(args):
                if a == "-n" and i + 1 < len(args) and args[i + 1].lstrip("+").isdigit():
                    n = int(args[i + 1].lstrip("+"))
                elif re.fullmatch(r"-\d+", a):
                    n = int(a[1:])
            for f in files:
                views.append(View(normalize_path(f), 1 if prog == "head" else -n,
                                  n if prog == "head" else None, call.entry_index, prog))
    return views


def _shown_lines(call: ToolCall) -> int | None:
    """Number of complete lines in a truncated run_command stdout (None if not truncated)."""
    res = call.result_json()
    stdout = res.get("stdout")
    if stdout is None and isinstance(res.get("details"), dict):
        stdout = res["details"].get("stdout")
    if not isinstance(stdout, str) or len(stdout) < 5000:
        return None
    return stdout.count("\n")


def view_ranges(trace: Trace, before_first_edit: bool = True) -> list[View]:
    calls = trace.calls_before_first_edit() if before_first_edit else list(trace.tool_calls())
    out: list[View] = []
    for c in calls:
        out.extend(views_of_call(c))
    return out


def view_locations(trace: Trace, files: dict[str, str],
                   known: dict[str, set[int]] | None = None) -> list[Location]:
    """L_view: symbols the teacher viewed before its first edit, in first-view order.
    Files without available code are skipped (reported by the caller)."""
    seen: dict[tuple, Location] = {}
    for v in view_ranges(trace):
        text = files.get(v.path)
        if text is None:
            continue
        idx = SymbolIndex(v.path, text, (known or {}).get(v.path))
        start = v.start if v.start > 0 else max(1, idx.n_lines + v.start + 1)
        end = idx.n_lines if v.end is None else min(v.end, idx.n_lines)
        for loc in idx.symbols_overlapping(start, end):
            seen.setdefault((loc.path, loc.symbol, loc.start), loc)
    return list(seen.values())
