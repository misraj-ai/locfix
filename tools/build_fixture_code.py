"""Build tests/fixtures/code/<instance_id>.json for the four fixture traces (developer utility).

The data pipeline's `code/<id>.json` holds {path: text} of the gold and viewed files at the base
commit with the SWE-smith bug applied. For the fixtures we rebuild the same thing from public
sources and the teacher's own observations:

  1. the file at the upstream commit (`git show <commit>:<path>`, repos cloned with
     `--filter=blob:none` into --repos);
  2. every line the teacher saw before its first edit (`read_file` contents, plain `sed -n` output)
     overrides the upstream text; unseen runs are filled from upstream, aligned on the seen lines
     (SWE-smith bugs rewrite a function, e.g. drop its docstring, so line numbers shift);
  3. validation: replaying every recorded `read_file`, `edit_file` and plain `sed -n` call of the
     whole trajectory on the rebuilt files with the harness's own tools must reproduce the recorded
     outputs exactly (edit diffs carry line numbers, so this also checks the unseen parts' length).

    python tools/build_fixture_code.py --repos /path/to/clones
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from locfix.data.locations import normalize_path
from locfix.data.traces import ToolCall, load_trace
from locfix.harness.local_sandbox import LocalWorkspace

FIXTURES = {
    # trace prefix: (repo dir, commit, files)
    "agronholm__exceptiongroup": ("exceptiongroup", "0b4f4937",
                                  ["src/exceptiongroup/_formatting.py", "tests/test_formatting.py"]),
    "arrow-py__arrow": ("arrow", "1d70d009", ["arrow/factory.py", "arrow/arrow.py"]),
    "astropy__astropy-13803": ("astropy", None, ["astropy/coordinates/angles.py"]),
    "benoitc__gunicorn": ("gunicorn", "bacbf8aa",
                          ["gunicorn/reloader.py", "tests/test_reload.py", "gunicorn/workers/base.py"]),
}

_SED = re.compile(r"^(?:cd /workspace\s*&&\s*)?sed -n '?(\d+),(\d+)p'? (\S+)$")


def git_show(repo: Path, commit: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], check=True,
                          capture_output=True, text=True).stdout


def plain_sed(call: ToolCall) -> tuple[str, int, int, str] | None:
    """(path, a, b, stdout) for a call that is exactly one `sed -n 'a,bp' file`."""
    if call.tool != "run_command":
        return None
    m = _SED.match(call.args.get("command", "").strip())
    if not m:
        return None
    res = call.result_json()
    return normalize_path(m.group(3)), int(m.group(1)), int(m.group(2)), res.get("stdout", "")


def observations(trace) -> tuple[dict[str, dict[int, str]], dict[str, int]]:
    """Lines of the base state the teacher saw before its first edit, and file lengths."""
    seen: dict[str, dict[int, str]] = {}
    total: dict[str, int] = {}
    for c in trace.calls_before_first_edit():
        if c.tool == "read_file":
            r = c.result_json()
            if r.get("status") != "ok":
                continue
            p = normalize_path(c.args["filepath"])
            total[p] = int(r["total_lines"])
            lines = r["content"].split("\n")
            for i, text in enumerate(lines[: int(r["end_line"]) - int(r["start_line"]) + 1]):
                seen.setdefault(p, {})[int(r["start_line"]) + i] = text
        sed = plain_sed(c)
        if sed:
            p, a, _b, out = sed
            lines = out.split("\n")
            complete = lines[:-1]                       # last element: after final newline / cut line
            for i, text in enumerate(complete):
                seen.setdefault(p, {})[a + i] = text
    return seen, total


def splice(upstream: str, seen: dict[int, str], n_total: int | None) -> str:
    up = upstream.split("\n")
    trailing_nl = upstream.endswith("\n")
    if trailing_nl:
        up = up[:-1]
    n = n_total if n_total is not None else len(up)
    out: list[str | None] = [seen.get(i) for i in range(1, n + 1)]
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < n and out[j] is None:
            j += 1
        # unseen run [i, j); choose the upstream offset from the nearest seen anchor
        if i == 0:
            off = 0 if j == n else _offset_after(up, out, j)
        else:
            off = _offset_before(up, out, i - 1)
        for k in range(i, j):
            if not 0 <= k + off < len(up):
                raise ValueError(f"cannot fill line {k + 1}: upstream too short (offset {off})")
            out[k] = up[k + off]
        i = j
    return "\n".join(out) + ("\n" if trailing_nl else "")  # type: ignore[arg-type]


def _offset_before(up: list[str], out: list[str | None], anchor: int) -> int:
    """Offset d with up[anchor + d] == out[anchor], nearest to the length difference."""
    target = out[anchor]
    cands = [k - anchor for k, t in enumerate(up) if t == target]
    if not cands:
        raise ValueError(f"anchor line {anchor + 1} not found upstream: {target!r}")
    tail_off = len(up) - len(out)
    return min(cands, key=lambda d: abs(d - tail_off))


def _offset_after(up: list[str], out: list[str | None], anchor: int) -> int:
    target = out[anchor]
    cands = [k - anchor for k, t in enumerate(up) if t == target]
    if not cands:
        raise ValueError(f"anchor line {anchor + 1} not found upstream: {target!r}")
    return min(cands, key=abs)


def validate(trace, files: dict[str, str]) -> dict[str, int]:
    """Replay recorded read_file / edit_file / plain sed calls on the rebuilt files."""
    counts = {"read_file": 0, "edit_file": 0, "sed": 0}
    with LocalWorkspace(files) as ws:
        for c in trace.tool_calls():
            if c.tool in ("read_file", "edit_file"):
                p = normalize_path(c.args["filepath"])
                if p not in files:
                    continue
                got = ws.call(c.tool, c.args)
                if got != c.result:
                    raise AssertionError(f"{trace.instance_id}: entry {c.entry_index} {c.tool} differs\n"
                                         f"got:      {got[:600]}\nrecorded: {c.result[:600]}")
                counts[c.tool] += 1
            sed = plain_sed(c)
            if sed and sed[0] in files:
                p, a, b, out = sed
                lines = ws.read_text(p).split("\n")
                want = "".join(x + "\n" for x in lines[a - 1:b])
                if len(out) >= 5000:                    # truncated stdout: compare the shown prefix
                    want = want[: len(out)]
                if not want.startswith(out[: len(want)]) or (len(out) < 5000 and want != out):
                    raise AssertionError(f"{trace.instance_id}: entry {c.entry_index} sed output differs")
                counts["sed"] += 1
    return counts


def find_commit(repo: Path, path: str, trace) -> str:
    """For SWE-bench tasks without a commit in the trace: the most recent version of `path` that
    contains every edit's old_string and every recorded line the teacher printed with grep -n."""
    revs = subprocess.run(["git", "-C", str(repo), "log", "--all", "--format=%H", "--since=2021-01-01",
                           "--until=2023-03-01", "--", path], check=True, capture_output=True,
                          text=True).stdout.split()
    olds = [c.args["old_string"] for c in trace.tool_calls() if c.tool == "edit_file"]
    for rev in revs:
        for cand in (rev, rev + "^"):
            text = git_show(repo, cand, path)
            if not all(o in text for o in olds):
                continue
            try:
                validate(trace, {path: text})
            except AssertionError:
                continue
            return cand
    raise SystemExit(f"no commit of {path} matches the trace")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repos", required=True, help="folder with blob-less clones of the four repos")
    ap.add_argument("--out", default=str(ROOT / "tests/fixtures/code"))
    args = ap.parse_args()
    repos, out = Path(args.repos), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for prefix, (repo_dir, commit, paths) in FIXTURES.items():
        (trace_path,) = sorted((ROOT / "tests/fixtures/traces").glob(prefix + "*.json"))
        trace = load_trace(trace_path)
        repo = repos / repo_dir
        if commit is None:
            commit = find_commit(repo, paths[0], trace)
        seen, total = observations(trace)
        files = {p: splice(git_show(repo, commit, p), seen.get(p, {}), total.get(p)) for p in paths}
        counts = validate(trace, files)
        dest = out / f"{trace.instance_id}.json"
        dest.write_text(json.dumps(files, indent=0, ensure_ascii=False) + "\n")
        meta = {p: (len(t.splitlines()), len(seen.get(p, {}))) for p, t in files.items()}
        print(f"{trace.instance_id}: commit {commit[:12]}, files (lines, seen) {meta}, verified {counts}")


if __name__ == "__main__":
    main()
