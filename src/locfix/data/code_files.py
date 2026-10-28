"""Base-commit code files (`code/<instance_id>.json`) for the solved tasks.

The two-agent records need the full files the fixer may read: every file the teacher's patch
touches and every file the teacher looked at. Without them the code is rebuilt from what the
teacher printed (`code_from_trace`), and tasks whose synthetic reads need unseen lines are lost.

    locfix data.code-paths   traces -> code_paths.jsonl: the files each task needs
                             (fetched at the base commit by notebooks/fetch_solved_code.ipynb)
    locfix data.check-code   do the fetched files reproduce the teacher's recorded reads?

`code/<id>.json` = `{path: text}` (DATA_PIPELINE.md §3). The record builder checks every code
file against the trace (`check_code`) and falls back to the trace's own code when they disagree.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .code_from_trace import _CAT, _SED, seen_code
from .edits import PatchError, parse_patch
from .locations import normalize_path
from .traces import Trace, load_trace

# paths in common read-only shell commands: sed -n / cat / head / tail / nl / grep -n PATTERN FILE
_FILE_ARG = re.compile(r"(?:^|\s)((?:/workspace/|\./)?[\w./-]+\.(?:py|pyi|pyx|cfg|toml|ini|txt|rst|md|json|yaml|yml|c|h|cpp|js|ts))(?=\s|$|['\"]|:)")


def patch_paths(patch: str) -> list[str]:
    """Files of the patch that exist at the base commit (not files the patch creates)."""
    try:
        return sorted({fp.old_path for fp in parse_patch(patch or "") if fp.old_path})
    except PatchError:
        return sorted(set(re.findall(r"(?m)^--- a/(\S+)", patch or "")))


def trace_paths(trace: Trace) -> list[str]:
    """Files the teacher read anywhere in the trajectory: `read_file` paths, `sed -n`/`cat` files
    and file-like arguments of other shell commands (a generous list; missing paths are skipped
    when fetching)."""
    out: set[str] = set()
    for c in trace.tool_calls():
        if c.tool == "read_file" and c.args.get("filepath"):
            out.add(normalize_path(c.args["filepath"]))
        elif c.tool in ("edit_file", "write_file") and c.args.get("filepath"):
            out.add(normalize_path(c.args["filepath"]))
        elif c.tool == "run_command":
            cmd = (c.args.get("command") or "").strip()
            for rx in (_SED, _CAT):
                m = rx.match(cmd)
                if m:
                    out.add(normalize_path(m.group(m.lastindex)))
            for m in _FILE_ARG.finditer(cmd):
                out.add(normalize_path(m.group(1)))
    out |= set(seen_code(trace).total)
    return sorted(p for p in out if p and not p.startswith(("tmp/", "dev/", "proc/")) and ".." not in p)


def needed_paths(trace: Trace, patch: str) -> list[str]:
    return sorted(set(patch_paths(patch)) | set(trace_paths(trace)))


def code_file_name(instance_id: str) -> str:
    return instance_id.replace("/", "__") + ".json"


def _harness_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("swegemma") is not None


def check_code(trace: Trace, files: dict[str, str], patch: str, exact: bool | None = None) -> dict[str, Any]:
    """Do `files` hold what the teacher saw before its first edit, and every file the patch edits?

    - line check (always, no harness needed): every complete line the teacher printed
      (`read_file` content, plain `sed -n` / `cat` output) must equal the file's line, and the
      `total_lines` it was told must equal the file's length;
    - exact check (`exact=True`, or `None` = when the harness wheels are installed): every recorded
      `read_file` response is reproduced byte for byte by the harness's own `read_file`."""
    seen = seen_code(trace)
    missing_edit = [p for p in patch_paths(patch) if p not in files]
    differs: set[str] = set()
    missing: set[str] = set()
    lines_checked = 0
    for p, known in seen.lines.items():
        if p not in files:
            missing.add(p)
            continue
        flines = files[p].splitlines()
        total = seen.total.get(p)
        if total is not None and total != len(flines):
            differs.add(p)
            continue
        for n, text in known.items():
            lines_checked += 1
            if n > len(flines) or flines[n - 1].rstrip("\r") != text.rstrip("\r"):
                differs.add(p)
                break
    reads_ok = None
    if exact or (exact is None and _harness_available()):
        from .render_obs import check_recorded_reads

        reads_ok, bad = check_recorded_reads(trace, files, before_first_edit=True)
        differs |= {b["path"] for b in bad if b["reason"] == "differs"}
        missing |= {b["path"] for b in bad if b["reason"] == "missing_file"}
    status = ("differs" if differs else "missing_edited_file" if missing_edit
              else "missing_read_file" if missing else "ok" if lines_checked or reads_ok
              else "unverified")              # the teacher printed nothing of these files to compare
    return {"status": status, "lines_checked": lines_checked, "reads_ok": reads_ok,
            "differs": sorted(differs), "missing_read": sorted(missing), "missing_edit": missing_edit}


# --------------------------------------------------------------------------- CLI
def _solved_rows(path: str) -> dict[str, dict]:
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["instance_id"]] = r
    return rows


def _traces(args) -> list[Path]:
    from .build_common import read_tasks

    paths = sorted(Path(args.traces).glob("*.json"))
    if getattr(args, "tasks", None):
        keep = read_tasks(args.tasks)
        paths = [p for p in paths if p.name in keep]
    return paths[: args.limit] if args.limit else paths


def add_args_cmd_code_paths(p: argparse.ArgumentParser) -> None:
    p.add_argument("traces", help="folder of cleaned traces")
    p.add_argument("--solved", required=True, help="solved.jsonl (teacher.patch, repo, base_commit)")
    p.add_argument("--out", required=True, help="output JSONL: one row per task")
    p.add_argument("--tasks", help="only these trace files (e.g. s1_agent.tasks.txt)")
    p.add_argument("--limit", type=int, default=0)


def cmd_code_paths(args: argparse.Namespace, cfg: dict) -> int:
    """One row per task: instance_id, repo, base_commit, setup_patch (if any), and the files to
    fetch at the base commit. Attempts of one task share the base: their paths are merged."""
    rows = _solved_rows(args.solved)
    by_task: dict[str, dict[str, Any]] = {}
    counts: Counter = Counter()
    for p in _traces(args):
        try:
            tr = load_trace(p)
        except Exception as e:  # noqa: BLE001 - counted and reported
            counts[f"error:{type(e).__name__}"] += 1
            continue
        row = rows.get(tr.instance_id)
        if row is None:
            counts["skip_not_in_solved"] += 1
            continue
        t = by_task.setdefault(tr.instance_id, {
            "instance_id": tr.instance_id, "repo": row.get("repo"), "base_commit": row.get("base_commit"),
            "setup_patch": row.get("setup_patch"), "verifier": row.get("verifier"),
            "code_file": code_file_name(tr.instance_id), "traces": [], "paths": []})
        t["traces"].append(p.name)
        t["paths"] = sorted(set(t["paths"]) | set(needed_paths(tr, row["teacher"]["patch"])))
        counts["traces"] += 1
    missing_repo = [t["instance_id"] for t in by_task.values() if not t["repo"] or not t["base_commit"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for t in by_task.values():
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    n_paths = sorted(len(t["paths"]) for t in by_task.values())
    rep = {"tasks": len(by_task), **counts, "without_repo_or_commit": len(missing_repo),
           "with_setup_patch": sum(bool(t["setup_patch"]) for t in by_task.values()),
           "paths_per_task": {"median": n_paths[len(n_paths) // 2] if n_paths else 0,
                              "max": n_paths[-1] if n_paths else 0}}
    print(json.dumps(rep, indent=2))
    if missing_repo:
        print("no repo/base_commit in solved.jsonl for:", missing_repo[:10])
    return 0


def add_args_cmd_check_code(p: argparse.ArgumentParser) -> None:
    p.add_argument("traces", help="folder of cleaned traces")
    p.add_argument("--solved", required=True)
    p.add_argument("--code", action="append", default=[], help="code folders (repeatable, searched in order)")
    p.add_argument("--out", required=True, help="report JSON (counts + one row per trace)")
    p.add_argument("--tasks", help="only these trace files")
    p.add_argument("--limit", type=int, default=0)


def cmd_check_code(args: argparse.Namespace, cfg: dict) -> int:
    """For every trace: is there a code file, and does it reproduce the teacher's recorded reads?
    `differs` means the file is not what the teacher saw (wrong commit, bug not applied, files
    changed by a command before the first edit); the record builder then uses the trace's code."""
    rows = _solved_rows(args.solved)
    dirs = [Path(d) for d in args.code]
    per: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for p in _traces(args):
        try:
            tr = load_trace(p)
        except Exception as e:  # noqa: BLE001
            counts[f"error:{type(e).__name__}"] += 1
            continue
        row = rows.get(tr.instance_id)
        if row is None:
            counts["skip_not_in_solved"] += 1
            continue
        cf = next((d / code_file_name(tr.instance_id) for d in dirs
                   if (d / code_file_name(tr.instance_id)).exists()), None)
        if cf is None:
            r = {"status": "no_code_file"}
        else:
            r = check_code(tr, json.loads(cf.read_text()), row["teacher"]["patch"])
        counts[r["status"]] += 1
        per.append({"trace": p.name, **r})
    rep = {"counts": dict(counts), "traces": per}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(dict(counts), indent=2))
    return 0
