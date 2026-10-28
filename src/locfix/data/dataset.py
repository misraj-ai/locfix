"""The solved training set on disk (`codeagent_final/cleaned/`), task by task.

    data = SolvedData("codeagent_final")                   # or explicit paths
    tr_path = data.sample(random.Random(0))
    task, info = data.task(tr_path)                         # Task for loc_fix_records

Layout (DATA_PIPELINE.md §3): `cleaned/solved.jsonl`, `cleaned/traces/<id>__a<N>.json`, and
base-commit code in `code/<id>.json` (searched in `cleaned/code`, `experiment/code`, or the
folders given). Without a code file, the code is rebuilt from what the teacher printed
(`code_from_trace`); synthetic reads are then limited to lines the teacher saw.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..harness.prompts import parse_task_prompt
from .code_files import check_code, code_file_name
from .code_from_trace import seen_code
from .loc_fix_records import Task, TaskSkip
from .traces import load_trace, parse_trace_name


@dataclass
class CodeInfo:
    source: str                            # "code_json" | "code_json+trace" | "trace" | "trace:code_json_differs"
    path: str | None = None
    coverage: dict[str, float] | None = None


class SolvedData:
    def __init__(self, root: str | Path | None = None, *, traces: str | Path | None = None,
                 solved: str | Path | None = None, code_dirs: list[str | Path] | None = None,
                 exclude_repos: list[str] | None = None):
        r = Path(root) if root else None
        self.traces_dir = Path(traces) if traces else (r / "cleaned" / "traces")
        self.solved_path = Path(solved) if solved else (r / "cleaned" / "solved.jsonl")
        if code_dirs is None:
            code_dirs = [r / "cleaned" / "code", r / "experiment" / "code"] if r else []
        self.code_dirs = [Path(c) for c in code_dirs]
        self.exclude = [x.lower() for x in (exclude_repos or [])]
        self.rows: dict[str, dict[str, Any]] = {}
        with open(self.solved_path) as f:
            for line in f:
                row = json.loads(line)
                self.rows[row["instance_id"]] = row

    def trace_paths(self) -> list[Path]:
        return sorted(self.traces_dir.glob("*.json"))

    def sample(self, rng: random.Random, k: int = 1) -> list[Path]:
        paths = self.trace_paths()
        return rng.sample(paths, min(k, len(paths)))

    def code_file(self, instance_id: str) -> Path | None:
        names = dict.fromkeys([f"{instance_id}.json", code_file_name(instance_id)])
        for d in self.code_dirs:
            for n in names:
                p = d / n
                if p.exists():
                    return p
        return None

    def row(self, instance_id: str) -> dict[str, Any]:
        row = self.rows.get(instance_id)
        if row is None:
            raise TaskSkip("not_in_solved", instance_id)
        return row

    def source(self, row: dict[str, Any], attempt: int | None) -> dict[str, Any]:
        return {"set": "solved", "dataset": row.get("verifier", "other"), "fold": row.get("fold", "A"),
                "attempt": attempt or 1}

    def task(self, trace_path: str | Path, code_fallback: bool = True) -> tuple[Task, CodeInfo]:
        tr = load_trace(trace_path)
        iid, _ = parse_trace_name(trace_path)
        iid = tr.instance_id or iid
        repo = parse_task_prompt(tr.task_prompt).repo.lower()
        if any(repo.split("/")[-1] == x for x in self.exclude):
            raise TaskSkip("dev_repo", repo)
        row = self.row(iid)
        cf = self.code_file(iid)
        files = known = info = None
        if cf is not None:
            files = json.loads(cf.read_text())
            chk = check_code(tr, files, row["teacher"]["patch"])
            if chk["status"] in ("ok", "unverified"):
                known, info = None, CodeInfo("code_json", str(cf))
            elif chk["status"] in ("missing_read_file", "missing_edited_file"):
                # files the teacher printed but the code file lacks: take them from the trace
                # (synthetic reads of those are limited to the lines it saw)
                seen = seen_code(tr)
                extra = {p: t for p, t in seen.files().items() if p not in files}
                known = {p: set(range(1, len(t.splitlines()) + 1)) for p, t in files.items()}
                known.update({p: k for p, k in seen.known().items() if p in extra})
                files = {**files, **extra}
                info = CodeInfo("code_json+trace", str(cf))
            else:
                # the code file does not reproduce what the teacher saw: do not mix the two
                files, info = None, CodeInfo("trace:code_json_differs", str(cf))
        if files is None:
            if not code_fallback:
                raise TaskSkip("no_code_file" if cf is None else "code_json_differs", iid)
            seen = seen_code(tr)
            files, known = seen.files(), seen.known()
            info = CodeInfo(info.source if info else "trace", info.path if info else None, seen.coverage())
        task = Task(tr, files, row["teacher"]["patch"], row["problem_statement"],
                    source=self.source(row, tr.attempt), known_lines=known)
        return task, info
