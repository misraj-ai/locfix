"""Fetch base-commit code files for the solved tasks, straight from `cleaned_v2/`.

    locfix data.fetch-code codeagent_final/cleaned_v2/traces \\
        --solved codeagent_final/cleaned_v2/solved.jsonl --out codeagent_final/cleaned_v2/code

For every task (attempts of a task share the base): the files it needs (`code_files.needed_paths`:
the files of the teacher's patch + every file the teacher read), taken with plain git at
`base_commit` from the task's repository, with the part of `setup_patch` that touches each file
applied (SWE-smith: the injected bug). If those files do not reproduce what the teacher read and the
repository has a branch named after the instance (SWE-smith mirrors keep each task's buggy code
there), the branch is used instead. Written to `<out>/<instance_id>.json` as `{path: text}`, then
checked against the teacher's recorded reads (`code_files.check_code`).

Git: one cache folder per repository; each commit is fetched alone (`--depth 1`,
`--filter=blob:none`: commit and trees only) and file contents are downloaded on demand by
`git show`, so only the needed files cross the network. Repositories run in parallel threads, the
tasks of one repository one after another. Re-running skips tasks whose file exists.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .code_files import check_code, code_file_name, needed_paths
from .traces import load_trace


def repo_url(repo: str, template: str) -> str:
    if repo.startswith(("http://", "https://", "git@", "file://", "/")):
        return repo
    return template.format(repo=repo)


def file_section(patch: str | None, path: str) -> str:
    """The part of a unified diff that touches `path` (as `+++ b/<path>`)."""
    parts = re.split(r"(?m)^(?=diff --git )", patch or "")
    return "".join(p for p in parts if re.search(rf"(?m)^\+\+\+ b/{re.escape(path)}$", p))


def apply_section(text: str, path: str, section: str) -> str | None:
    """`text` with the diff `section` applied, or None if it does not apply."""
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        (Path(tmp) / "s.patch").write_text(section if section.endswith("\n") else section + "\n")
        subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
        r = subprocess.run(["git", "apply", "--recount", "s.patch"], cwd=tmp, capture_output=True)
        return f.read_text() if r.returncode == 0 else None


class GitCache:
    """One partial clone per repository; commits fetched one at a time."""

    def __init__(self, root: str | Path, url_template: str, timeout: int = 600):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.template = url_template
        self.timeout = timeout

    def _dir(self, repo: str) -> Path:
        return self.root / re.sub(r"[^\w.-]+", "__", repo)

    def git(self, repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        # never wait for a password: a missing or private repository fails at once
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true"}
        return subprocess.run(["git", "-C", str(self._dir(repo)), *args], capture_output=True,
                              check=check, timeout=self.timeout, env=env)

    def ensure_commit(self, repo: str, commit: str) -> str | None:
        """None if `commit` is available locally, else the reason it is not."""
        d = self._dir(repo)
        if not (d / ".git").exists():
            d.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-q", str(d)], check=True)
            self.git(repo, "remote", "add", "origin", repo_url(repo, self.template))
            self.git(repo, "config", "extensions.partialClone", "origin")
            self.git(repo, "config", "remote.origin.promisor", "true")
            self.git(repo, "config", "remote.origin.partialclonefilter", "blob:none")
        if self.git(repo, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode == 0:
            return None
        r = self.git(repo, "fetch", "-q", "--no-tags", "--depth", "1", "--filter=blob:none",
                     "origin", commit, check=False)
        if r.returncode:
            msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
            return "fetch failed: " + (msg[-1] if msg else f"exit {r.returncode}")[:200]
        return None

    def branch_commit(self, repo: str, branch: str) -> str | None:
        """Fetch `refs/heads/<branch>` (SWE-smith keeps every task's buggy code on a branch named
        after the instance) and return its commit, or None if there is no such branch."""
        r = self.git(repo, "fetch", "-q", "--no-tags", "--depth", "1", "--filter=blob:none",
                     "origin", f"refs/heads/{branch}", check=False)
        if r.returncode:
            return None
        return self.git(repo, "rev-parse", "FETCH_HEAD").stdout.decode().strip() or None

    def tree(self, repo: str, commit: str) -> set[str]:
        out = self.git(repo, "ls-tree", "-r", "--name-only", commit).stdout.decode("utf-8", "replace")
        return set(out.splitlines())

    def show(self, repo: str, commit: str, path: str) -> str | None:
        r = self.git(repo, "show", f"{commit}:{path}", check=False)
        return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def fetch_task(cache: GitCache, task: dict[str, Any], commit: str | None = None,
               apply_setup: bool = True) -> tuple[dict[str, str] | None, str]:
    repo, commit = task.get("repo"), commit or task.get("base_commit")
    if not repo or not commit:
        return None, "no_repo_or_commit"
    err = cache.ensure_commit(repo, commit)
    if err:
        return None, err
    tree = cache.tree(repo, commit)
    files: dict[str, str] = {}
    setup_failed = 0
    for p in task["paths"]:
        if p not in tree:
            continue
        text = cache.show(repo, commit, p)
        if text is None:
            continue
        sec = file_section(task.get("setup_patch"), p) if apply_setup else ""
        if sec:
            new = apply_section(text, p, sec)
            if new is None:
                setup_failed += 1
            else:
                text = new
        files[p] = text
    if not files:
        return None, "no_needed_file_at_commit"
    return files, "fetched_setup_patch_failed" if setup_failed else "fetched"


def _first_check(task: dict[str, Any], files: dict[str, str]) -> str:
    return check_code(load_trace(task["traces"][0]), files, task["patch"])["status"]


def plan_tasks(traces_dir: str | Path, solved: str | Path, tasks_file: str | None = None,
               limit: int = 0) -> tuple[list[dict[str, Any]], Counter]:
    """One entry per task: repo, base_commit, setup_patch, patch, traces and the paths to fetch."""
    from .build_common import read_tasks

    rows = {}
    with open(solved) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["instance_id"]] = r
    paths = sorted(Path(traces_dir).glob("*.json"))
    if tasks_file:
        keep = read_tasks(tasks_file)
        paths = [p for p in paths if p.name in keep]
    if limit:
        paths = paths[:limit]
    counts: Counter = Counter()
    by_task: dict[str, dict[str, Any]] = {}
    for p in paths:
        try:
            tr = load_trace(p)
        except Exception as e:  # noqa: BLE001 - counted
            counts[f"trace_error:{type(e).__name__}"] += 1
            continue
        row = rows.get(tr.instance_id)
        if row is None:
            counts["skip_not_in_solved"] += 1
            continue
        t = by_task.setdefault(tr.instance_id, {
            "instance_id": tr.instance_id, "repo": row.get("repo"), "base_commit": row.get("base_commit"),
            "setup_patch": row.get("setup_patch"), "verifier": row.get("verifier"),
            "patch": row["teacher"]["patch"], "traces": [], "paths": set()})
        t["traces"].append(str(p))
        t["paths"] |= set(needed_paths(tr, row["teacher"]["patch"]))
    for t in by_task.values():
        t["paths"] = sorted(t["paths"])
    return list(by_task.values()), counts


def add_args_cmd_fetch_code(p: argparse.ArgumentParser) -> None:
    p.add_argument("traces", help="folder of cleaned traces (cleaned_v2/traces)")
    p.add_argument("--solved", required=True, help="solved.jsonl (repo, base_commit, setup_patch, teacher.patch)")
    p.add_argument("--out", required=True, help="code folder: <out>/<instance_id>.json")
    p.add_argument("--cache", default="work/repo_cache", help="git cache folder (local disk)")
    p.add_argument("--url-template", default="https://github.com/{repo}.git",
                   help="repository URL from the `repo` field of solved.jsonl")
    p.add_argument("--workers", type=int, default=8, help="repositories fetched in parallel")
    p.add_argument("--tasks", help="only these trace files (e.g. s1_agent.tasks.txt)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--report", help="report JSON (default <out>/../fetch_code_report.json)")


def cmd_fetch_code(args: argparse.Namespace, cfg: dict) -> int:
    tasks, counts = plan_tasks(args.traces, args.solved, args.tasks, args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = GitCache(args.cache, args.url_template)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_repo[t["repo"] or ""].append(t)
    print(f"{len(tasks)} tasks in {len(by_repo)} repositories; "
          f"verifiers {dict(Counter(t['verifier'] for t in tasks))}; "
          f"with setup_patch {sum(bool(t['setup_patch']) for t in tasks)}", flush=True)
    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    t0 = time.time()

    def one_repo(repo_tasks: list[dict]) -> None:
        for t in sorted(repo_tasks, key=lambda x: x["base_commit"] or ""):
            target = out / code_file_name(t["instance_id"])
            if target.exists():
                files, status = json.loads(target.read_text()), "already_there"
            else:
                try:
                    files, status = fetch_task(cache, t)
                    if files is None or _first_check(t, files) == "differs":
                        # SWE-smith: the task's buggy code is on a branch named after the instance
                        alt = cache.branch_commit(t["repo"], t["instance_id"]) if t.get("repo") else None
                        if alt:
                            f2, s2 = fetch_task(cache, t, commit=alt, apply_setup=False)
                            if f2 is not None and _first_check(t, f2) != "differs":
                                files, status = f2, s2 + "_from_instance_branch"
                except Exception as e:  # noqa: BLE001 - one task must not stop the others
                    files, status = None, f"error:{type(e).__name__}: {str(e)[:150]}"
                if files is not None:
                    tmp = target.with_suffix(".part")
                    tmp.write_text(json.dumps(files, ensure_ascii=False))
                    tmp.replace(target)
            check = None
            if files is not None:
                check = {}
                for tp in t["traces"]:
                    try:
                        check[Path(tp).name] = check_code(load_trace(tp), files, t["patch"])["status"]
                    except Exception as e:  # noqa: BLE001
                        check[Path(tp).name] = f"check_error:{type(e).__name__}"
            with lock:
                results[t["instance_id"]] = {"repo": t["repo"], "status": status, "check": check,
                                             "n_paths": len(t["paths"]),
                                             "n_files": len(files) if files else 0}
                if len(results) % 50 == 0:
                    print(f"  {time.time() - t0:5.0f}s  {len(results)}/{len(tasks)} tasks", flush=True)

    with ThreadPoolExecutor(max(1, args.workers)) as pool:
        list(pool.map(one_repo, by_repo.values()))
    fetch = Counter(r["status"].split(":")[0] if r["status"].startswith(("fetch failed", "error"))
                    else r["status"] for r in results.values())
    checks = Counter(s for r in results.values() if r["check"] for s in r["check"].values())
    report = {"tasks": len(tasks), "repositories": len(by_repo), "trace_counts": dict(counts),
              "fetch": dict(fetch), "check_per_trace": dict(checks),
              "seconds": round(time.time() - t0), "per_task": results}
    rp = Path(args.report) if args.report else out.parent / "fetch_code_report.json"
    rp.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_task"}, indent=2))
    failed = Counter(r["status"] for r in results.values() if not r["status"].startswith(("fetched", "already")))
    if failed:
        print("not fetched, by reason (top 10):")
        for k, v in failed.most_common(10):
            print(f"  {v:4d}  {k}")
    print("report:", rp)
    return 0
