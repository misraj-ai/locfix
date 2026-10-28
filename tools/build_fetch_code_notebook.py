"""Write notebooks/fetch_solved_code.ipynb: base-commit code files for the solved tasks (Colab).

    python tools/build_fetch_code_notebook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb_common import GET_CODE_CELL, write_notebook  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# Code files for the solved tasks (Colab, CPU runtime)

Works directly on `codeagent_final/cleaned_v2/` (traces + `solved.jsonl`); nothing else from the
data pipeline is needed. The two-agent records need the full files at the base commit
(`code/<instance_id>.json`); the solved set has none, so ~600 of 1,384 tasks were lost
(`skip:missing_code`, `drop:*:unseen_code`). The file *text* is not in the traces, so it is
downloaded from the task's GitHub repository, but only the needed files, with plain git:

```
cleaned_v2/traces + solved.jsonl
   │  locfix data.fetch-code
   │    1. files each task needs: the teacher's patch files + every file it read
   │    2. git: fetch base_commit (commit + trees only), download just those files,
   │       apply setup_patch (SWE-smith bug) if present; else the SWE-smith instance branch
   │    3. check: the files must reproduce the teacher's recorded read_file outputs
   ▼
cleaned_v2/code/<instance_id>.json  →  build_records.ipynb with REBUILD_STEPS = ["locfix"]
```

Re-running skips tasks already fetched. The record builder checks every code file against the
trace again: a file that does not reproduce what the teacher saw is not used.
""")

md("## 0. Setup")
code(GET_CODE_CELL)
code('''
# --- Paths
DATA_DIR = "/content/drive/MyDrive/codeagent_final"
TRACES = f"{DATA_DIR}/cleaned_v2/traces"
SOLVED = f"{DATA_DIR}/cleaned_v2/solved.jsonl"
CODE_OUT = f"{DATA_DIR}/cleaned_v2/code"                # build_records.ipynb looks here first
TASKS_FILE = None        # e.g. ".../locfix_records/b36_t8_32k/records/s1_agent.tasks.txt" (None = all traces)
REPO_CACHE = "/content/repo_cache"                       # git data (local disk; only commits + needed files)
WORKERS = 8                                              # repositories fetched in parallel
LIMIT = 0                                                # > 0: first N traces only (a quick try)

import os, sys, json, pathlib, subprocess
from collections import Counter
from pathlib import Path

if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")
for p in (TRACES, SOLVED):
    print(p, "found" if Path(p).exists() else "NOT FOUND")
Path(CODE_OUT).mkdir(parents=True, exist_ok=True)
print("code files already in", CODE_OUT, ":", len(list(Path(CODE_OUT).glob("*.json"))))

# what solved.jsonl offers for fetching
rows = [json.loads(l) for l in open(SOLVED) if l.strip()]
print(len(rows), "rows | with repo + base_commit:", sum(bool(r.get("repo") and r.get("base_commit")) for r in rows),
      "| with setup_patch:", Counter(r.get("verifier") for r in rows if r.get("setup_patch")))
print("repo field examples:", Counter(r.get("verifier") for r in rows),
      {v: next(r.get("repo") for r in rows if r.get("verifier") == v) for v in {r.get("verifier") for r in rows}})
''')

md("""
## 1. Fetch and check

Progress every 50 tasks. Roughly 1–3 s per task (one commit and a few files each) with 8
repositories in parallel: about 10–20 minutes for ~1,400 tasks.
""")
code('''
cmd = [sys.executable, "-m", "locfix.cli", "data.fetch-code", TRACES, "--solved", SOLVED,
       "--out", CODE_OUT, "--cache", REPO_CACHE, "--workers", str(WORKERS)]
if TASKS_FILE:
    cmd += ["--tasks", TASKS_FILE]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]
env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="")
assert proc.wait() == 0, "data.fetch-code failed (output above)"
''')

md("""
## 2. Report

- `fetch`: `fetched` (base commit, setup_patch applied if any), `fetched_from_instance_branch`
  (SWE-smith branch), `already_there`, `fetch failed` (repository or commit not reachable),
  `no_repo_or_commit`, `no_needed_file_at_commit`.
- `check_per_trace`: `ok` (reproduces every recorded read), `missing_read_file` /
  `missing_edited_file` (the builder fills them from the trace), `differs` (not what the teacher
  saw: the builder ignores the code file for that trace).
""")
code('''
rep = json.loads((Path(CODE_OUT).parent / "fetch_code_report.json").read_text())
print({k: v for k, v in rep.items() if k != "per_task"})
per = rep["per_task"]
failed = Counter(r["status"] for r in per.values() if not r["status"].startswith(("fetched", "already")))
print("not fetched:", failed.most_common(10))
print("failed repositories:", Counter(r["repo"] for r in per.values()
                                      if not r["status"].startswith(("fetched", "already"))).most_common(10))
for iid, r in list((k, v) for k, v in per.items() if v["check"] and "differs" in v["check"].values())[:5]:
    print("differs:", iid, r["repo"], r["status"])
print("code files now:", len(list(Path(CODE_OUT).glob("*.json"))))
''')

md("""
## 3. Next

Re-run `build_records.ipynb` with the same `TAG` and `REBUILD_STEPS = ["locfix"]` (the single-agent
records do not use code files and are kept; set it back to `[]` if you must resume after a
disconnect). In the new `summary.json`, `code:code_json` should replace most of `code:trace`, and
`skip:missing_code` / `drop:*:unseen_code` should mostly disappear.
""")


def main() -> None:
    write_notebook(CELLS, ROOT / "notebooks" / "fetch_solved_code.ipynb")


if __name__ == "__main__":
    main()
