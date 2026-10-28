"""Write notebooks/build_records.ipynb: build all training records on a Colab CPU runtime.

    python tools/build_records_notebook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb_common import GET_CODE_CELL, TOKENIZER_CELL, packages_cell, write_notebook  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# Build the training records (Colab, CPU runtime)

Turns the cleaned teacher traces into the record files the trainer reads (docs/DATA.md, step 2),
checks them, and saves one archive to Google Drive for the training node.

```
cleaned_v2/traces + solved.jsonl (+ code/)
   │  A. locfix data.agent-records    single agent   → s1/s1_agent.jsonl  + s1_agent.tasks.txt (kept trajectories)
   │  B. locfix data.locfix-records   two agents     → locfix/s2_*.jsonl, s3_loc_r0.jsonl  (same trajectories)
   │  C. locfix data.check-records + data.stats      → checks/, stats/stats.md
   │  D. locfix data.split                      → records/ (train), records/val/ (3% of tasks)
   ▼
records_<tag>.tar.gz on Drive  →  training node
```

Settings come from `configs/defaults.yaml` (printed in section 1): budget **36 tool calls,
8 minutes** (two agents: 44 calls), trajectories whose records render longer than
**32,768 tokens** are dropped (`data.max_trace_tokens`), localizer explorations up to 15 calls.

**Runtime:** CPU is enough (no GPU needed). Work is split into `N_SHARDS` parts run in parallel, one per CPU
core. Rough time on the free runtime (2 cores): A ≈ 15 min, B ≈ 45–60 min. A Colab Pro high-RAM CPU
runtime (8 cores) is ≈ 4× faster. Finished parts are saved on Drive, so after a disconnect re-run
all cells: finished parts are skipped.

**Needs:** the code (repo or `locfix_code.zip`), the three harness wheels, the data folder on
Drive, Kaggle access to the Gemma 4 tokenizer (3 small files, no weights).
""")

md("## 0. Setup")
code(GET_CODE_CELL)
code('''
# --- Paths (Google Drive)
DATA_DIR = "/content/drive/MyDrive/codeagent_final"
TRACES = f"{DATA_DIR}/cleaned_v2/traces"                # cleaned teacher traces (*.json)
SOLVED = f"{DATA_DIR}/cleaned_v2/solved.jsonl"
CODE_DIRS = [f"{DATA_DIR}/cleaned_v2/code", f"{DATA_DIR}/experiment/code"]   # code/<id>.json; missing folders are skipped
WHEELS_DIR = "/content/drive/MyDrive/wheels"            # swegemma, adk_submission, adk_eval_core .whl
OUT_DIR = "/content/drive/MyDrive/locfix_records"       # results (Drive: survives a disconnect)
TAG = "b36_t8_32k"                                      # name of this data version (budget/time/length)

# --- Run
N_SHARDS = 8            # parts of the data; keep it fixed between re-runs (finished parts are reused)
WORKERS = None          # parallel processes; None = number of CPU cores
LIMIT = 0               # > 0: only the first LIMIT traces (a quick try; use a different TAG)
EXTRA_SETTINGS = []     # extra overrides, e.g. ["budget.max_tool_calls=40"] (then change TAG too)
REBUILD_STEPS = []      # build these steps again even if finished, e.g. ["locfix"] after adding code files
                        # (after a disconnect during that rebuild, set it back to [] to resume)

import os, sys, json, pathlib, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
for p in (TRACES, SOLVED, WHEELS_DIR):
    print(p, "found" if pathlib.Path(p).exists() else "NOT FOUND")
CODE_DIRS = [d for d in CODE_DIRS if pathlib.Path(d).is_dir()]
print("code folders:", CODE_DIRS or "none (code rebuilt from the traces)")
WORKERS = WORKERS or os.cpu_count()
RUN_DIR = pathlib.Path(OUT_DIR) / TAG
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("cores:", os.cpu_count(), "| workers:", WORKERS, "| results:", RUN_DIR)
''')
code(packages_cell("True"))
code(TOKENIZER_CELL)

md("""
## 1. Copy the traces to local disk and check the settings

Reading ~1,600 small files from Drive is slow; a local copy makes the build several times faster.
""")
code('''
import shutil, time

LOCAL = pathlib.Path("/content/data")
t0 = time.time()
if not (LOCAL / "traces").exists():
    shutil.copytree(TRACES, LOCAL / "traces.tmp")
    (LOCAL / "traces.tmp").rename(LOCAL / "traces")
shutil.copy(SOLVED, LOCAL / "solved.jsonl")
local_code = []
for i, d in enumerate(CODE_DIRS):
    dst = LOCAL / f"code{i}"
    if not dst.exists():
        shutil.copytree(d, dst)
    local_code.append(str(dst))
n_traces = len(list((LOCAL / "traces").glob("*.json")))
print(f"{n_traces} traces, {len(local_code)} code folders copied in {time.time() - t0:.0f}s")

ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
SET = [x for s in EXTRA_SETTINGS for x in ("--set", s)]


def locfix(*args, log=None):
    """Run `locfix <args>` from the repository root (agent folders are relative to it)."""
    cmd = [sys.executable, "-m", "locfix.cli", *map(str, args)]
    if log is None:
        r = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-5000:], r.stderr[-5000:])
            raise RuntimeError(f"failed: {' '.join(cmd)}")
        return r.stdout
    return subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=open(log, "w"), stderr=subprocess.STDOUT)


cfg = json.loads(locfix("config", *SET))
show = {"budget": cfg["budget"], "budget_locfix": cfg["budget_locfix"],
        "data.max_trace_tokens": cfg["data"]["max_trace_tokens"],
        "data.time_scale": cfg["data"]["time_scale"], "data.teacher_time_minutes": cfg["data"]["teacher_time_minutes"],
        "data.exclude_repos (dev set)": cfg["data"]["exclude_repos"],
        "em.localizer_max_calls": cfg["em"]["localizer_max_calls"], "seed": cfg["seed"]}
print(json.dumps(show, indent=1))
(RUN_DIR / "settings.json").write_text(json.dumps(show, indent=1))
''')

md("""
## 2. Run helper: parts in parallel, finished parts skipped

Each part writes to `RUN_DIR/parts/<step>/<k>/`; a part is finished when its `run_meta.json`
exists. Progress is printed every 30 s (parts finished, MB of records written). Then the parts are merged
(`locfix data.merge-records`) into `RUN_DIR/<step>/`.
""")
code('''
def mb_written(d):
    return sum(f.stat().st_size for f in pathlib.Path(d).glob("*.jsonl")) / 1e6


def run_parts(step, make_args):
    """make_args(k, part_dir) -> locfix arguments for part k of N_SHARDS."""
    parts = [RUN_DIR / "parts" / step / str(k) for k in range(N_SHARDS)]
    if step in REBUILD_STEPS:
        for d in parts:
            (d / "run_meta.json").unlink(missing_ok=True)     # makes every part run again
    todo = [k for k, d in enumerate(parts) if not (d / "run_meta.json").exists()]
    print(f"{step}: {N_SHARDS - len(todo)} parts already finished, {len(todo)} to run, {WORKERS} at a time")
    running, t0 = {}, time.time()
    while todo or running:
        while todo and len(running) < WORKERS:
            k = todo.pop(0)
            if parts[k].exists():
                shutil.rmtree(parts[k])                    # a part interrupted earlier: start it again
            parts[k].mkdir(parents=True)
            running[k] = locfix(*make_args(k, parts[k]), log=parts[k].parent / f"{k}.log")
        time.sleep(30)
        for k, proc in list(running.items()):
            if proc.poll() is not None:
                del running[k]
                if proc.returncode:
                    print(open(parts[k].parent / f"{k}.log").read()[-5000:])
                    raise RuntimeError(f"{step} part {k} failed (log above)")
        done = sum((d / "run_meta.json").exists() for d in parts)
        print(f"  {time.time() - t0:5.0f}s  parts done {done}/{N_SHARDS}  written {sum(mb_written(d) for d in parts if d.exists()):.0f} MB")
    out = RUN_DIR / step
    print(locfix("data.merge-records", *parts, "--out", out, *SET)[-3000:])
    return out


LIMIT_ARGS = ["--limit", LIMIT] if LIMIT else []
''')

md("""
## 3. Step A: single-agent records

Drops (counted in the report): trajectories over the budget (`skip_over_budget`), longer than
`data.max_trace_tokens` (`skip_too_long`), dev-set repositories (`skip_dev_repo`), not in
`solved.jsonl` (`skip_not_in_solved`), and traces that fail to load (`error:*`, e.g. the 8 traces
whose last call has no response).
""")
code('''
S1 = run_parts("s1", lambda k, d: ["data.agent-records", LOCAL / "traces", "--solved", LOCAL / "solved.jsonl",
                                   "--out", d / "s1_agent.jsonl", "--tokenizer", TOKENIZER_DIR,
                                   "--shard", f"{k}/{N_SHARDS}", *LIMIT_ARGS, *SET])
TASKS = S1 / "s1_agent.tasks.txt"
print("trajectories kept:", len(TASKS.read_text().split()))
# quick check before the long step B: schema + rendering of 200 random records (stops here if any fails)
print(locfix("data.check-records", S1 / "s1_agent.jsonl", "--tokenizer", TOKENIZER_DIR, "--sample", 200, *SET)[-1500:])
''')

md("""
## 4. Step B: localizer and fixer records (the same trajectories)

`--tasks` restricts the two-agent records to the trajectories kept in step A, so the single-agent
baseline and our method train on the same tasks. A localizer or fixer record longer than the limit
is dropped on its own (`drop:<stage>:too_long`).
""")
code('''
CODE_ARGS = [x for d in local_code for x in ("--code", d)]
LF = run_parts("locfix", lambda k, d: ["data.locfix-records", LOCAL / "traces", "--solved", LOCAL / "solved.jsonl",
                                       *CODE_ARGS, "--out", d, "--tasks", TASKS, "--tokenizer", TOKENIZER_DIR,
                                       "--shard", f"{k}/{N_SHARDS}", *LIMIT_ARGS, *SET])
''')

md("""
## 5. Step C: check the records and write the statistics

`data.check-records` validates every record against the schema and renders it with the trainer's
renderer; `data.stats --records` gives exact token counts per stage and the multi-GPU step-time
simulation (read `stats/stats.md`).
""")
code('''
files = [S1 / "s1_agent.jsonl", *sorted(LF.glob("*.jsonl"))]
print(locfix("data.check-records", *files, "--tokenizer", TOKENIZER_DIR, "--max-tokens", cfg["data"]["max_trace_tokens"],
             "--preview", RUN_DIR / "checks", *SET)[-6000:])
locfix("data.stats", "--records", S1 / "s1_agent.jsonl", LF, "--tokenizer", TOKENIZER_DIR,
       "--out", RUN_DIR / "stats", "--world-size", 8, "--global-batch", 16, *SET)
from IPython.display import Markdown, display
display(Markdown((RUN_DIR / "stats" / "stats.md").read_text()[:20000]))
''')

md("""
## 6. Training / validation split, summary, archive for the training node

`locfix data.split` puts `data.val_fraction` (3%) of the **tasks** into validation files (all
records of a task on the same side, the same tasks in every stage; seeded). The layout is the one
the run configs expect (`${paths.records}/s1_agent.jsonl`, `.../locfix/...`, `.../val/...`).
""")
code('''
REC = RUN_DIR / "records"
print(locfix("data.split", S1 / "s1_agent.jsonl", LF, "--out", REC, *SET))
shutil.copy(S1 / "manifest.json", REC / "s1_manifest.json")
shutil.copy(S1 / "s1_agent.tasks.txt", REC / "s1_agent.tasks.txt")
shutil.copy(LF / "manifest.json", REC / "locfix_manifest.json")
shutil.copytree(RUN_DIR / "stats", REC / "stats", dirs_exist_ok=True)
shutil.copy(RUN_DIR / "settings.json", REC / "settings.json")

summary = {"settings": show,
           "s1": json.loads((S1 / "manifest.json").read_text()),
           "locfix": json.loads((LF / "manifest.json").read_text())}
(REC / "summary.json").write_text(json.dumps(summary, indent=1))
print("single agent:", summary["s1"]["files"], "| counts:", json.dumps(summary["s1"]["counts"]))
print("two agents:  ", summary["locfix"]["files"])
print("two-agent counts:", json.dumps(summary["locfix"]["counts"], indent=1, sort_keys=True))

archive = pathlib.Path(OUT_DIR) / f"records_{TAG}.tar.gz"
subprocess.run(["tar", "czf", str(archive), "-C", str(RUN_DIR), "records"], check=True)
print(f"{archive}  {archive.stat().st_size / 1e6:.0f} MB")
''')

md("""
## 7. On the training node

Download `records_<TAG>.tar.gz` from Drive (any of these):

```bash
# a) share the file ("anyone with the link"), then
pip install gdown && gdown --fuzzy "<share link>" -O records.tar.gz
# b) rclone with a Google Drive remote
rclone copy gdrive:locfix_records/records_<TAG>.tar.gz .
```

```bash
mkdir -p work && tar xzf records.tar.gz -C work        # -> work/records/
#   work/records/s1_agent.jsonl                       S1 runs (train)
#   work/records/locfix/s2_*.jsonl, s3_loc_r0.jsonl   S2 / S3 runs (train)
#   work/records/val/...                              the same files for validation
#   work/records/stats/stats.md, summary.json         what was built and dropped
locfix train.plan --config configs/runs/S1-e4b.yaml --world-size 8 --tokenizer <tokenizer dir>
```
`paths.records` defaults to `./work/records` (or set it in `configs/paths.yaml`).
""")


def main() -> None:
    write_notebook(CELLS, ROOT / "notebooks" / "build_records.ipynb")


if __name__ == "__main__":
    main()
