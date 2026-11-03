"""`locfix em.fold-split`: per-fold record files for Stage 3 cross-fitting (spec 06 §1).

Every task belongs to fold A or B. A model trained on fold A only ever generates candidates for,
scores, or is tested on fold B tasks, and the other way round, so no Stage 3 number comes from a
model that has seen the task's teacher solution.

    locfix em.fold-split work/records/locfix --out work/records/folds
      -> work/records/folds/A/locfix/s3_loc_r0.jsonl, .../A/locfix/s2_fix_gold.jsonl, ...
         work/records/folds/B/locfix/...
         work/records/folds/folds.json        {instance_id: "A" | "B"}  (read by every later step)
         work/records/folds/fold_split.json   counts per file and fold

`--by source` (default) uses the `fold` field of solved.jsonl that the record builders copied into
`source.fold`; the split is refused when one fold has under `--min-share` of the tasks (e.g.
solved.jsonl had no `fold` column and every task defaulted to A). `--by hash` assigns folds by a
seeded hash of the instance id instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

FOLDS = ("A", "B")


def other(fold: str) -> str:
    if fold not in FOLDS:
        raise ValueError(f"fold must be A or B, got {fold!r}")
    return "B" if fold == "A" else "A"


def hash_fold(instance_id: str, seed: int) -> str:
    h = int.from_bytes(hashlib.sha256(f"{seed}:fold:{instance_id}".encode()).digest()[:8], "big")
    return "A" if h % 2 == 0 else "B"


def load_folds(path: str | Path) -> dict[str, str]:
    """folds.json written by `em.fold-split` (the folder or the file)."""
    p = Path(path)
    if p.is_dir():
        p = p / "folds.json"
    folds = json.loads(p.read_text())
    bad = {k: v for k, v in folds.items() if v not in FOLDS}
    if bad:
        raise ValueError(f"{p}: folds other than A/B: {list(bad.items())[:3]}")
    return folds


def _jobs(inputs: list[str]) -> list[tuple[Path, Path]]:
    jobs: list[tuple[Path, Path]] = []
    for x in inputs:
        p = Path(x)
        if p.is_dir():
            jobs += [(f, Path(p.name) / f.name) for f in sorted(p.glob("*.jsonl"))]
        elif p.is_file():
            jobs.append((p, Path(p.name)))
        else:
            raise SystemExit(f"{p}: not found")
    if not jobs:
        raise SystemExit("no .jsonl record files in the inputs")
    return jobs


def assign_folds(jobs: list[tuple[Path, Path]], by: str, seed: int) -> dict[str, str]:
    folds: dict[str, str] = {}
    for src, _ in jobs:
        with open(src) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                iid = r["instance_id"]
                fold = hash_fold(iid, seed) if by == "hash" else (r.get("source") or {}).get("fold")
                if fold not in FOLDS:
                    raise SystemExit(f"{src}: record {r.get('id')} has source.fold={fold!r} "
                                     f"(use --by hash if solved.jsonl has no fold column)")
                if folds.setdefault(iid, fold) != fold:
                    raise SystemExit(f"task {iid} is in fold {folds[iid]} and {fold} (records disagree)")
    return folds


def fold_split(inputs: list[str], out: str | Path, by: str = "source", seed: int = 1234,
               min_share: float = 0.3) -> dict:
    jobs = _jobs(inputs)
    folds = assign_folds(jobs, by, seed)
    n = Counter(folds.values())
    share = min(n[f] for f in FOLDS) / max(1, len(folds))
    if share < min_share:
        raise SystemExit(f"fold sizes {dict(n)}: the smaller fold has {share:.0%} of the tasks (< {min_share:.0%}). "
                         "Check the fold column of solved.jsonl, or use --by hash.")
    out = Path(out)
    report: dict[str, dict[str, int]] = {}
    for src, rel in jobs:
        handles = {}
        cnt = Counter()
        try:
            for f in FOLDS:
                (out / f / rel).parent.mkdir(parents=True, exist_ok=True)
                handles[f] = open(out / f / rel, "w")  # noqa: SIM115
            with open(src) as fi:
                for line in fi:
                    if not line.strip():
                        continue
                    fold = folds[json.loads(line)["instance_id"]]
                    handles[fold].write(line if line.endswith("\n") else line + "\n")
                    cnt[fold] += 1
        finally:
            for h in handles.values():
                h.close()
        report[str(rel)] = {f: cnt[f] for f in FOLDS}
    (out / "folds.json").write_text(json.dumps(dict(sorted(folds.items())), indent=0))
    summary = {"by": by, "seed": seed, "tasks": {f: n[f] for f in FOLDS}, "files": report}
    (out / "fold_split.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- CLI
def add_args_cmd_fold_split(p: argparse.ArgumentParser) -> None:
    p.add_argument("inputs", nargs="+", help="record files (-> <out>/<fold>/<name>) or folders "
                   "(-> <out>/<fold>/<folder>/<name>); use the training files (not val/)")
    p.add_argument("--out", required=True)
    p.add_argument("--by", choices=["source", "hash"], default="source",
                   help="source: the records' source.fold (from solved.jsonl); hash: seeded hash of the id")
    p.add_argument("--min-share", type=float, default=0.3, help="refuse if a fold has fewer tasks than this share")


def cmd_fold_split(args: argparse.Namespace, cfg: dict) -> int:
    from ..runmeta import write_run_meta

    s = fold_split(args.inputs, args.out, args.by, int(cfg["seed"]), args.min_share)
    write_run_meta(Path(args.out), cfg, s)
    print(json.dumps(s, indent=2))
    return 0
