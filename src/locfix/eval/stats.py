"""Statistics on evaluation results (`task_results.jsonl` written by the harness's evaluator).

- Resolve rate with a Wilson 95% interval.
- Paired comparison of two systems on the same tasks: exact McNemar test (two-sided binomial test
  on the discordant tasks). With several seeds, each (task, seed) pair is one paired observation,
  and the per-seed results are reported too.

    locfix eval.summary work/eval/S1-e4b/seed0 work/eval/S2b-e4b/seed0 ...
    locfix eval.compare --a work/eval/S2b-e4b/seed0 work/eval/S2b-e4b/seed1 \
                        --b work/eval/S3-e4b/seed0  work/eval/S3-e4b/seed1 --name-a S2b --name-b S3
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: b = only A solved, c = only B solved."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_results(results_dir: str | Path) -> dict[str, dict[str, Any]]:
    """instance_id -> the task's result row (the last one, if a task was evaluated twice)."""
    path = Path(results_dir) / "task_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path}: no results (did the evaluation run?)")
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["instance_id"]] = row
    return out


def _error_kind(err: str | None) -> str:
    if not err:
        return "none"
    e = err.lower()
    for key in ("timeout", "timed out", "budget", "apply", "context", "length", "connection", "docker"):
        if key in e:
            return key
    return err.split(":")[0][:40]


def summarize(results_dir: str | Path) -> dict[str, Any]:
    rows = list(load_results(results_dir).values())
    n = len(rows)
    k = sum(bool(r.get("resolved")) for r in rows)
    lo, hi = wilson(k, n)
    by_repo: dict[str, list[int]] = {}
    for r in rows:
        by_repo.setdefault(r.get("repo") or "?", [0, 0])
        by_repo[r.get("repo") or "?"][0] += bool(r.get("resolved"))
        by_repo[r.get("repo") or "?"][1] += 1

    def med(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(statistics.median(vals), 2) if vals else None

    return {
        "results_dir": str(results_dir), "tasks": n, "resolved": k,
        "resolve_rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)],
        "empty_patch": sum(1 for r in rows if not r.get("agent_patch_size")),
        "errors": dict(Counter(_error_kind(r.get("error")) for r in rows if r.get("error"))),
        "median_tool_calls": med("tool_calls"), "median_duration_s": med("duration_seconds"),
        "median_llm_calls": med("total_llm_calls"),
        "by_repo": {repo: {"resolved": a, "tasks": b} for repo, (a, b) in sorted(by_repo.items())},
    }


def compare(a_dirs: list[str], b_dirs: list[str], name_a: str = "A", name_b: str = "B") -> dict[str, Any]:
    """Paired comparison on the tasks both systems were evaluated on. a_dirs[i] and b_dirs[i] are
    the same seed (pass the seeds in the same order)."""
    if len(a_dirs) != len(b_dirs):
        raise ValueError("pass the same number of result folders (one per seed) for both systems")
    per_seed, pooled = [], {"both": 0, "only_a": 0, "only_b": 0, "neither": 0}
    for da, db in zip(a_dirs, b_dirs, strict=True):
        ra, rb = load_results(da), load_results(db)
        common = sorted(set(ra) & set(rb))
        cnt = {"both": 0, "only_a": 0, "only_b": 0, "neither": 0}
        for t in common:
            x, y = bool(ra[t].get("resolved")), bool(rb[t].get("resolved"))
            cnt["both" if x and y else "only_a" if x else "only_b" if y else "neither"] += 1
        for key in cnt:
            pooled[key] += cnt[key]
        n = len(common)
        per_seed.append({"a": str(da), "b": str(db), "tasks": n,
                         "missing_in_a": len(set(rb) - set(ra)), "missing_in_b": len(set(ra) - set(rb)),
                         f"rate_{name_a}": round((cnt["both"] + cnt["only_a"]) / n, 4) if n else None,
                         f"rate_{name_b}": round((cnt["both"] + cnt["only_b"]) / n, 4) if n else None,
                         **cnt, "p_mcnemar": round(mcnemar_exact(cnt["only_a"], cnt["only_b"]), 4)})
    n = sum(pooled.values())
    ra_ = (pooled["both"] + pooled["only_a"]) / n if n else 0.0
    rb_ = (pooled["both"] + pooled["only_b"]) / n if n else 0.0
    return {
        "name_a": name_a, "name_b": name_b, "pairs": n, **pooled,
        f"rate_{name_a}": round(ra_, 4), f"rate_{name_b}": round(rb_, 4),
        "difference_b_minus_a": round(rb_ - ra_, 4),
        "p_mcnemar": round(mcnemar_exact(pooled["only_a"], pooled["only_b"]), 4),
        "per_seed": per_seed,
    }


# --------------------------------------------------------------------------- CLI
def add_args_cmd_summary(p: argparse.ArgumentParser) -> None:
    p.add_argument("results", nargs="+", help="result folders written by eval.run")
    p.add_argument("--out", help="write the summaries as JSON here")


def cmd_summary(args: argparse.Namespace, cfg: dict) -> int:
    sums = [summarize(d) for d in args.results]
    print("| results | tasks | resolved | rate | 95% CI | empty patch | median calls | median s |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in sums:
        print(f"| {s['results_dir']} | {s['tasks']} | {s['resolved']} | {s['resolve_rate']} | "
              f"{s['ci95'][0]}–{s['ci95'][1]} | {s['empty_patch']} | {s['median_tool_calls']} | "
              f"{s['median_duration_s']} |")
    if args.out:
        Path(args.out).write_text(json.dumps(sums, indent=2))
    return 0


def add_args_cmd_compare(p: argparse.ArgumentParser) -> None:
    p.add_argument("--a", nargs="+", required=True, help="result folders of system A, one per seed")
    p.add_argument("--b", nargs="+", required=True, help="result folders of system B, same seeds in the same order")
    p.add_argument("--name-a", default="A")
    p.add_argument("--name-b", default="B")
    p.add_argument("--out", help="write the comparison as JSON here")


def cmd_compare(args: argparse.Namespace, cfg: dict) -> int:
    res = compare(args.a, args.b, args.name_a, args.name_b)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return 0
