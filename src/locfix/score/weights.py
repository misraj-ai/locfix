"""`locfix score.weights`: scores -> EM weights (spec 06 §5), CPU.

    score(z, a) = logp_plus_mean − λ_ctx · n_z_tokens / 1000 − λ_req · n_requests
    w(z, a)     = softmax over the task's (z, a) pairs of score / τ_i
    w(z)        = Σ_a w(z, a)                         (marginal weight of a localization)
    z*          = argmax_z w(z)                       (ties: the lower candidate id, z_T first)

- **λ calibration** (from these scores, round 0): σ = median over tasks of the standard deviation of
  `logp_plus_mean` across the task's candidates (mode `none`, tasks with ≥ 2 candidates).
  λ_ctx makes a median-sized z (median `n_z_tokens` of the mode-`none` sequences) cost
  `em.ctx_cost_sigma` × σ; λ_req makes one request cost `em.req_cost_sigma` × σ. Both can be fixed
  instead with `--lambda-ctx` / `--lambda-req` (e.g. to reuse round-0 values in later rounds).
- **τ per task** is set by bisection on log τ so that the effective sample size 1/Σ w² equals
  `em.ess_target` (within 1%). ESS target 1 is hard EM (all weight on the best pair); a target at or
  above the number of pairs gives uniform weights.

Outputs: `weights.jsonl` (one row per task: pairs with score and w, candidates with w(z), z*) and
`weights_summary.json` (λ, σ, ESS, and the diagnostics of spec 06: weight entropy, share of tasks
whose z* is the teacher label, z sizes, requests, recall of L_edit).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- math
def softmax(xs: list[float], tau: float) -> list[float]:
    if tau <= 0 or math.isinf(tau):
        raise ValueError("tau must be finite and > 0")
    m = max(xs)
    e = [math.exp((x - m) / tau) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def ess(w: list[float]) -> float:
    return 1.0 / sum(v * v for v in w)


def one_hot_best(xs: list[float]) -> list[float]:
    i = max(range(len(xs)), key=lambda k: (xs[k], -k))
    return [1.0 if k == i else 0.0 for k in range(len(xs))]


def weights_for_ess(scores: list[float], target: float, tol: float = 0.01) -> tuple[list[float], float, float]:
    """(w, τ, achieved ESS). τ = 0 means hard (one-hot); τ = inf means uniform."""
    n = len(scores)
    if n == 1:
        return [1.0], math.inf, 1.0
    if target <= 1.0:
        return one_hot_best(scores), 0.0, 1.0
    if target >= n or max(scores) - min(scores) < 1e-12:
        return [1.0 / n] * n, math.inf, float(n)
    lo, hi = -30.0, 30.0                       # log τ; ESS rises with τ
    if ess(softmax(scores, math.exp(lo))) > target * (1 + tol):
        w = softmax(scores, math.exp(lo))
        return w, math.exp(lo), ess(w)
    for _ in range(200):
        mid = (lo + hi) / 2
        w = softmax(scores, math.exp(mid))
        e = ess(w)
        if abs(e - target) <= tol * target:
            return w, math.exp(mid), e
        if e < target:
            lo = mid
        else:
            hi = mid
    w = softmax(scores, math.exp((lo + hi) / 2))
    return w, math.exp((lo + hi) / 2), ess(w)


def entropy(w: list[float]) -> float:
    return -sum(v * math.log(v) for v in w if v > 0)


# --------------------------------------------------------------------------- calibration
def calibrate(rows_by_task: dict[str, list[dict[str, Any]]], ctx_cost_sigma: float, req_cost_sigma: float
              ) -> dict[str, float]:
    stds, ztok = [], []
    for rows in rows_by_task.values():
        none = [r for r in rows if r["mode"] == "none"]
        ztok += [r["n_z_tokens"] for r in none]
        if len(none) >= 2:
            stds.append(statistics.pstdev([r["logp_plus_mean"] for r in none]))
    if not stds or not ztok:
        raise ValueError("cannot calibrate λ: no task with 2+ scored candidates")
    sigma = statistics.median(stds)
    med_z = statistics.median(ztok)
    return {"sigma": sigma, "median_z_tokens": med_z,
            "lambda_ctx": ctx_cost_sigma * sigma / max(med_z / 1000.0, 1e-9),
            "lambda_req": req_cost_sigma * sigma, "tasks_used": len(stds)}


def pair_score(r: dict[str, Any], lam_ctx: float, lam_req: float) -> float:
    return r["logp_plus_mean"] - lam_ctx * r["n_z_tokens"] / 1000.0 - lam_req * r["n_requests"]


# --------------------------------------------------------------------------- per task
def _locset(z: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(e["path"], e["symbol"]) for e in z}


def task_weights(rows: list[dict[str, Any]], cand: dict[str, Any], lam_ctx: float, lam_req: float,
                 target: float) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: (r["cand_id"], r["mode"]))
    sc = [pair_score(r, lam_ctx, lam_req) for r in rows]
    w, tau, e = weights_for_ess(sc, target)
    wz: dict[int, float] = {}
    for r, v in zip(rows, w, strict=True):
        wz[r["cand_id"]] = wz.get(r["cand_id"], 0.0) + v
    zstar = max(sorted(wz), key=lambda c: (wz[c], -c))
    zs = {c["cand_id"]: c for c in cand["candidates"]}
    best_pair = max(range(len(rows)), key=lambda k: (sc[k], -k))
    raw_best = max(sorted(wz), key=lambda c: (max(r["logp_plus_mean"] for r in rows if r["cand_id"] == c), -c))
    teacher_none = next((r for r in rows if r["cand_id"] == 0 and r["mode"] == "none"), None)
    star_none = next((r for r in rows if r["cand_id"] == zstar and r["mode"] == "none"), None)
    l_edit = _locset(cand.get("l_edit", []))
    z_star = zs[zstar]["z"]
    return {
        "instance_id": cand["instance_id"], "fold": cand["fold"], "trace": cand.get("trace"), "tau": None if math.isinf(tau) else tau, "ess": round(e, 4),
        "entropy": round(entropy(w), 5), "n_pairs": len(rows), "n_candidates": len(wz),
        "pairs": [{"cand_id": r["cand_id"], "mode": r["mode"], "source": zs[r["cand_id"]]["source"],
                   "logp_plus_mean": r["logp_plus_mean"], "n_z_tokens": r["n_z_tokens"],
                   "n_requests": r["n_requests"], "score": round(s, 6), "w": round(v, 6)}
                  for r, s, v in zip(rows, sc, w, strict=True)],
        "candidates": [{"cand_id": c, "source": zs[c]["source"], "z": zs[c]["z"], "w_z": round(wz[c], 6)}
                       for c in sorted(wz)],
        "z_star": zstar, "z_star_source": zs[zstar]["source"], "z_star_is_teacher": zstar == 0,
        "z_star_z": z_star, "z_teacher": zs[0]["z"] if 0 in zs else None,
        "best_pair": {"cand_id": rows[best_pair]["cand_id"], "mode": rows[best_pair]["mode"]},
        "raw_best": raw_best,
        "gap_vs_teacher": (round(star_none["logp_plus_mean"] - teacher_none["logp_plus_mean"], 6)
                           if star_none and teacher_none else None),
        "z_star_tokens": star_none["n_z_tokens"] if star_none else None,
        "teacher_tokens": teacher_none["n_z_tokens"] if teacher_none else None,
        "z_star_recall_edit": (len(_locset(z_star) & l_edit) / len(l_edit)) if l_edit else None,
    }


def summarize(out_rows: list[dict[str, Any]], cal: dict[str, float], target: float, skipped: Counter) -> dict[str, Any]:
    n = len(out_rows)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.fmean(xs), 5) if xs else None

    srcs = Counter(r["z_star_source"] for r in out_rows)
    return {
        "tasks": n, "skipped": dict(skipped), "ess_target": target, **{k: round(v, 6) for k, v in cal.items()},
        "ess_mean": mean([r["ess"] for r in out_rows]), "entropy_mean": mean([r["entropy"] for r in out_rows]),
        "z_star_is_teacher": round(sum(r["z_star_is_teacher"] for r in out_rows) / n, 4) if n else None,
        "z_star_source": dict(srcs),
        "raw_best_is_teacher": round(sum(r["raw_best"] == 0 for r in out_rows) / n, 4) if n else None,
        "best_pair_uses_requests": round(sum(r["best_pair"]["mode"] == "recover" for r in out_rows) / n, 4) if n else None,
        "z_star_tokens_mean": mean([r["z_star_tokens"] for r in out_rows]),
        "teacher_tokens_mean": mean([r["teacher_tokens"] for r in out_rows]),
        "gap_vs_teacher_mean": mean([r["gap_vs_teacher"] for r in out_rows]),
        "z_star_recall_edit_mean": mean([r["z_star_recall_edit"] for r in out_rows]),
        "candidates_mean": mean([r["n_candidates"] for r in out_rows]),
    }


def compute_weights(score_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]], em: dict[str, Any], *,
                    lambda_ctx: float | None = None, lambda_req: float | None = None,
                    ess_target: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cands = {c["instance_id"]: c for c in cand_rows}
    by: dict[str, list[dict[str, Any]]] = {}
    skipped: Counter = Counter()
    for r in score_rows:
        if r.get("error"):
            skipped[f"pair:{r['error']}"] += 1
            continue
        by.setdefault(r["instance_id"], []).append(r)
    for iid in list(by):
        if iid not in cands:
            raise ValueError(f"{iid}: scores without a candidates row (pass the same score.prepare outputs)")
        if not any(r["cand_id"] == 0 for r in by[iid]):
            skipped["task:teacher_not_scored"] += 1
            del by[iid]
    cal = calibrate(by, float(em["ctx_cost_sigma"]), float(em["req_cost_sigma"]))
    if lambda_ctx is not None:
        cal["lambda_ctx"] = lambda_ctx
    if lambda_req is not None:
        cal["lambda_req"] = lambda_req
    target = float(ess_target if ess_target is not None else em["ess_target"])
    out = [task_weights(by[i], cands[i], cal["lambda_ctx"], cal["lambda_req"], target) for i in sorted(by)]
    return out, summarize(out, cal, target, skipped)


# --------------------------------------------------------------------------- CLI
def add_args_cmd_weights(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scores", nargs="+", required=True, help="score.run outputs")
    p.add_argument("--candidates", nargs="+", required=True, help="candidates.jsonl of score.prepare")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--ess-target", type=float, help="default em.ess_target (1 = hard EM)")
    p.add_argument("--lambda-ctx", type=float, help="fix λ_ctx (per 1000 z tokens) instead of calibrating")
    p.add_argument("--lambda-req", type=float, help="fix λ_req (per request) instead of calibrating")


def cmd_weights(args: argparse.Namespace, cfg: dict) -> int:
    from ..em.tasks import read_jsonl_many, write_jsonl
    from ..runmeta import write_run_meta

    rows, summ = compute_weights(read_jsonl_many(args.scores), read_jsonl_many(args.candidates), cfg["em"],
                                 lambda_ctx=args.lambda_ctx, lambda_req=args.lambda_req, ess_target=args.ess_target)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "weights.jsonl", rows)
    (out / "weights_summary.json").write_text(json.dumps(summ, indent=2))
    write_run_meta(out, cfg, {"weights": summ})
    print(json.dumps(summ, indent=2))
    return 0
