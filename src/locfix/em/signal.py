"""`locfix em.signal-test`: does the EM target help the fixer? (spec 06 §8; run once after round 0)

For each Stage 3 task (training tasks, cross-fitted: fold B tasks with the fold-A fixer and the other
way round), the fold fixer makes its own attempt twice (`em.rollout`, greedy):

    T   given z_T, the teacher label (L_edit ∪ L_view, pruned): what round 0 trains the localizer on
    S   given z*, the candidate with the highest weight (score.weights): what round 1 would train on

and each attempt is compared with the verified patch. When z* = z_T the two attempts are the same
(one rollout, counted for both). Paired comparison per metric (exact McNemar on the discordant
tasks), primary metric `all_hunks` (every hunk of the verified patch reproduced).

Decision (`decide`):
    pass          S − T ≥ +2 points on the primary metric and McNemar p < 0.10
                  -> run round 1 (the EM target helps the fixer)
    pass_cheaper  |S − T| ≤ 1 point and z* is ≥ 20% smaller than z_T (mean z tokens)
                  -> run round 1 (same help for less context: a smaller, faster localizer answer)
    more_tasks    S − T ≥ +2 points but p ≥ 0.10 -> rerun with more tasks (--limit 0 = all)
    uninformative T reproduces y* on < 5% of tasks, or the fixer edits in < 30% of T rollouts:
                  the proxy cannot see a difference; check the stop reasons before deciding
    fail          otherwise -> stop after round 0; the paper reports the baselines and the analysis

All thresholds are options. The vLLM server is the harness's (`adk_submission.VllmServer`, gemma4
tool and reasoning parsers, max_model_len 32768) with both fold fixers as LoRA modules; or pass
`--server-url` of a running one. Results are appended to `rollouts.jsonl` as they finish; re-running
with the same --out continues.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

METRICS = ("all_hunks", "any_hunk", "first_edit_loc", "edited")
CONDITIONS = ("T", "S")


# --------------------------------------------------------------------------- report
def paired(results: dict[str, dict[str, dict[str, Any]]], metric: str) -> dict[str, Any]:
    from ..eval.stats import mcnemar_exact, wilson

    n = both = only_t = only_s = 0
    for r in results.values():
        if "T" not in r or "S" not in r:
            continue
        t, s = bool(r["T"].get(metric)), bool(r["S"].get(metric))
        n += 1
        both += t and s
        only_t += t and not s
        only_s += s and not t
    rt = (both + only_t) / n if n else 0.0
    rs = (both + only_s) / n if n else 0.0
    return {"tasks": n, "rate_T": round(rt, 4), "rate_S": round(rs, 4), "diff_S_minus_T": round(rs - rt, 4),
            "only_T": only_t, "only_S": only_s, "p_mcnemar": round(mcnemar_exact(only_t, only_s), 4),
            "ci95_T": [round(x, 4) for x in wilson(both + only_t, n)],
            "ci95_S": [round(x, 4) for x in wilson(both + only_s, n)]}


def decide(rep: dict[str, Any], primary: str = "all_hunks", min_gain: float = 0.02, alpha: float = 0.10,
           tie_band: float = 0.01, min_token_cut: float = 0.20, min_base_rate: float = 0.05,
           min_edit_rate: float = 0.30) -> dict[str, str]:
    m = rep["metrics"][primary]
    cut = rep["z_tokens"]["cut"]
    if m["rate_T"] < min_base_rate or rep["metrics"]["edited"]["rate_T"] < min_edit_rate:
        return {"decision": "uninformative",
                "why": f"T reproduces y* on {m['rate_T']:.1%} and edits on {rep['metrics']['edited']['rate_T']:.1%} "
                       f"of tasks: look at stop reasons (rollouts end before editing?) before deciding"}
    d = m["diff_S_minus_T"]
    if d >= min_gain and m["p_mcnemar"] < alpha:
        return {"decision": "pass", "why": f"S − T = {d:+.1%} on {primary}, p = {m['p_mcnemar']}"}
    if abs(d) <= tie_band and cut is not None and cut >= min_token_cut:
        return {"decision": "pass_cheaper", "why": f"S − T = {d:+.1%} (tie) with z* {cut:.0%} smaller than z_T"}
    if d >= min_gain:
        return {"decision": "more_tasks", "why": f"S − T = {d:+.1%} but p = {m['p_mcnemar']} >= {alpha}: rerun on more tasks"}
    return {"decision": "fail", "why": f"S − T = {d:+.1%} on {primary} (p = {m['p_mcnemar']}), z* {cut or 0:.0%} smaller"}


def is_broken(r: dict[str, Any]) -> bool:
    return r["stop"] in ("model_error", "replay_error") or r["stop"].startswith("prefix:")


def report(rows: list[dict[str, Any]], weights: dict[str, dict[str, Any]], **thresholds: Any) -> dict[str, Any]:
    res: dict[str, dict[str, dict[str, Any]]] = {}
    stops: dict[str, Counter] = {c: Counter() for c in CONDITIONS}
    for r in rows:
        res.setdefault(r["instance_id"], {})[r["condition"]] = r
        stops[r["condition"]][r["stop"]] += 1
    for iid, r in res.items():                     # z* = z_T: one rollout serves both conditions
        if "T" in r and "S" not in r and weights[iid]["z_star_is_teacher"]:
            r["S"] = r["T"]
    # infrastructure failures (server, replay, prefix) say nothing about z: such tasks are left out
    broken = {i for i, r in res.items() for x in r.values() if is_broken(x)}
    done = [i for i, r in res.items() if "T" in r and "S" in r and i not in broken]
    differ = {i: res[i] for i in done if not weights[i]["z_star_is_teacher"]}
    zt = [weights[i]["teacher_tokens"] for i in done if weights[i].get("teacher_tokens") is not None]
    zs = [weights[i]["z_star_tokens"] for i in done if weights[i].get("z_star_tokens") is not None]
    cut = (1 - sum(zs) / sum(zt)) if zt and zs and sum(zt) else None
    rec = {c: [res[i][c]["added_recall"] for i in done if res[i][c].get("added_recall") is not None] for c in CONDITIONS}
    rep = {
        "tasks": len(done), "z_star_differs": len(differ), "excluded_broken": len(broken),
        "metrics": {m: paired({i: res[i] for i in done}, m) for m in METRICS},
        "metrics_where_z_star_differs": {m: paired(differ, m) for m in METRICS},
        "added_recall_mean": {c: round(sum(v) / len(v), 4) if v else None for c, v in rec.items()},
        "z_tokens": {"T_mean": round(sum(zt) / len(zt), 1) if zt else None,
                     "S_mean": round(sum(zs) / len(zs), 1) if zs else None,
                     "cut": round(cut, 4) if cut is not None else None},
        "stop_reasons": {c: dict(v.most_common()) for c, v in stops.items()},
        "z_star_source": dict(Counter(weights[i]["z_star_source"] for i in done)),
    }
    rep.update(decide(rep, **thresholds))
    rep["thresholds"] = thresholds
    return rep


# --------------------------------------------------------------------------- jobs
_W: dict[str, Any] = {}


def _init(traces: str, solved: str, code: list[str], cfg: dict, url: str, seed: int, max_calls: int) -> None:
    from .tasks import TaskLoader

    _W.update(loader=TaskLoader(traces, solved, code, cfg), url=url, seed=seed, max_calls=max_calls)


def _job(job: dict[str, Any]) -> dict[str, Any]:
    from .folds import other
    from .rollout import http_model_fn, rollout
    from .tasks import z_from_json

    base = {k: job[k] for k in ("instance_id", "fold", "condition", "cand_id")}
    try:
        task, tl, _ = _W["loader"].load(job["trace"])
    except Exception as e:  # noqa: BLE001
        return {**base, "stop": f"load:{type(e).__name__}", "error": str(e)[:300]}
    if task.known_lines is not None:
        return {**base, "stop": "partial_code"}
    fixer = f"fix_{other(job['fold'])}"
    r = rollout(task, tl, z_from_json(job["z"]), _W["loader"].lc, http_model_fn(_W["url"], fixer, _W["seed"]),
                _W["max_calls"])
    return {**base, "fixer": fixer, **r.to_json()}


def build_jobs(weights: list[dict[str, Any]], limit: int, seed: int, done: set[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = sorted(weights, key=lambda r: r["instance_id"])
    if limit and limit < len(rows):
        rows = sorted(random.Random(seed).sample(rows, limit), key=lambda r: r["instance_id"])
    jobs = []
    for w in rows:
        conds = [("T", 0, w["z_teacher"])]
        if not w["z_star_is_teacher"]:
            conds.append(("S", w["z_star"], w["z_star_z"]))
        for c, cid, z in conds:
            if (w["instance_id"], c) not in done:
                jobs.append({"instance_id": w["instance_id"], "fold": w["fold"], "trace": w["trace"],
                             "condition": c, "cand_id": cid, "z": z})
    return jobs


# --------------------------------------------------------------------------- CLI
def add_args_cmd_signal(p: argparse.ArgumentParser) -> None:
    p.add_argument("--weights", required=True, help="weights.jsonl from score.weights")
    p.add_argument("--traces", required=True)
    p.add_argument("--solved", required=True)
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--out", required=True, help="output folder (rollouts.jsonl, signal_report.json)")
    p.add_argument("--fix-A", help="exported fold-A fixer (attempts fold B tasks)")
    p.add_argument("--fix-B", help="exported fold-B fixer (attempts fold A tasks)")
    p.add_argument("--ct", help="the -w4a16-ct checkpoint (not needed with --server-url)")
    p.add_argument("--model", default="e4b", help="model key in the config (sets vLLM overrides)")
    p.add_argument("--server-url", help="a running server that serves fix_A and fix_B (http://host:port/v1)")
    p.add_argument("--limit", type=int, default=0, help="random sample of N tasks (default 0 = all: most power)")
    p.add_argument("--workers", type=int, default=64, help="rollouts in parallel (each a process)")
    p.add_argument("--max-model-calls", type=int, default=8, help="live fixer turns per rollout")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--vllm-arg", action="append", default=[], help="extra vLLM flag, e.g. --vllm-arg=--data-parallel-size=8")
    p.add_argument("--no-parity-check", action="store_true")
    p.add_argument("--report-only", action="store_true", help="recompute the report from rollouts.jsonl")
    p.add_argument("--primary", default="all_hunks", choices=METRICS)
    p.add_argument("--min-gain", type=float, default=0.02)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--tie-band", type=float, default=0.01)
    p.add_argument("--min-token-cut", type=float, default=0.20)


def cmd_signal(args: argparse.Namespace, cfg: dict) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from ..runmeta import write_run_meta
    from .tasks import read_jsonl

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = {w["instance_id"]: w for w in read_jsonl(args.weights)}
    if any(not w.get("trace") for w in weights.values()):
        raise SystemExit(f"{args.weights}: rows without `trace` (rerun score.weights with this version)")
    path = out / "rollouts.jsonl"
    prev = read_jsonl(path) if path.exists() else []
    done = {(r["instance_id"], r["condition"]) for r in prev}
    jobs = [] if args.report_only else build_jobs(list(weights.values()), args.limit, int(cfg["seed"]), done)
    print(json.dumps({"jobs": len(jobs), "already_done": len(done)}), flush=True)
    if jobs:
        server = None
        url = args.server_url
        try:
            if not url:
                if not (args.ct and args.fix_A and args.fix_B):
                    raise SystemExit("--ct, --fix-A and --fix-B are required without --server-url")
                if not args.no_parity_check:
                    from ..score.vllm_scorer import check_parity

                    check_parity(args.fix_A)
                    check_parity(args.fix_B)
                from adk_submission import VllmServer

                from ..eval.run import vllm_config

                vc = vllm_config(args.ct, cfg["models"][args.model]["serve_id"], cfg, tp=args.tp,
                                 gpu_util=args.gpu_memory_utilization, port=args.port, extra_args=args.vllm_arg)
                server = VllmServer(vc, lora_modules={"fix_A": args.fix_A, "fix_B": args.fix_B})
                print("starting vLLM:", " ".join(server.build_cmd()), flush=True)
                server.start()
                url = server.base_url
            init = (args.traces, args.solved, args.code, cfg, url, int(cfg["seed"]), args.max_model_calls)
            with open(path, "a") as f, ProcessPoolExecutor(args.workers, initializer=_init, initargs=init) as pool:
                futs = [pool.submit(_job, j) for j in jobs]
                for k, fu in enumerate(as_completed(futs)):
                    f.write(json.dumps(fu.result()) + "\n")
                    f.flush()
                    if (k + 1) % 25 == 0:
                        print(f"{k + 1}/{len(jobs)} rollouts", flush=True)
        finally:
            if server is not None:
                server.stop()
    rows = [r for r in read_jsonl(path) if r["stop"] != "partial_code" and not r["stop"].startswith("load:")]
    skipped = Counter(r["stop"] for r in read_jsonl(path) if r not in rows)
    th = dict(primary=args.primary, min_gain=args.min_gain, alpha=args.alpha, tie_band=args.tie_band,
              min_token_cut=args.min_token_cut)
    rep = report(rows, weights, **th)
    rep["skipped_tasks"] = dict(skipped)
    (out / "signal_report.json").write_text(json.dumps(rep, indent=2))
    write_run_meta(out, cfg, {"signal": {k: rep[k] for k in ("tasks", "decision", "why")},
                              "fix_A": args.fix_A, "fix_B": args.fix_B, "limit": args.limit})
    m = rep["metrics"][args.primary]
    print(json.dumps({"tasks": rep["tasks"], "z_star_differs": rep["z_star_differs"], args.primary: m,
                      "z_tokens": rep["z_tokens"], "decision": rep["decision"], "why": rep["why"]}, indent=2))
    return 0
