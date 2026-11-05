"""`locfix em.r1-records`: Round 1 localizer records from the Round 0 weights (spec 06 §7, M step).

For every task, the localizer is trained on its own candidates z_k, weighted by w(z_k)
(`score.weights`), plus the teacher label z_T as an anchor:

    keep        the top `--top-m` candidates by w(z) with w(z) >= `--min-weight`, renormalized to 1
    anchor      z_T with weight β (`em.beta`, 0.5); merged with z_T's own weight if z_T is kept
    normalize   all weights of the task divided by (1 + β): every task carries total weight 1, as
                in Round 0 and Stage 2 (same loss scale, same per-task influence)

Soft vs hard EM is the weights file: `score.weights` with `--ess-target 2` (soft, the default) or
`--ess-target 1` (hard: all weight on the best candidate). Tasks without weights (not scored, e.g.
an unfinished scoring shard) keep their Round 0 record with weight 1 (z_T only), unless
`--scored-only`.

A record is the task's s3_loc_r0 record (the teacher's exploration, then the answer) with only the
final answer's text replaced by z_k. That is exactly what replaying the localizer with z_k gives:
the exploration and the answer's thought do not depend on z (`localizer_session`; checked by a test
against the s2_loc_edit records, which are the same replay with another z). The record weight scales
every trained turn of the record (`render.chat.render_record`): the exploration turns of a task are
trained with total weight 1 across its records, the answers with the EM weights.

    locfix em.r1-records --loc-records records/locfix/s3_loc_r0.jsonl \
        --weights work/s3r0/weights/weights.jsonl --out records/s3r1/soft/s3_loc_r1.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..data.locations import Location
from ..data.zloc import z_to_json

STAGE = "s3_loc"


def z_from_json(z: list[dict[str, Any]]) -> list[Location]:
    # (local copy of em.tasks.z_from_json: this command runs without the harness wheels)
    return [Location(e["path"], e["symbol"], int(e["lines"][0]), int(e["lines"][1]), e.get("role", "context"))
            for e in z]


def task_weights(row: dict[str, Any], beta: float, top_m: int, min_weight: float) -> list[dict[str, Any]]:
    """[{cand_id, source, z, w, anchor}] for one task; the weights sum to 1."""
    cands = sorted(row["candidates"], key=lambda c: (-c["w_z"], c["cand_id"]))
    keep = [c for c in cands if c["w_z"] >= min_weight][:top_m] or cands[:1]
    tot = sum(c["w_z"] for c in keep)
    out = {c["cand_id"]: {"cand_id": c["cand_id"], "source": c["source"], "z": c["z"],
                          "w": c["w_z"] / tot, "anchor": False} for c in keep}
    zt = row.get("z_teacher") or next(c["z"] for c in row["candidates"] if c["cand_id"] == 0)
    if 0 in out:
        out[0]["w"] += beta
        out[0]["anchor"] = True
    else:
        out[0] = {"cand_id": 0, "source": "teacher", "z": zt, "w": beta, "anchor": True}
    norm = 1.0 + beta
    res = sorted(out.values(), key=lambda d: d["cand_id"])
    for d in res:
        d["w"] = d["w"] / norm
    return res


def with_answer(rec: dict[str, Any], z_json: list[dict[str, Any]], cand: dict[str, Any], tag: str,
                round_: int) -> dict[str, Any]:
    """The record with its final answer replaced by z (and weight, ids, meta for round `round_`)."""
    r = copy.deepcopy(rec)
    last = r["messages"][-1]
    if last.get("role") != "assistant" or last.get("tool_calls"):
        raise ValueError(f"{rec['id']}: last message is not the localizer's text answer")
    if not any(e["message_index"] == len(r["messages"]) - 1 and e.get("train", True) for e in r["loss"]):
        raise ValueError(f"{rec['id']}: the answer turn is not trained")
    last["content"] = z_to_json(z_from_json(z_json))
    r["id"] = rec["id"].replace(f"{rec['stage']}__", f"{STAGE}__", 1) + f"__r{round_}{tag}"
    r["stage"] = STAGE
    r["round"] = round_
    r["z"] = z_json
    r["weight"] = round(float(cand["w"]), 8)
    r["meta"] = {**rec.get("meta", {}), "z": z_json, "variant": f"r{round_}:{cand['source']}",
                 "em": {"round": round_, "cand_id": cand["cand_id"], "source": cand["source"],
                        "anchor": cand["anchor"], "weight": r["weight"]}}
    return r


def build(loc_records: str | Path, weights: str | Path, out: str | Path, *, beta: float, top_m: int,
          min_weight: float, scored_only: bool = False, round_: int = 1) -> dict[str, Any]:
    with open(weights) as f:
        W = {w["instance_id"]: w for w in map(json.loads, filter(str.strip, f))}
    n = Counter()
    per_task: list[int] = []
    anchor_w: list[float] = []
    srcs: Counter = Counter()
    seen: set[str] = set()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(loc_records) as fi, open(out, "w") as fo:
        for line in fi:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("stage") != "s3_loc_r0":
                raise ValueError(f"{rec.get('id')}: stage {rec.get('stage')}, expected s3_loc_r0")
            m = rec.get("meta") or {}
            iid = rec["instance_id"]
            last_window = m.get("window", 0) == m.get("n_windows", 1) - 1
            if not last_window:                 # exploration-only window (no answer): unchanged
                if scored_only and iid not in W:
                    continue
                fo.write(json.dumps({**rec, "id": rec["id"].replace("s3_loc_r0__", f"{STAGE}__", 1) + f"__r{round_}",
                                     "stage": STAGE, "round": round_}) + "\n")
                n["exploration_windows"] += 1
                continue
            if iid in seen:                     # one answer record per task (first trace), as in Stage 3
                n["extra_traces_skipped"] += 1
                continue
            seen.add(iid)
            w = W.get(iid)
            if w is None:
                if scored_only:
                    n["tasks_unscored_dropped"] += 1
                    continue
                cands = [{"cand_id": 0, "source": "teacher", "z": rec["z"], "w": 1.0, "anchor": True}]
                n["tasks_unscored_anchor_only"] += 1
            else:
                if w["z_teacher"] is not None and w["z_teacher"] != rec["z"]:
                    raise ValueError(f"{iid}: the weights' z_T differs from the record's z "
                                     "(weights built from other records)")
                cands = task_weights(w, beta, top_m, min_weight)
                n["tasks_weighted"] += 1
            per_task.append(len(cands))
            anchor_w.append(sum(c["w"] for c in cands if c["anchor"]))
            for c in cands:
                tag = "anchor" if c["cand_id"] == 0 else f"z{c['cand_id']}"
                fo.write(json.dumps(with_answer(rec, c["z"], c, tag, round_), ensure_ascii=False) + "\n")
                n["records"] += 1
                srcs[c["source"]] += 1
    summary = {"tasks": len(per_task), **dict(n), "records_per_task_mean": round(sum(per_task) / max(1, len(per_task)), 3),
               "anchor_weight_mean": round(sum(anchor_w) / max(1, len(anchor_w)), 4), "records_by_source": dict(srcs),
               "beta": beta, "top_m": top_m, "min_weight": min_weight, "weights": str(weights)}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- CLI
def add_args_cmd_r1(p: argparse.ArgumentParser) -> None:
    p.add_argument("--loc-records", required=True, help="training s3_loc_r0.jsonl (records/locfix/s3_loc_r0.jsonl)")
    p.add_argument("--weights", required=True, help="weights.jsonl from score.weights (ESS 2: soft, ESS 1: hard)")
    p.add_argument("--out", required=True, help="output records file, e.g. records/s3r1/soft/s3_loc_r1.jsonl")
    p.add_argument("--beta", type=float, help="anchor weight of z_T (default em.beta)")
    p.add_argument("--top-m", type=int, help="candidates kept per task (default em.resample_m)")
    p.add_argument("--min-weight", type=float, default=0.01, help="drop candidates with w(z) below this")
    p.add_argument("--scored-only", action="store_true", help="drop tasks without weights (default: z_T only)")


def cmd_r1(args: argparse.Namespace, cfg: dict) -> int:
    em = cfg["em"]
    s = build(args.loc_records, args.weights, args.out,
              beta=float(args.beta if args.beta is not None else em["beta"]),
              top_m=int(args.top_m if args.top_m is not None else em["resample_m"]),
              min_weight=args.min_weight, scored_only=args.scored_only)
    print(json.dumps(s, indent=2))
    return 0
