"""The Stage 3 task list and task loading, shared by every Stage 3 step.

Stage 3 works on the tasks of the round-0 localizer records (`s3_loc_r0.jsonl`, training split):
each such record holds the localizer's final request (the teacher's exploration, then the answer
z_T = L_edit ∪ L_view, pruned). From it we take:

- the **candidate prompt**: the record's messages without its last (answer) message, its tools and
  request parameters: exactly the context the localizer was trained to answer in;
- the trace file name (`meta.trace`) and the fold (from `folds.json`).

Tasks are loaded (trace + code + teacher patch) with the same `SolvedData` and settings as
`data.locfix-records`, and the same `LocFixConfig`, so every Stage 3 sequence is built by the same
replay as the training records.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.loc_fix_records import LocFixConfig, Task, TaskLocations, Variant, solved_variants, task_locations
from ..data.locations import Location
from ..harness.prompts import Budget


@dataclass
class Stage3Task:
    instance_id: str
    trace: str                                  # trace file name (meta.trace)
    fold: str
    prompt: dict[str, Any]                      # {"messages", "tools", "request_params"} of the localizer's answer turn
    z_teacher: list[dict[str, Any]]             # z of the record (JSON form)
    record_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def localizer_prompt(record: dict[str, Any]) -> dict[str, Any]:
    """The request the localizer answers with z: the record's messages minus the final answer."""
    msgs = record["messages"]
    if msgs[-1]["role"] != "assistant" or msgs[-1].get("tool_calls"):
        raise ValueError(f"{record['id']}: last message is not the localizer's text answer")
    return {"messages": msgs[:-1], "tools": record.get("tools") or [],
            "request_params": (record.get("meta") or {}).get("request_params", {})}


def read_stage3_tasks(loc_records: str | Path, folds: dict[str, str]) -> list[Stage3Task]:
    """One task per instance id (the first trace that has a round-0 localizer record), in file order.
    Only records whose last window holds the answer are used (a localizer stream is one window)."""
    out: dict[str, Stage3Task] = {}
    with open(loc_records) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("stage") != "s3_loc_r0":
                raise ValueError(f"{loc_records}: record {r.get('id')} has stage {r.get('stage')}, expected s3_loc_r0")
            m = r.get("meta") or {}
            if m.get("window", 0) != m.get("n_windows", 1) - 1:
                continue
            iid = r["instance_id"]
            if iid in out:
                continue
            if iid not in folds:
                raise ValueError(f"{iid} is not in folds.json (run em.fold-split on the same records)")
            out[iid] = Stage3Task(iid, m["trace"], folds[iid], localizer_prompt(r), r["z"], r["id"])
    return list(out.values())


def write_jsonl(path: str | Path, rows) -> int:
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def read_jsonl_many(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Rows of several files (e.g. the --shard outputs of one step)."""
    rows: list[dict[str, Any]] = []
    for p in paths:
        rows += read_jsonl(p)
    return rows


# --------------------------------------------------------------------------- loading
def locfix_config(cfg: dict[str, Any]) -> LocFixConfig:
    """The LocFixConfig of `data.locfix-records` (cmd_locfix_records), built from the same keys."""
    from ..harness.budget import time_scale_from_cfg

    b = cfg["budget_locfix"]
    return LocFixConfig(
        model_alias=cfg["models"]["31b"]["serve_id"],
        budget=Budget(b["max_time_minutes"], b["max_tool_calls"], b["max_turns"]),
        compaction=cfg["harness"]["compaction"] if cfg["data"].get("replay_compaction") else None,
        layout="segments" if cfg["data"].get("training_layout", "auto") in ("auto", "one_sequence") else "per_turn",
        em=cfg["em"], seed=cfg["seed"], time_scale=time_scale_from_cfg(cfg, "budget_locfix"),
    )


class TaskLoader:
    """Task (trace, base-commit code, teacher patch) for a trace file name, as the record builder
    loads it."""

    def __init__(self, traces: str | Path, solved: str | Path, code_dirs: list[str], cfg: dict[str, Any]):
        from ..data.dataset import SolvedData

        self.data = SolvedData(traces=traces, solved=solved, code_dirs=code_dirs,
                               exclude_repos=cfg["data"].get("exclude_repos", []))
        self.cfg = cfg
        self.lc = locfix_config(cfg)

    def load(self, trace_name: str) -> tuple[Task, TaskLocations, str]:
        task, info = self.data.task(self.data.traces_dir / trace_name)
        return task, task_locations(task), info.source


def variants_of(task: Task, tl: TaskLocations, lc: LocFixConfig) -> dict[str, Variant]:
    """The record builder's variants (same seeded rng): gold (= z_T), edit_only, view_only,
    gold_minus_one."""
    import random

    rng = random.Random(f"{lc.seed}:{task.instance_id}")
    return {v.name: v for v in solved_variants(tl, task.files, lc.em, rng)}


def z_from_json(z: list[dict[str, Any]]) -> list[Location]:
    return [Location(e["path"], e["symbol"], int(e["lines"][0]), int(e["lines"][1]), e.get("role", "context"))
            for e in z]
