"""Shared options of the record builders (`data.agent-records`, `data.locfix-records`).

- `--shard K/N`: build only the traces whose index (in sorted file order) is K mod N, so N
  processes can build one data set in parallel; the outputs are concatenated afterwards
  (`locfix data.merge-records`). Results do not depend on the sharding: every random choice is
  seeded per instance.
- Length limit (`data.max_trace_tokens`, needs `--tokenizer`): every record is rendered with the
  Gemma 4 chat template (the trainer's renderer) and its token count stored in
  `meta.n_tokens`; the single-agent builder drops a whole trajectory if any of its records is
  longer than the limit.
- Task list: the single-agent builder writes the trace files it kept (`<out>.tasks.txt`); the
  two-agent builder can be restricted to them (`--tasks`), so both methods train on the same tasks.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_shard(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--shard must look like K/N, got {text!r}") from e
    if n < 1 or not 0 <= k < n:
        raise argparse.ArgumentTypeError(f"--shard K/N needs 0 <= K < N, got {text!r}")
    return k, n


def in_shard(index: int, shard: tuple[int, int] | None) -> bool:
    return shard is None or index % shard[1] == shard[0]


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tokenizer", help="Gemma 4 tokenizer folder: store meta.n_tokens and apply "
                   "data.max_trace_tokens (required when that limit is set)")
    p.add_argument("--shard", type=parse_shard, default=None,
                   help="K/N: build only traces with index %% N == K (run N processes, then data.merge-records)")


class Lengths:
    """Token counts of records, rendered exactly as the trainer renders them."""

    def __init__(self, tokenizer_dir: str | None, limit: int | None):
        self.limit = int(limit) if limit else None
        if self.limit and not tokenizer_dir:
            raise SystemExit("data.max_trace_tokens is set: pass --tokenizer <Gemma 4 tokenizer folder> "
                             "(or --set data.max_trace_tokens=null to keep every length)")
        self.tok = None
        if tokenizer_dir:
            from ..render.chat import load_tokenizer

            self.tok = load_tokenizer(tokenizer_dir)

    def annotate(self, records: list[dict[str, Any]]) -> int:
        """Set meta.n_tokens on every record; return the longest (0 without a tokenizer)."""
        if self.tok is None:
            return 0
        from ..render.chat import render_record

        longest = 0
        for r in records:
            n = len(render_record(r, self.tok).input_ids)
            r.setdefault("meta", {})["n_tokens"] = n
            longest = max(longest, n)
        return longest

    def too_long(self, n_tokens: int) -> bool:
        return self.limit is not None and n_tokens > self.limit


def tasks_file_for(out: Path) -> Path:
    """`work/records/s1_agent.jsonl` -> `work/records/s1_agent.tasks.txt`."""
    return out.with_suffix(".tasks.txt")


def read_tasks(path: str | Path) -> set[str]:
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


# --------------------------------------------------------------------------- merge shards
def add_args_cmd_merge_records(p: argparse.ArgumentParser) -> None:
    p.add_argument("shards", nargs="+", help="shard output folders, in order (each with *.jsonl and run_meta.json)")
    p.add_argument("--out", required=True, help="merged folder: one JSONL per file name, summed counts")


def cmd_merge_records(args: argparse.Namespace, cfg: dict) -> int:
    """Concatenate the shards of `data.agent-records` / `data.locfix-records` (same file names in
    every shard folder) and sum their counts. Raises if a shard is missing a run_meta.json (a shard
    that did not finish) or if two shards hold the same record id."""
    import json

    from ..runmeta import write_run_meta

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    n_lines: dict[str, int] = {}
    tasks: dict[str, list[str]] = {}
    shards = [Path(s) for s in args.shards]
    for d in shards:
        meta_path = d / "run_meta.json"
        if not meta_path.exists():
            raise SystemExit(f"{d}: no run_meta.json (shard not finished?)")
        for k, v in json.loads(meta_path.read_text()).get("counts", {}).items():
            if isinstance(v, int):
                counts[k] = counts.get(k, 0) + v
        for t in sorted(d.glob("*.tasks.txt")):
            tasks.setdefault(t.name, []).extend(sorted(read_tasks(t)))
    names = sorted({f.name for d in shards for f in d.glob("*.jsonl")})
    for name in names:
        with open(out / name, "w") as fo:
            for d in shards:
                f = d / name
                if not f.exists():
                    continue
                with open(f) as fi:
                    for line in fi:
                        if not line.strip():
                            continue
                        rid = json.loads(line)["id"]
                        if rid in seen_ids:
                            raise SystemExit(f"record id {rid} appears in two shards: overlapping --shard values?")
                        seen_ids.add(rid)
                        fo.write(line if line.endswith("\n") else line + "\n")
                        n_lines[name] = n_lines.get(name, 0) + 1
    for name, ids in tasks.items():
        (out / name).write_text("".join(f"{t}\n" for t in sorted(ids)))
    report = {"counts": counts, "files": n_lines, "shards": [str(d) for d in shards],
              "tasks_files": {k: len(v) for k, v in tasks.items()}}
    write_run_meta(out, cfg, report)
    (out / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- train / val split
def is_val(instance_id: str, fraction: float, seed: int) -> bool:
    """Deterministic, per task: every record of a task (all stages, all attempts) lands on the same
    side, in every record file and every rebuild."""
    import hashlib

    h = int.from_bytes(hashlib.sha256(f"{seed}:split:{instance_id}".encode()).digest()[:8], "big")
    return h / 2**64 < fraction


def add_args_cmd_split(p: argparse.ArgumentParser) -> None:
    p.add_argument("inputs", nargs="+", help="record files (-> <out>/<name>) or folders (-> <out>/<folder>/<name>)")
    p.add_argument("--out", required=True, help="training files under <out>/, validation files under <out>/val/")
    p.add_argument("--fraction", type=float, help="validation share of tasks (default data.val_fraction)")


def cmd_split(args: argparse.Namespace, cfg: dict) -> int:
    """Split record files into training and validation files by task (`data.val_fraction`, seeded).
    The layout matches the run configs: `<out>/s1_agent.jsonl`, `<out>/locfix/s2_fix_gold.jsonl`,
    `<out>/val/s1_agent.jsonl`, `<out>/val/locfix/...`."""
    import json

    from ..runmeta import write_run_meta

    frac = args.fraction if args.fraction is not None else float(cfg["data"]["val_fraction"])
    seed = int(cfg["seed"])
    out = Path(args.out)
    jobs: list[tuple[Path, Path]] = []
    for x in args.inputs:
        p = Path(x)
        if p.is_dir():
            jobs += [(f, Path(p.name) / f.name) for f in sorted(p.glob("*.jsonl"))]
        elif p.is_file():
            jobs.append((p, Path(p.name)))
        else:
            raise SystemExit(f"{p}: not found")
    report: dict[str, dict[str, int]] = {}
    val_tasks: set[str] = set()
    all_tasks: set[str] = set()
    for src, rel in jobs:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / "val" / rel).parent.mkdir(parents=True, exist_ok=True)
        n = {"train": 0, "val": 0}
        with open(src) as fi, open(out / rel, "w") as ft, open(out / "val" / rel, "w") as fv:
            for line in fi:
                if not line.strip():
                    continue
                iid = json.loads(line)["instance_id"]
                all_tasks.add(iid)
                side = "val" if is_val(iid, frac, seed) else "train"
                if side == "val":
                    val_tasks.add(iid)
                (fv if side == "val" else ft).write(line if line.endswith("\n") else line + "\n")
                n[side] += 1
        report[str(rel)] = n
    summary = {"fraction": frac, "seed": seed, "tasks": len(all_tasks), "val_tasks": len(val_tasks),
               "files": report}
    (out / "val" / "val_tasks.txt").write_text("".join(f"{t}\n" for t in sorted(val_tasks)))
    write_run_meta(out, cfg, {"split": summary})
    print(json.dumps(summary, indent=2))
    return 0
