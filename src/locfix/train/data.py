"""Training data: record files → token examples → per-epoch mixture → packed rows.

    sources (files, with weight and repeat)
      └─ tokenize_source: render_record on every record (cached on disk)  → Examples
           └─ epoch_plan: which examples an epoch uses (repeat ≥ 1 copies, fractional sampling)
                └─ pack_rows: examples → rows of ≤ max_tokens tokens (position_ids restart per example)

A row is one micro-batch item. `loss_weights` = turn weight × record weight × source weight;
the trainer normalizes Σ w·CE by Σ w over the whole global batch.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..render.chat import RenderError, render_record


@dataclass
class Source:
    path: str
    weight: float = 1.0          # multiplies the loss weights of every token from this file
    repeat: float = 1.0          # copies per epoch: 2 = twice, 0.5 = a random half (new half each epoch)
    name: str | None = None

    @property
    def label(self) -> str:
        return self.name or Path(self.path).stem


@dataclass
class Example:
    input_ids: np.ndarray        # int32 [T]
    labels: np.ndarray           # int32 [T], -100 where not trained
    weights: np.ndarray          # float32 [T]
    record_id: str
    instance_id: str
    stage: str
    source: int = 0              # index into the source list

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


@dataclass
class SourceReport:
    source: str
    records: int = 0
    examples: int = 0
    too_long: int = 0
    render_errors: int = 0
    no_trained_tokens: int = 0
    tokens: int = 0
    trained_tokens: int = 0
    errors: list[str] = field(default_factory=list)


def _file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _renderer_key(tok, max_seq_len: int) -> str:
    from ..render import chat

    h = hashlib.sha256()
    h.update(Path(chat.__file__).read_bytes())
    h.update((tok.chat_template or "").encode())
    h.update(str(len(tok)).encode())
    h.update(str(max_seq_len).encode())
    return h.hexdigest()[:16]


def tokenize_source(path: str | Path, tok, max_seq_len: int, cache_dir: str | Path | None = None,
                    source_index: int = 0) -> tuple[list[Example], SourceReport]:
    """Render every record of a JSONL file. Records longer than `max_seq_len` are dropped and
    counted; render errors are counted (they should not happen: `locfix data.check-records`)."""
    path = Path(path)
    rep = SourceReport(str(path))
    cache = None
    if cache_dir is not None:
        key = f"{path.stem}.{_file_hash(path)[:16]}.{_renderer_key(tok, max_seq_len)}.pt"
        cache = Path(cache_dir) / key
        if cache.exists():
            blob = torch.load(cache, weights_only=False)
            exs = [Example(**e, source=source_index) for e in blob["examples"]]
            return exs, SourceReport(**blob["report"])
    out: list[Example] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rep.records += 1
            try:
                r = render_record(rec, tok)
            except RenderError as e:
                rep.render_errors += 1
                rep.errors.append(f"{rec.get('id')}: {e}")
                continue
            n = len(r.input_ids)
            if n > max_seq_len:
                rep.too_long += 1
                continue
            if r.n_trained == 0:
                rep.no_trained_tokens += 1
                continue
            out.append(Example(np.asarray(r.input_ids, dtype=np.int32), np.asarray(r.labels, dtype=np.int32),
                               np.asarray(r.weights, dtype=np.float32), rec["id"], rec["instance_id"],
                               rec["stage"], source_index))
            rep.examples += 1
            rep.tokens += n
            rep.trained_tokens += r.n_trained
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        torch.save({"examples": [{k: getattr(e, k) for k in ("input_ids", "labels", "weights", "record_id",
                                                              "instance_id", "stage")} for e in out],
                    "report": rep.__dict__}, tmp)
        tmp.replace(cache)
    return out, rep


def load_sources(sources: list[Source], tok, max_seq_len: int, cache_dir: str | Path | None = None
                 ) -> tuple[list[Example], list[SourceReport]]:
    examples, reports = [], []
    for i, s in enumerate(sources):
        exs, rep = tokenize_source(s.path, tok, max_seq_len, cache_dir, source_index=i)
        examples.extend(exs)
        reports.append(rep)
    return examples, reports


# --------------------------------------------------------------------------- epochs
def epoch_plan(examples: list[Example], sources: list[Source], epoch: int, seed: int) -> list[int]:
    """Example indices used in `epoch`: floor(repeat) copies of every example of a source, plus a
    random subset of size round(frac(repeat)·n) (different each epoch); shuffled."""
    rng = random.Random(f"{seed}:{epoch}")
    by_source: dict[int, list[int]] = {}
    for i, e in enumerate(examples):
        by_source.setdefault(e.source, []).append(i)
    out: list[int] = []
    for si, s in enumerate(sources):
        idx = by_source.get(si, [])
        whole = int(math.floor(s.repeat))
        out.extend(idx * whole)
        frac = s.repeat - whole
        if frac > 0 and idx:
            out.extend(rng.sample(idx, int(round(frac * len(idx)))))
    rng.shuffle(out)
    return out


def pack_rows(indices: list[int], lengths: list[int], max_tokens: int, packing: bool,
              seed: int = 0) -> list[list[int]]:
    """Group examples into rows of at most `max_tokens` tokens.

    `packing=False`: one example per row. `packing=True`: first-fit decreasing (long examples
    first, each placed in the first row with room), then rows shuffled."""
    if not packing:
        return [[i] for i in indices]
    order = sorted(indices, key=lambda i: -lengths[i])
    rows: list[list[int]] = []
    room: list[int] = []
    for i in order:
        n = lengths[i]
        if n > max_tokens:
            raise ValueError(f"example of {n} tokens exceeds max_tokens={max_tokens}")
        for r in range(len(rows)):
            if room[r] >= n:
                rows[r].append(i)
                room[r] -= n
                break
        else:
            rows.append([i])
            room.append(max_tokens - n)
    random.Random(seed).shuffle(rows)
    return rows


@dataclass
class TrainPlan:
    rows: list[list[int]]                 # all epochs, in training order
    epoch_end_rows: list[int]             # row count at the end of each epoch
    dropped_rows: int                     # tail rows dropped so every optimizer step is full
    rows_per_step: int

    @property
    def steps(self) -> int:
        return len(self.rows) // self.rows_per_step

    def epoch_end_steps(self) -> list[int]:
        return [min(r // self.rows_per_step, self.steps) for r in self.epoch_end_rows]


def make_plan(examples: list[Example], sources: list[Source], epochs: int, rows_per_step: int,
              max_tokens: int, packing: bool, seed: int) -> TrainPlan:
    """Rows for every epoch. Each epoch's rows are trimmed to a multiple of `rows_per_step`
    (world size × gradient accumulation × rows per device) so that an epoch ends on an optimizer
    step and every process gets the same number of rows."""
    lengths = [len(e) for e in examples]
    rows: list[list[int]] = []
    ends: list[int] = []
    dropped = 0
    for ep in range(epochs):
        idx = epoch_plan(examples, sources, ep, seed)
        ep_rows = pack_rows(idx, lengths, max_tokens, packing, seed=seed * 1000 + ep)
        keep = len(ep_rows) - len(ep_rows) % rows_per_step
        dropped += len(ep_rows) - keep
        rows.extend(ep_rows[:keep])
        ends.append(len(rows))
    return TrainPlan(rows, ends, dropped, rows_per_step)


def plan_summary(plan: TrainPlan, examples: list[Example], sources: list[Source]) -> dict[str, Any]:
    """Numbers to check before a run: steps, tokens, and how often each task's patch is seen."""
    used = [i for r in plan.rows for i in r]
    visits = Counter(examples[i].instance_id for i in used)
    per_source = Counter(sources[examples[i].source].label for i in used)
    toks = sum(len(examples[i]) for i in used)
    trained = sum(int((examples[i].labels != -100).sum()) for i in used)
    vals = sorted(visits.values())
    return {
        "optimizer_steps": plan.steps, "rows": len(plan.rows), "rows_per_step": plan.rows_per_step,
        "dropped_tail_rows": plan.dropped_rows, "examples_used": len(used),
        "examples_per_source": dict(per_source), "tokens": toks, "trained_tokens": trained,
        "mean_row_tokens": round(toks / max(len(plan.rows), 1)),
        "tasks": len(visits),
        "visits_per_task": {"min": vals[0] if vals else 0, "median": vals[len(vals) // 2] if vals else 0,
                            "max": vals[-1] if vals else 0},
        "epoch_end_steps": plan.epoch_end_steps(),
    }


# --------------------------------------------------------------------------- rows → tensors
class RowDataset(torch.utils.data.Dataset):
    """Each item is one packed row as tensors: input_ids, position_ids (restart at 0 for every
    example), labels, loss_weights (source weight applied), source_ids (per token)."""

    def __init__(self, rows: list[list[int]], examples: list[Example], sources: list[Source]):
        self.rows, self.examples, self.sources = rows, examples, sources

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        ids, pos, lab, w, src = [], [], [], [], []
        for i in self.rows[k]:
            e = self.examples[i]
            n = len(e)
            ids.append(torch.from_numpy(e.input_ids.astype(np.int64)))
            pos.append(torch.arange(n, dtype=torch.long))
            lab.append(torch.from_numpy(e.labels.astype(np.int64)))
            w.append(torch.from_numpy(e.weights) * float(self.sources[e.source].weight))
            src.append(torch.full((n,), e.source, dtype=torch.long))
        return {"input_ids": torch.cat(ids), "position_ids": torch.cat(pos), "labels": torch.cat(lab),
                "loss_weights": torch.cat(w), "source_ids": torch.cat(src)}


def collate_rows(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """One row per micro-batch (packing replaces padding)."""
    if len(items) != 1:
        raise ValueError("use per_device_train_batch_size=1: rows are packed, not padded")
    return {k: v.unsqueeze(0) for k, v in items[0].items()}
