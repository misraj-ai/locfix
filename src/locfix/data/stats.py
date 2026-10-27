"""`locfix data.stats`: statistics to choose the budget, the sequence length and the batching.

Two levels, either or both:

- **traces** (`--traces`, fast, no harness wheels; token counts need `--tokenizer`): tool calls,
  turns, nudges, exploration length (calls before the first edit), parallel calls, teacher time,
  approximate token length (entry texts + a fixed allowance for the tool declarations; within ~1.5%
  of the rendered records on the fixtures). Tables: what each tool-call budget keeps, what each
  maximum length keeps, both together.
- **records** (`--records`, exact, needs `--tokenizer`): rendered length, trained tokens and turns
  per stage, where the tokens go (system / task / tool outputs / thinking / calls), what each
  maximum length keeps per stage, and a **step-time simulation** for multi-GPU training.

Step-time simulation: every optimizer step, each of the P GPUs processes its rows; a micro-step
lasts as long as the slowest GPU's row. Cost of a row of T tokens ≈ T·(1 + T/attn_scale) (the
quadratic term is attention; attn_scale = 131,072 makes attention ~25% of the cost at 32K for
Gemma 4's mostly sliding-window layers). Efficiency = useful cost / (P · Σ slowest), compared for:
records in random order (no packing), length-grouped steps (`train.length_grouping`), and packing
to the maximum length (`train.packing`).

Writes `<out>/stats.md` (the report), `stats.json`, `traces.jsonl`, `records.jsonl`, and PNG
histograms when matplotlib is available.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .traces import Compaction, Nudge, Trace, load_trace

TOOL_DECL_TOKENS = 841          # rendered tool declarations of the single agent (fixtures)
ENTRY_OVERHEAD = 6              # template tokens around each entry
BUDGETS = (16, 20, 24, 28, 30, 32, 36, 40, 50, 100)
LENGTHS = (8192, 12288, 16384, 20480, 24576, 28672, 32768)
EXPLORATION_CAPS = (6, 8, 10, 12, 15, 20)


# --------------------------------------------------------------------------- helpers
def pct(xs: list[float], ps=(50, 75, 90, 95, 99)) -> dict[str, float]:
    if not xs:
        return {}
    a = np.asarray(xs, dtype=float)
    out = {"n": int(a.size), "mean": round(float(a.mean()), 1), "min": float(a.min())}
    out.update({f"p{p}": round(float(np.percentile(a, p)), 1) for p in ps})
    out["max"] = float(a.max())
    return out


def k(n: float) -> str:
    return f"{n / 1024:.0f}K" if n >= 1024 else str(int(n))


def md_table(rows: list[dict[str, Any]], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(none)_\n"
    cols = cols or list(rows[0])
    out = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
    for r in rows:
        out += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return out


def _entry_text(e: dict[str, Any]) -> str:
    t = e.get("type")
    if t == "tool_call":
        return str(e.get("tool", "")) + json.dumps(e.get("args", {}), ensure_ascii=False)
    if t == "tool_response":
        return str(e.get("result") or "")
    c = e.get("content")
    return c if isinstance(c, str) else ""


# --------------------------------------------------------------------------- traces
def trace_row(tr: Trace, dataset: str, tok=None) -> dict[str, Any]:
    calls = list(tr.tool_calls())
    first = tr.first_edit_call()
    before = tr.calls_before_first_edit()
    row: dict[str, Any] = {
        "trace": Path(tr.path).name, "instance_id": tr.instance_id, "dataset": dataset,
        "counted_calls": tr.n_counted_calls, "all_calls": len(calls), "turns": len(tr.turns),
        "parallel_turns": sum(t.is_parallel for t in tr.turns),
        "nudges": sum(isinstance(e, Nudge) for e in tr.events),
        "compactions": sum(isinstance(e, Compaction) for e in tr.events),
        "calls_before_first_edit": sum(1 for c in before if c.counted),
        "has_edit": first is not None, "first_edit_tool": first.tool if first else None,
        "duration_s": round(tr.duration_s or 0.0, 1),
        "marked_entries": sum(1 for e in tr.entries if "train_weight" in e),
    }
    if tok is not None:
        lens = [len(tok(_entry_text(e), add_special_tokens=False)["input_ids"]) + ENTRY_OVERHEAD
                for e in tr.entries]
        row["approx_tokens"] = sum(lens) + TOOL_DECL_TOKENS
        cut = first.entry_index if first is not None else len(tr.entries)
        row["approx_tokens_before_edit"] = sum(lens[:cut]) + TOOL_DECL_TOKENS
        resp = [n for e, n in zip(tr.entries, lens, strict=True) if e.get("type") == "tool_response"]
        row["max_tool_output_tokens"] = max(resp) if resp else 0
        think = [n for e, n in zip(tr.entries, lens, strict=True) if e.get("type") == "thinking"]
        row["thinking_tokens"] = sum(think)
    return row


def trace_rows(traces_dir: str | Path, solved: str | Path | None, tok=None, limit: int = 0
               ) -> tuple[list[dict], list[str]]:
    datasets: dict[str, str] = {}
    if solved:
        for line in open(solved):
            r = json.loads(line)
            datasets[r["instance_id"]] = r.get("verifier", "other")
    rows, errors = [], []
    for i, p in enumerate(sorted(Path(traces_dir).glob("*.json"))):
        if limit and i >= limit:
            break
        try:
            tr = load_trace(p)
        except Exception as e:  # noqa: BLE001 - reported
            errors.append(f"{p.name}: {type(e).__name__}: {e}")
            continue
        rows.append(trace_row(tr, datasets.get(tr.instance_id, "unknown"), tok))
    return rows, errors


def trace_report(rows: list[dict], budgets=BUDGETS, lengths=LENGTHS) -> tuple[str, dict]:
    js: dict[str, Any] = {}
    md = "## 1. Traces\n\n"
    by_ds = Counter(r["dataset"] for r in rows)
    md += f"{len(rows)} traces, {len({r['instance_id'] for r in rows})} tasks; by dataset: " + \
        ", ".join(f"{d} {n}" for d, n in sorted(by_ds.items())) + ".\n\n"

    fields = ["counted_calls", "turns", "calls_before_first_edit", "parallel_turns", "nudges", "duration_s"]
    if rows and "approx_tokens" in rows[0]:
        fields += ["approx_tokens", "approx_tokens_before_edit", "max_tool_output_tokens", "thinking_tokens"]
    dist = {f: pct([r[f] for r in rows]) for f in fields}
    js["distributions"] = dist
    md += "### Distributions\n\n" + md_table([{"quantity": f, **dist[f]} for f in fields],
                                              ["quantity", "n", "mean", "min", "p50", "p75", "p90", "p95", "p99", "max"])
    md += ("\n`counted_calls`: run_command/read_file/edit_file/write_file (what the budget counts). "
           "`calls_before_first_edit`: the localizer's exploration in the two-agent split. "
           "`duration_s`: the teacher's wall time (Kimi K3), not Gemma's.\n\n")

    per_ds = []
    for d in sorted(by_ds):
        sub = [r for r in rows if r["dataset"] == d]
        x = {"dataset": d, "traces": len(sub)}
        for f in ("counted_calls", "turns") + (("approx_tokens",) if "approx_tokens" in rows[0] else ()):
            p = pct([r[f] for r in sub])
            x[f"{f} p50"] = p["p50"]
            x[f"{f} p90"] = p["p90"]
        per_ds.append(x)
    md += "### By dataset\n\n" + md_table(per_ds) + "\n"

    sweep = []
    n = len(rows)
    for b in budgets:
        kept = [r for r in rows if r["counted_calls"] <= b]
        sweep.append({"max_tool_calls": b, "traces kept": len(kept), "kept %": round(100 * len(kept) / max(n, 1), 1),
                      "warning from call": max(b - 10, 1) if b >= 20 else "never"})
    js["budget_sweep"] = sweep
    md += ("### Tool-call budget: what it keeps\n\nA trajectory needing more counted calls than the "
           "budget is dropped from the single-agent records (the two-agent records also add the fixer's "
           "reads of z, see `budget_locfix`). The harness adds `budget_warning` once ≤ 10 calls remain "
           "(budgets ≥ 20).\n\n" + md_table(sweep) + "\n")

    expl = []
    for c in EXPLORATION_CAPS:
        kept = sum(1 for r in rows if r["has_edit"] and r["calls_before_first_edit"] <= c)
        expl.append({"localizer_max_calls": c, "traces kept": kept, "kept %": round(100 * kept / max(n, 1), 1)})
    js["exploration_sweep"] = expl
    md += ("### Exploration length (two-agent split)\n\nTraces whose exploration before the first edit "
           "fits `em.localizer_max_calls`.\n\n" + md_table(expl) + "\n")

    if "approx_tokens" in rows[0]:
        lsweep = []
        tot_tok = sum(r["approx_tokens"] for r in rows)
        for L in lengths:
            kept = [r for r in rows if r["approx_tokens"] <= L]
            lsweep.append({"max_seq_len": k(L), "traces kept": len(kept), "kept %": round(100 * len(kept) / max(n, 1), 1),
                           "dropped": n - len(kept),
                           "tokens kept %": round(100 * sum(r["approx_tokens"] for r in kept) / max(tot_tok, 1), 1)})
        js["length_sweep_approx"] = lsweep
        md += ("### Sequence length (approximate, whole trajectory as one sequence)\n\n"
               "Length of the full single-agent trajectory (one record unless it has a nudge).\n\n"
               + md_table(lsweep) + "\n")
        grid = []
        for b in budgets:
            x: dict[str, Any] = {"max_tool_calls": b}
            for L in lengths:
                x[k(L)] = sum(1 for r in rows if r["counted_calls"] <= b and r["approx_tokens"] <= L)
            grid.append(x)
        js["budget_x_length"] = grid
        md += ("### Budget × length: traces kept\n\n" + md_table(grid) + "\n")
    return md, js


# --------------------------------------------------------------------------- records
def _msg_text(m: dict[str, Any]) -> tuple[str, str, str]:
    """(content, thinking, tool calls) text of a message."""
    c = m.get("content")
    content = c if isinstance(c, str) else "".join(p.get("text", "") for p in (c or []) if isinstance(p, dict))
    thinking = m.get("reasoning") or m.get("reasoning_content") or ""
    calls = "".join(tc["function"]["name"] + tc["function"]["arguments"] for tc in m.get("tool_calls", []))
    return content, thinking, calls


def record_row(rec: dict[str, Any], tok) -> dict[str, Any]:
    from ..render.chat import RenderError, render_record

    meta = rec.get("meta") or {}
    row: dict[str, Any] = {
        "id": rec["id"], "stage": rec["stage"], "role": rec["role"], "instance_id": rec["instance_id"],
        "dataset": (rec.get("source") or {}).get("dataset", "other"), "variant": meta.get("variant"),
        "window": meta.get("window", 0), "n_windows": meta.get("n_windows", 1), "messages": len(rec["messages"]),
        "trained_turns": sum(1 for e in rec["loss"] if e.get("train", True) and e.get("weight", 1) > 0),
        "counted_calls": meta.get("counted_calls", meta.get("n_counted_calls")),
        "requests": len(meta.get("requests") or []),
    }
    try:
        r = render_record(rec, tok)
    except RenderError as e:
        row["render_error"] = str(e)[:200]
        return row
    row["tokens"] = len(r.input_ids)
    row["trained_tokens"] = r.n_trained
    row["max_trained_turn_tokens"] = max((len(tok(r.text[s:e], add_special_tokens=False)["input_ids"])
                                          for s, e, _ in r.target_spans), default=0)
    cat = defaultdict(int)
    tool_max = 0
    count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"]) if s else 0  # noqa: E731
    for i, m in enumerate(rec["messages"]):
        content, thinking, calls = _msg_text(m)
        role = m["role"]
        if role == "system":
            cat["system"] += count(content)
        elif role == "user":
            cat["task prompt" if i <= 1 else "nudges/summaries"] += count(content)
        elif role == "tool":
            n = count(content)
            cat["tool outputs"] += n
            tool_max = max(tool_max, n)
        else:
            cat["thinking"] += count(thinking)
            cat["tool calls"] += count(calls)
            cat["final text"] += count(content)
    cat["tool declarations"] = count(json.dumps(rec.get("tools", []), ensure_ascii=False))
    row["max_tool_output_tokens"] = tool_max
    row["categories"] = dict(cat)
    return row


def record_rows(paths: list[str | Path], tok, limit: int = 0) -> list[dict]:
    files: list[Path] = []
    for x in paths:
        p = Path(x)
        files.extend(sorted(p.rglob("*.jsonl")) if p.is_dir() else [p])
    rows = []
    for f in files:
        with open(f) as fh:
            for i, line in enumerate(fh):
                if limit and i >= limit:
                    break
                if line.strip():
                    row = record_row(json.loads(line), tok)
                    row["file"] = f.name
                    rows.append(row)
    return rows


def cost(T: float, attn_scale: float) -> float:
    return T * (1 + T / attn_scale)


def step_efficiency(lengths: list[int], world: int, accum: int, attn_scale: float, mode: str,
                    max_tokens: int, seed: int = 0) -> dict[str, Any]:
    """Simulated GPU utilization of one epoch (see module docstring)."""
    rng = random.Random(seed)
    if mode == "packed":
        order = sorted(lengths, reverse=True)
        rows: list[int] = []
        room: list[int] = []
        for n in order:
            for i in range(len(rows)):
                if room[i] >= n:
                    rows[i] += n
                    room[i] -= n
                    break
            else:
                rows.append(n)
                room.append(max_tokens - n)
        rng.shuffle(rows)
    else:
        rows = list(lengths)
        rng.shuffle(rows)
    per_step = world * accum
    rows = rows[: len(rows) - len(rows) % per_step] if len(rows) >= per_step else rows
    if mode == "grouped":
        rows.sort()
        steps = [rows[i:i + per_step] for i in range(0, len(rows), per_step)]
        rng.shuffle(steps)
        rows = [x for s in steps for x in s]
    useful = sum(cost(t, attn_scale) for t in rows)
    wall = 0.0
    for i in range(0, len(rows), world):
        wall += max(cost(t, attn_scale) for t in rows[i:i + world])
    eff = useful / (world * wall) if wall else 0.0
    return {"mode": mode, "rows": len(rows), "optimizer_steps": len(rows) // per_step if per_step else 0,
            "mean_row_tokens": int(sum(rows) / max(len(rows), 1)), "gpu_efficiency %": round(100 * eff, 1),
            "relative epoch time": round(wall / max(useful / world, 1e-9), 3)}


def record_report(rows: list[dict], lengths=LENGTHS, world: int = 8, global_batch: int = 16,
                  attn_scale: float = 131072, tokens_per_s: float | None = None) -> tuple[str, dict]:
    js: dict[str, Any] = {}
    ok = [r for r in rows if "tokens" in r]
    errs = [r for r in rows if "render_error" in r]
    md = "## 2. Records (exact, rendered with the Gemma 4 chat template)\n\n"
    md += f"{len(rows)} records, {len(errs)} render errors.\n\n"
    stages = sorted({r["stage"] for r in ok})
    st_rows = []
    for s in stages:
        sub = [r for r in ok if r["stage"] == s]
        tp = pct([r["tokens"] for r in sub])
        tr = sum(r["trained_tokens"] for r in sub)
        tot = sum(r["tokens"] for r in sub)
        st_rows.append({"stage": s, "role": sub[0]["role"], "records": len(sub),
                        "tasks": len({r["instance_id"] for r in sub}),
                        "tokens p50": k(tp["p50"]), "p90": k(tp["p90"]), "p99": k(tp["p99"]), "max": k(tp["max"]),
                        "total tokens": f"{tot / 1e6:.1f}M", "trained %": round(100 * tr / max(tot, 1), 1),
                        "trained turns p50": pct([r["trained_turns"] for r in sub])["p50"],
                        "with requests": sum(1 for r in sub if r["requests"])})
    js["stages"] = st_rows
    md += "### Per stage\n\n" + md_table(st_rows) + "\n"

    lsw = []
    for s in stages:
        sub = [r for r in ok if r["stage"] == s]
        tot_tr = sum(r["trained_tokens"] for r in sub)
        x: dict[str, Any] = {"stage": s}
        for L in lengths:
            kept = [r for r in sub if r["tokens"] <= L]
            x[k(L)] = f"{100 * len(kept) / len(sub):.0f}% / " \
                      f"{100 * sum(r['trained_tokens'] for r in kept) / max(tot_tr, 1):.0f}% (−{len(sub) - len(kept)})"
        lsw.append(x)
    js["length_sweep"] = lsw
    md += ("### Maximum length: records kept per stage\n\nEach cell: records kept % / trained tokens kept % "
           "(records dropped).\n\n" + md_table(lsw) + "\n")

    cats = defaultdict(int)
    for r in ok:
        for c, n in r.get("categories", {}).items():
            cats[c] += n
    tot = sum(cats.values())
    cat_rows = [{"part": c, "tokens": f"{n / 1e6:.2f}M", "share %": round(100 * n / max(tot, 1), 1)}
                for c, n in sorted(cats.items(), key=lambda x: -x[1])]
    js["token_categories"] = cat_rows
    md += ("### Where the tokens go (all records)\n\nThinking, tool calls and final text are trained "
           "(except zero-weight turns); the rest is context.\n\n" + md_table(cat_rows) + "\n")
    md += "### Largest single items\n\n" + md_table([
        {"quantity": "tool output (tokens)", **pct([r["max_tool_output_tokens"] for r in ok])},
        {"quantity": "trained turn (tokens)", **pct([r["max_trained_turn_tokens"] for r in ok])},
    ], ["quantity", "p50", "p90", "p99", "max"]) + "\n"

    accum = max(global_batch // world, 1)
    sim = []
    for s_group in [("all", ok)] + [(s, [r for r in ok if r["stage"] == s]) for s in stages]:
        name, sub = s_group
        for L in (24576, 32768):
            lens = [r["tokens"] for r in sub if r["tokens"] <= L]
            if len(lens) < world * accum:
                continue
            for mode in ("random", "grouped", "packed"):
                res = step_efficiency(lens, world, accum, attn_scale, mode, L)
                if tokens_per_s:
                    res["epoch hours"] = round(sum(lens) / (tokens_per_s * world) * res["relative epoch time"] / 3600, 2)
                sim.append({"records": name, "max_seq_len": k(L), **res})
    js["step_simulation"] = sim
    md += (f"### Multi-GPU step time ({world} GPUs, global batch {global_batch} rows, accumulation {accum})\n\n"
           "`gpu_efficiency`: share of GPU time doing useful work when each micro-step waits for the "
           "longest row. `random`: one record per row, random order (current default). `grouped`: "
           "records of similar length in the same step (`train.length_grouping: true`). `packed`: "
           "records packed into rows of up to max_seq_len tokens (`train.packing: true`; fewer, longer "
           "rows, so fewer optimizer steps per epoch at the same global batch; with sdpa the attention of a "
           "packed row is computed over the whole row, so its cost is quadratic in the row length). "
           "`relative epoch time`: wall time ÷ ideal (1.0 = no waiting)."
           + (f" `epoch hours`: at {tokens_per_s:,.0f} tokens/s per GPU (measure it on the first run)." if tokens_per_s else "")
           + "\n\n" + md_table(sim) + "\n")
    return md, js


# --------------------------------------------------------------------------- plots
def plots(out: Path, trows: list[dict], rrows: list[dict]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made = []

    def hist(values, title, fname, xlabel, lines=()):
        if not values:
            return
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(values, bins=60, color="#4c78a8")
        for x in lines:
            ax.axvline(x, color="#e45756", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=120)
        plt.close(fig)
        made.append(fname)

    if trows:
        hist([r["counted_calls"] for r in trows], "Counted tool calls per trace", "calls.png", "calls", (20, 30, 40))
        hist([r["calls_before_first_edit"] for r in trows], "Calls before the first edit", "exploration.png",
             "calls", (12,))
        if "approx_tokens" in trows[0]:
            hist([r["approx_tokens"] for r in trows], "Trajectory length (approx. tokens)", "trace_tokens.png",
                 "tokens", (24576, 32768))
    for s in sorted({r["stage"] for r in rrows if "tokens" in r}):
        hist([r["tokens"] for r in rrows if r.get("stage") == s and "tokens" in r], f"Record length: {s}",
             f"records_{s}.png", "tokens", (24576, 32768))
    return made


# --------------------------------------------------------------------------- CLI
def add_args_cmd_stats(p: argparse.ArgumentParser) -> None:
    p.add_argument("--traces", help="folder of traces (cleaned_v2/traces)")
    p.add_argument("--solved", help="solved.jsonl (adds the dataset of each trace)")
    p.add_argument("--records", nargs="*", default=[], help="record files or folders (exact token statistics)")
    p.add_argument("--tokenizer", help="Gemma 4 tokenizer folder (token counts)")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--global-batch", type=int, default=16)
    p.add_argument("--attn-scale", type=float, default=131072,
                   help="T at which attention cost equals the linear cost (step-time model)")
    p.add_argument("--tokens-per-s", type=float, help="training tokens/s per GPU, to estimate epoch hours")
    p.add_argument("--limit", type=int, default=0, help="first N traces / records per file (quick look)")


def cmd_stats(args: argparse.Namespace, cfg: dict) -> int:
    if not args.traces and not args.records:
        raise SystemExit("give --traces and/or --records")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = None
    if args.tokenizer:
        from ..render.chat import load_tokenizer

        tok = load_tokenizer(args.tokenizer)
    md = "# Data statistics\n\n"
    js: dict[str, Any] = {}
    trows: list[dict] = []
    rrows: list[dict] = []
    if args.traces:
        trows, errors = trace_rows(args.traces, args.solved, tok, args.limit)
        with open(out / "traces.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in trows)
        if errors:
            md += f"**{len(errors)} traces failed to load** (first: {errors[0]})\n\n"
        t_md, js["traces"] = trace_report(trows)
        md += t_md
    if args.records:
        if tok is None:
            raise SystemExit("--records needs --tokenizer")
        rrows = record_rows(args.records, tok, args.limit)
        with open(out / "records.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rrows)
        r_md, js["records"] = record_report(rrows, world=args.world_size, global_batch=args.global_batch,
                                            attn_scale=args.attn_scale, tokens_per_s=args.tokens_per_s)
        md += r_md
    figs = plots(out, trows, rrows)
    if figs:
        md += "## Figures\n\n" + "".join(f"![{f}]({f})\n\n" for f in figs)
    (out / "stats.md").write_text(md)
    (out / "stats.json").write_text(json.dumps(js, indent=2, default=str))
    print(md)
    print(f"\nwritten: {out / 'stats.md'}")
    return 0

