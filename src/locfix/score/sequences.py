"""`locfix score.prepare`: the scoring sequences of Stage 3 (spec 06 §3), on CPU.

For every task and candidate z_k, and for each request mode a:

    a = none     the fixer gets z_k only (the localizer call, then reads of z_k);
    a = recover  as in the S2b records, the fixer also reads what z_k misses of N = L_edit ∪ L_view
                 (only when z_k misses something; the request steps are the record builder's own);

the sequence is replayed through the real agent stack exactly like the training records
(`fixer_script` / `localizer_session` of `data.loc_fix_records`, same budget walker, same
`LocFixConfig`), with the teacher's trajectory after its first edit replaced by the **canonical edit
calls** of the verified patch y* (diff -> `edit_file`, spec 04 §8), one call per turn, each
preceded by a fixed thought (`EDIT_THINKING`, identical for every candidate, so it carries no
information about z). The calls' outputs are the harness's own (`LocalWorkspace`).

Output, per sequence: token ids (training renderer), the positions of the y*₊ tokens (`spans.py`),
`n_z_tokens` (tokens the reads of z_k add to the context), `n_ctx_tokens` (all context added after
the localizer's answer: reads of z_k + requests), and `n_requests` (tool calls the fixer makes for
what z_k misses: |a_k| of spec 06 §5). A candidate whose sequence cannot
be built (unknown code, budget) is listed with the reason.

    locfix score.prepare --loc-records records/locfix/s3_loc_r0.jsonl --folds records/folds \
        --samples work/s3r0/samples/*.jsonl --traces .../traces --solved .../solved.jsonl \
        --code .../code --tokenizer <ct> --out work/s3r0/prepare/part0 --shard 0/4 --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..data.loc_fix_records import (
    FIXER,
    LOCALIZER,
    LocFixConfig,
    Task,
    TaskLocations,
    TaskSkip,
    fixer_script,
    localizer_session,
)
from ..data.locations import Location, normalize_path
from ..data.zloc import minus
from ..harness.budget import BudgetExceeded, BudgetWalker
from ..harness.prompts import parse_task_prompt, rebuild_with_budget
from ..harness.replay import ScriptCall, ScriptTurn, replay
from .spans import SpanError, added_char_spans, tokens_overlapping

EDIT_THINKING = ("I have the code I need. I'll make the change.", "Next change.")
FINAL_TEXT = "Done."
MODES = ("none", "recover")


class RecordsMismatch(RuntimeError):
    """The replay does not reproduce the localizer prompt stored in the records: the records were
    built with other agent files (agents/locfix) or settings than this run uses."""


class SequenceError(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


# --------------------------------------------------------------------------- canonical edits
def canonical_edits(task: Task) -> list[tuple[str, dict[str, Any], str]]:
    """(tool, args, harness output) of the verified patch as edit calls, in order."""
    from ..data.edits import PatchError, patch_to_edit_calls
    from ..harness.local_sandbox import LocalWorkspace

    try:
        calls = patch_to_edit_calls(task.files, task.patch, filepath_prefix="/workspace/")
    except (PatchError, KeyError) as e:
        raise SequenceError("patch_not_convertible", str(e)) from e
    out = []
    with LocalWorkspace(task.files) as ws:
        for c in calls:
            out.append((c.tool, c.args, ws.call(c.tool, c.args)))
    return out


def fixer_view(tl: TaskLocations, z: list[Location], mode: str) -> TaskLocations:
    """The TaskLocations `fixer_script` sees: no teacher steps after the first edit; for `none`,
    N = z (nothing to recover, nothing missing)."""
    if mode == "none":
        return replace(tl, l_edit=[], l_view=list(z), post_events=[])
    if mode == "recover":
        return replace(tl, post_events=[])
    raise ValueError(mode)


def misses(tl: TaskLocations, z: list[Location], files: dict[str, str]) -> bool:
    return any(normalize_path(x.path) in files for x in minus(tl.needed, z))


def build_steps(task: Task, tl: TaskLocations, z: list[Location], mode: str, lc: LocFixConfig,
                edits: list[tuple[str, dict[str, Any], str]], with_edits: bool = True
                ) -> tuple[dict[str, list], dict[str, Any]]:
    """Scripts {fixer, localizer} for one (z, mode). `with_edits=False` stops after the fixer's
    context (the rollout prefix of the signal test)."""
    from ..data.render_obs import ObsRenderer, UnseenCode

    walker = BudgetWalker(max_tool_calls=lc.budget.tool_calls, time_minutes=lc.budget.time_minutes,
                          turns=lc.budget.turns)
    try:
        loc_steps, loc_meta = localizer_session(tl, task.trace, z, walker, lc.em.get("localizer_max_calls", 15),
                                                lc.time_scale)
        with ObsRenderer(task.files, task.known_lines) as r:
            fix_steps, fmeta = fixer_script(task, fixer_view(tl, z, mode), z, mode == "recover", walker, r,
                                            lc.time_scale)
        t_last = max([c.elapsed_s for t in tl.pre_turns for c in t.tool_calls] or [0.0]) * lc.time_scale
        if with_edits:
            for j, (tool, args, out) in enumerate(edits):
                fix_steps.append(ScriptTurn(thinking=EDIT_THINKING[min(j, 1)],
                                            calls=[ScriptCall(tool, args, walker.response(tool, out, t_last))],
                                            meta={"turn": None, "synthetic": "canonical_edit", "weight": 1.0}))
            fix_steps.append(ScriptTurn(text=FINAL_TEXT, meta={"turn": None, "synthetic": "final", "weight": 0.0}))
    except TaskSkip as e:
        raise SequenceError(e.reason, str(e)) from e
    except UnseenCode as e:
        raise SequenceError("unseen_code", str(e)) from e
    except BudgetExceeded as e:
        raise SequenceError("over_budget", str(e)) from e
    req_calls = sum(len(st.calls) for st in fix_steps if isinstance(st, ScriptTurn)
                    and st.meta.get("synthetic") in ("read_missing", "grep_missing"))
    meta = {**loc_meta, "requests": fmeta["requests"], "n_requests": req_calls, "n_request_locations": len(fmeta["requests"]),
            "unrecovered_context": fmeta["unrecovered_context"], "walker": walker}
    return {FIXER: fix_steps, LOCALIZER: loc_steps}, meta


def run_replay(task: Task, scripts: dict[str, list], lc: LocFixConfig, **kw: Any):
    prompt = rebuild_with_budget(task.trace.task_prompt, lc.budget, lc.declared_tools,
                                 problem_statement=task.problem_statement)
    hints = parse_task_prompt(task.trace.task_prompt).hints or None
    return replay(scripts, lc.agent_dir, prompt, task.problem_statement, lc.model_alias, hints=hints,
                  compaction=lc.compaction, **kw)


# --------------------------------------------------------------------------- one sequence
def scoring_record(res, task: Task) -> tuple[dict[str, Any], list[int]]:
    """A record (training format) whose trained turns are the canonical edit turns, and the
    exchange indices of those turns."""
    from ..data.agent_records import REQUEST_META_KEYS, history_echo

    fx = res.of_stream(FIXER)
    ex = fx.agent_exchanges
    edit_k = [k for k, m in enumerate(fx.turn_meta) if m.get("synthetic") == "canonical_edit"]
    if not edit_k:
        raise SequenceError("no_edit_turns")
    last = ex[edit_k[-1]]
    messages = list(last.request["messages"]) + [history_echo(last.response["choices"][0]["message"])]
    for k in edit_k:
        n = len(ex[k].request["messages"])
        if ex[k].request["messages"] != messages[:n]:
            raise SequenceError("not_one_sequence", f"exchange {k} is not a prefix of the last edit turn")
    rec = {"id": f"score__{task.instance_id}", "messages": messages, "tools": last.request.get("tools", []),
           "loss": [{"message_index": len(ex[k].request["messages"]), "train": True, "weight": 1.0} for k in edit_k],
           "weight": 1.0,
           "meta": {"request_params": {k: last.request[k] for k in REQUEST_META_KEYS if k in last.request}}}
    return rec, edit_k


def tokenize_sequence(rec: dict[str, Any], res, edit_k: list[int], tok) -> dict[str, Any]:
    """Token ids, y*₊ positions, and context sizes of one scoring record."""
    from ..render.chat import render_record, render_text

    r = render_record(rec, tok)
    enc = tok(r.text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) != list(r.input_ids):
        raise SequenceError("tokenization_mismatch")
    offsets = enc["offset_mapping"]
    msgs = rec["messages"]
    kw = rec["meta"]["request_params"].get("chat_template_kwargs") or {}
    spans: list[tuple[int, int]] = []
    if len(r.target_spans) != len(edit_k):
        raise SequenceError("target_spans", f"{len(r.target_spans)} spans for {len(edit_k)} edit turns")
    for (s, e, _), loss in zip(r.target_spans, rec["loss"], strict=True):
        tc = msgs[loss["message_index"]]["tool_calls"]
        if len(tc) != 1:
            raise SequenceError("edit_turn_calls", str(len(tc)))
        fn = tc[0]["function"]
        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
        try:
            spans += added_char_spans(r.text[s:e], s, fn["name"], args)
        except SpanError as err:
            raise SequenceError("span", str(err)) from err
    pos = tokens_overlapping(offsets, spans)
    if any(r.labels[t] == -100 for t in pos):
        raise SequenceError("span_outside_target")

    fx = res.of_stream(FIXER)
    ex = fx.agent_exchanges

    def n_tokens_before(char: int) -> int:
        return sum(1 for a, _ in offsets if a < char)

    def prompt_chars(k: int) -> int:
        return len(render_text(tok, ex[k].request["messages"], rec["tools"], kw, add_generation_prompt=True))

    synth = [m.get("synthetic") for m in fx.turn_meta]
    if synth[0] != "call_localizer":
        raise SequenceError("layout", f"first fixer turn is {synth[0]}")
    after_loc = n_tokens_before(prompt_chars(1))          # exchange 1: the first turn after the localizer's answer
    after_z = n_tokens_before(prompt_chars(synth.index("read_z") + 1)) if "read_z" in synth else after_loc
    first_edit = n_tokens_before(r.target_spans[0][0])
    return {"input_ids": list(map(int, r.input_ids)), "positions": pos, "n_plus_tokens": len(pos),
            "n_tokens": len(r.input_ids), "n_z_tokens": after_z - after_loc, "n_ctx_tokens": first_edit - after_loc}


def sequences_for_task(task: Task, tl: TaskLocations, cands, lc: LocFixConfig, tok,
                       modes: tuple[str, ...] = MODES, expect_loc_messages: list[dict] | None = None
                       ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(scoring items, failures) for the candidates of one task. `expect_loc_messages`: the
    localizer prompt of the task's s3_loc_r0 record; the z_T replay must reproduce it exactly."""
    edits = canonical_edits(task)
    items, fails = [], []
    for c in cands:
        for mode in modes:
            if mode == "recover" and not misses(tl, c.z, task.files):
                continue
            base = {"instance_id": task.instance_id, "cand_id": c.cand_id, "source": c.source, "mode": mode}
            try:
                scripts, meta = build_steps(task, tl, c.z, mode, lc, edits)
                res = run_replay(task, scripts, lc)
                if expect_loc_messages is not None and c.cand_id == 0 and mode == "none":
                    got = res.of_stream(LOCALIZER).agent_exchanges[-1].request["messages"]
                    if got != expect_loc_messages:
                        raise RecordsMismatch(task.instance_id)
                rec, edit_k = scoring_record(res, task)
                seq = tokenize_sequence(rec, res, edit_k, tok)
            except SequenceError as e:
                fails.append({**base, "reason": e.reason, "detail": str(e)[:300]})
                continue
            items.append({**base, "n_requests": meta["n_requests"], "n_request_locations": meta["n_request_locations"],
                          "requests": meta["requests"],
                          "n_edits": len(edits), **seq})
    return items, fails


# --------------------------------------------------------------------------- CLI
def add_args_cmd_prepare(p: argparse.ArgumentParser) -> None:
    from ..data.build_common import parse_shard

    p.add_argument("--loc-records", required=True, help="training s3_loc_r0.jsonl")
    p.add_argument("--folds", required=True, help="folds.json (or its folder)")
    p.add_argument("--samples", nargs="+", required=True, help="em.candidates output(s)")
    p.add_argument("--traces", required=True)
    p.add_argument("--solved", required=True)
    p.add_argument("--code", action="append", default=[], help="code folder(s), as for data.locfix-records")
    p.add_argument("--tokenizer", required=True, help="Gemma 4 tokenizer folder (the -ct checkpoint)")
    p.add_argument("--out", required=True, help="output folder: items.jsonl, candidates.jsonl, failures.jsonl")
    p.add_argument("--modes", default="none,recover", help="request modes to build (none, recover)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", type=parse_shard, default=None, help="K/N: tasks with index %% N == K")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))


_W: dict[str, Any] = {}


def _init_worker(traces: str, solved: str, code: list[str], tokenizer: str, cfg: dict) -> None:
    from ..em.tasks import TaskLoader
    from ..render.chat import load_tokenizer

    _W["loader"] = TaskLoader(traces, solved, code, cfg)
    _W["tok"] = load_tokenizer(tokenizer)
    _W["cfg"] = cfg


def _one(job: dict[str, Any]) -> dict[str, Any]:
    from ..em.candidates import assemble
    from ..em.tasks import variants_of, z_from_json

    st = job["task"]
    out: dict[str, Any] = {"instance_id": st["instance_id"], "fold": st["fold"], "trace": st["trace"]}
    try:
        task, tl, code_source = _W["loader"].load(st["trace"])
    except TaskSkip as e:
        return {**out, "skip": e.reason}
    except Exception as e:  # noqa: BLE001
        return {**out, "skip": f"error:{type(e).__name__}", "detail": str(e)[:300]}
    lc = _W["loader"].lc
    v = variants_of(task, tl, lc)
    if "gold" not in v:
        return {**out, "skip": "no_gold_variant"}
    zt = z_from_json(st["z_teacher"])
    if [x.to_json() for x in v["gold"].z] != st["z_teacher"]:
        return {**out, "skip": "z_teacher_differs_from_records"}   # records built from other code/config
    pert = {k: v[k].z for k in ("edit_only", "view_only", "gold_minus_one") if k in v}
    a = assemble(zt, job["texts"], pert, task.files, lc.em)
    try:
        items, fails = sequences_for_task(task, tl, a.candidates, lc, _W["tok"], tuple(job["modes"]),
                                          expect_loc_messages=job["task"].get("loc_messages"))
    except SequenceError as e:
        return {**out, "skip": e.reason, "detail": str(e)[:300]}
    except RecordsMismatch:
        return {**out, "skip": "records_differ_from_agents"}
    for it in items:
        it["fold"] = st["fold"]
    return {**out, "code_source": code_source, "candidates": [c.to_json() for c in a.candidates],
            "assemble": a.stats, "localizer": job.get("localizer"), "items": items, "failures": fails,
            "l_edit": [x.to_json() for x in tl.l_edit], "l_view": [x.to_json() for x in tl.l_view]}


def cmd_prepare(args: argparse.Namespace, cfg: dict) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from ..data.build_common import in_shard
    from ..em.folds import load_folds
    from ..em.tasks import read_jsonl_many, read_stage3_tasks
    from ..runmeta import write_run_meta

    tasks = read_stage3_tasks(args.loc_records, load_folds(args.folds))
    samples = {r["instance_id"]: r for r in read_jsonl_many(args.samples)}
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not set(modes) <= set(MODES):
        raise SystemExit(f"--modes: unknown {set(modes) - set(MODES)}")
    jobs = []
    for i, t in enumerate(tasks):
        if not in_shard(i, args.shard):
            continue
        s = samples.get(t.instance_id, {})
        jobs.append({"task": {"instance_id": t.instance_id, "fold": t.fold, "trace": t.trace,
                              "z_teacher": t.z_teacher, "loc_messages": t.prompt["messages"]},
                     "texts": s.get("texts", []), "localizer": s.get("localizer"), "modes": modes})
    if args.limit:
        jobs = jobs[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"tasks": 0, "tasks_without_samples": sum(1 for j in jobs if not j["texts"])}

    def bump(k: str, n: int = 1) -> None:
        counts[k] = counts.get(k, 0) + n

    with open(out / "items.jsonl", "w") as fi, open(out / "candidates.jsonl", "w") as fc, \
            open(out / "failures.jsonl", "w") as ff:
        def write(r: dict[str, Any]) -> None:
            bump("tasks")
            if r.get("skip"):
                bump(f"skip:{r['skip']}")
                ff.write(json.dumps({k: r[k] for k in r if k != "items"}) + "\n")
                return
            for it in r.pop("items"):
                fi.write(json.dumps(it) + "\n")
                bump(f"items:{it['mode']}")
            for f in r.pop("failures"):
                ff.write(json.dumps(f) + "\n")
                bump(f"failed:{f['mode']}:{f['reason']}")
            bump("candidates", len(r["candidates"]))
            fc.write(json.dumps(r, ensure_ascii=False) + "\n")

        init = (args.traces, args.solved, args.code, args.tokenizer, cfg)
        if args.workers <= 1:
            _init_worker(*init)
            for j in jobs:
                write(_one(j))
        else:
            with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=init) as pool:
                futs = [pool.submit(_one, j) for j in jobs]
                for k, fu in enumerate(as_completed(futs)):
                    write(fu.result())
                    if (k + 1) % 50 == 0:
                        print(f"{k + 1}/{len(jobs)} tasks", flush=True)
    write_run_meta(out, cfg, {"prepare": counts, "shard": args.shard, "modes": modes})
    (out / "prepare_counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True))
    print(json.dumps(counts, indent=2, sort_keys=True))
    bad = counts.get("skip:records_differ_from_agents", 0) + counts.get("skip:z_teacher_differs_from_records", 0)
    if bad:
        print(f"ERROR: {bad} task(s) do not reproduce their records: the records were built with other agent "
              "files or settings (agents/locfix, budget_locfix, em.*) than this code. Use the code version the "
              "records were built with, or rebuild the records. Do not score with this output.")
        return 1
    return 0
