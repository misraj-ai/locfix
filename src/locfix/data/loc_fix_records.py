"""Localizer and fixer records from teacher trajectories (spec 04, Stages 2-3 round 0).

Deployed structure: the fixer is the root agent; the localizer is a sub-agent it calls as a tool
(`agents/locfix`). One scripted replay runs both agents through the real stack:

  localizer stream  the teacher's exploration up to its first edit (run_command / read_file calls
                    with recorded outputs), then a final turn whose text is z (JSON);
  fixer stream      call the localizer -> read the locations of z (rendered offline with the
                    harness's read_file) -> optionally requests for what z misses -> the teacher's
                    trajectory from its first edit to the end (recorded outputs).

Both streams share one tool budget (the localizer's calls count toward the task budget, as in the
harness), so warnings and get_status are rewritten in execution order. The replay's captured
requests are cut into records per stream, exactly as for Stage 1 (`records_from_replay`).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

from ..harness.budget import BudgetExceeded, BudgetWalker
from ..harness.prompts import Budget, parse_task_prompt, rebuild_with_budget
from ..harness.replay import ReplayResult, ScriptCall, ScriptNudge, ScriptTurn, Step, replay
from .agent_records import DEFAULT_TOOLS, _turn_weight, records_from_replay
from .edits import make_unified_diff
from .locations import Location, edit_locations, normalize_path, view_locations
from .render_obs import ObsRenderer, read_calls_for
from .traces import Nudge, ToolCall, Trace, Turn, is_workspace_write_command, load_trace
from .zloc import key, minus, prune, union, z_to_json

LOCALIZER = "localizer"
FIXER = "fixer"
LOCALIZER_TOOLS = ("run_command", "read_file")

# Synthetic text. Templated on purpose: no LLM writing here (it would reintroduce recall).
TEMPLATES = {
    "first_request": "Find the code locations relevant to this issue.",
    "fixer_call_localizer": "I'll start by asking the localizer for the code relevant to this issue.",
    "fixer_read_z": "The localizer returned {n} location(s). I'll read them.",
    "fixer_read_z_empty": "The localizer returned no locations.",
    "fixer_read_missing": "I also need {what}, which the code I have read so far refers to. Let me read it.",
    "fixer_grep_missing": "I need {what}; let me search for it.",
    "localizer_answer": "I have found the relevant code. Returning the locations.",
}


class TaskSkip(RuntimeError):
    """A task or variant is dropped; `reason` is counted in the manifest."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


# --------------------------------------------------------------------------- inputs
@dataclass
class Task:
    trace: Trace
    files: dict[str, str]                       # base-commit code (code/<id>.json)
    patch: str                                  # teacher's verified patch (solved) / gold (unsolved)
    problem_statement: str
    hints: str | None = None
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def instance_id(self) -> str:
        return self.trace.instance_id


def patch_from_edits(trace: Trace, files: dict[str, str], strict: bool = True) -> str:
    """The teacher's patch recomputed from its edit_file/write_file calls (fixtures, cross-checks).
    With `strict`, raises if the trajectory may write files any other way (e.g. `git stash`)."""
    from ..harness.local_sandbox import LocalWorkspace

    with LocalWorkspace(files) as ws:
        touched: list[str] = []
        for c in trace.tool_calls():
            if c.tool in ("edit_file", "write_file"):
                ws.call(c.tool, c.args)
                p = normalize_path(c.args["filepath"])
                if p not in touched:
                    touched.append(p)
            elif strict and c.tool == "run_command" and is_workspace_write_command(c.args.get("command", "")):
                raise TaskSkip("shell_write", c.args.get("command", "")[:80])
        return "".join(make_unified_diff(p, files.get(p, ""), ws.read_text(p)) for p in touched)


@dataclass
class TaskLocations:
    l_edit: list[Location]
    l_view: list[Location]
    pre_turns: list[Turn]                       # teacher turns before the first edit
    post_events: list[Turn | Nudge]             # from the first-edit turn to the end
    searches: list[ToolCall]                    # grep/find/rg calls before the first edit

    @property
    def needed(self) -> list[Location]:
        """N = L_edit ∪ L_view, edit first."""
        return union(self.l_edit, self.l_view)


_SEARCH = re.compile(r"(^|[\s;&|(])(grep|rg|find|ag|git grep)\s")


def split_trace(trace: Trace) -> tuple[list[Turn], list[Turn | Nudge]]:
    first = trace.first_edit_call()
    if first is None:
        raise TaskSkip("no_edit")
    pre: list[Turn] = []
    post: list[Turn | Nudge] = []
    for ev in trace.events:
        if isinstance(ev, Turn) and not post:
            ids = [c.entry_index for c in ev.tool_calls]
            if first.entry_index in ids:
                if any(i < first.entry_index for i in ids):
                    raise TaskSkip("mixed_first_edit_turn")
                post.append(ev)
            else:
                pre.append(ev)
        elif isinstance(ev, Nudge) and not post:
            continue                             # sub-agents get no nudges; a pre-edit nudge is dropped
        elif isinstance(ev, (Turn, Nudge)):
            post.append(ev)
        else:
            raise TaskSkip("compaction_in_trace")
    return pre, post


def task_locations(task: Task) -> TaskLocations:
    pre, post = split_trace(task.trace)
    try:
        l_edit = edit_locations(task.files, task.patch)
    except KeyError as e:
        raise TaskSkip("missing_code", str(e)) from e
    if not l_edit:
        raise TaskSkip("empty_l_edit")
    l_view = view_locations(task.trace, task.files)
    searches = [c for c in task.trace.calls_before_first_edit()
                if c.tool == "run_command" and _SEARCH.search(c.args.get("command", ""))]
    return TaskLocations(l_edit, l_view, pre, post, searches)


# --------------------------------------------------------------------------- variants of z
@dataclass
class Variant:
    name: str
    z: list[Location]
    requests: bool                              # may the fixer recover what z misses?
    fixer_stage: str | None                     # stage of the fixer records, None = not kept
    localizer_stage: str | None                 # stage of the localizer records, None = not kept


def solved_variants(tl: TaskLocations, files: dict[str, str], em_cfg: dict[str, Any],
                    rng: random.Random) -> list[Variant]:
    """Stage 2 / round-0 variants for one solved task (spec 04 §4-§6).

    - gold: N = L_edit ∪ L_view pruned to the budget; fixer without requests (2a); its localizer
      stream is the round-0 localizer target;
    - edit_only (z = L_edit) and view_only (z = L_view, items the teacher edited marked `edit`):
      the Stage 2 localizer targets, and noisy fixer inputs with requests (2b);
    - gold_minus_one: one context item dropped (2b).
    A fixer record whose z equals an earlier variant's is not emitted twice. Requests are disabled
    for `no_request_fraction` of the variants whose z covers L_edit. Callers/callees and BM25
    distractors are built by `locfix.em.candidates` (spec 06)."""
    P = dict(max_locations=em_cfg.get("max_locations", 8), budget_tokens=em_cfg.get("ctx_budget_tokens", 6000))
    edit_keys = {key(x) for x in tl.l_edit}
    view = [Location(x.path, x.symbol, x.start, x.end, "edit" if key(x) in edit_keys else x.role)
            for x in tl.l_view]
    gold = prune(tl.needed, files, **P)
    out = [Variant("gold", gold, False, "s2_fix_gold", "s3_loc_r0"),
           Variant("edit_only", list(tl.l_edit), True, "s2_fix_noisy", "s2_loc_edit")]
    view_only = prune(view, files, **P)
    if view_only:
        out.append(Variant("view_only", view_only, True, "s2_fix_noisy", "s2_loc_view"))
    ctx = [x for x in gold if x.role != "edit"]
    if ctx:
        drop = rng.choice(ctx)
        out.append(Variant("gold_minus_one", [x for x in gold if x is not drop], True, "s2_fix_noisy", None))
    no_req = em_cfg.get("no_request_fraction", 0.4)
    seen_z: list[list[tuple]] = []
    for v in out:
        zk = [key(x) for x in v.z]
        if zk in seen_z:
            v.fixer_stage = None
        seen_z.append(zk)
        if v.requests and _covers_edit(v.z, tl.l_edit) and rng.random() < no_req:
            v.requests = False
    return [v for v in out if v.fixer_stage or v.localizer_stage]


def _covers_edit(z: list[Location], l_edit: list[Location]) -> bool:
    keys = {(normalize_path(x.path), x.symbol) for x in z}
    return all((normalize_path(x.path), x.symbol) in keys for x in l_edit)


# --------------------------------------------------------------------------- scripts
@dataclass
class Scripts:
    fixer: list[Step]
    localizer: list[Step]
    meta: dict[str, Any]


def _script_turn(turn: Turn, calls: list[ScriptCall], seen: dict, entries: list[dict],
                 thinking_prefix: str | None = None, fold_text: bool = True) -> ScriptTurn:
    weight, reason = _turn_weight(turn, seen, entries)
    thinking, text = turn.thinking_text or None, turn.visible_text or None
    if thinking_prefix:
        thinking = f"{thinking_prefix}\n\n{thinking}" if thinking else thinking_prefix
    merged = False
    if calls and text and fold_text:
        thinking = f"{thinking}\n\n{text}" if thinking else text
        text, merged = None, True
    return ScriptTurn(thinking=thinking, text=text, calls=calls,
                      meta={"turn": turn.index, "entries": turn.entry_indices, "weight": weight,
                            "reason": reason, "text_merged": merged})


def _remember(seen: dict, c: ToolCall) -> None:
    seen[c.tool + json.dumps(c.args, sort_keys=True)] = c.result or ""
    seen[c.tool + json.dumps({k: v for k, v in c.args.items() if k != "end_line"}, sort_keys=True)] = c.result or ""


def localizer_session(tl: TaskLocations, trace: Trace, z: list[Location], walker: BudgetWalker,
                      max_calls: int, time_scale: float = 1.0) -> tuple[list[Step], dict[str, Any]]:
    """The teacher's exploration (run_command/read_file only) followed by z."""
    steps: list[Step] = []
    seen: dict[str, str] = {}
    carry: str | None = None
    n_calls = 0
    dropped = 0
    for turn in tl.pre_turns:
        calls = []
        for c in turn.tool_calls:
            if c.tool not in LOCALIZER_TOOLS:
                dropped += 1
                continue
            out = walker.response(c.tool, c.result or "", elapsed_seconds=c.elapsed_s * time_scale)
            calls.append(ScriptCall(c.tool, c.args, out))
            n_calls += 1
        if not calls:
            # no call the localizer can make (e.g. get_status, or a text-only turn): keep the thought
            parts = [x for x in (carry, turn.thinking_text, turn.visible_text) if x]
            carry = "\n\n".join(parts) or None
            continue
        st = _script_turn(turn, calls, seen, trace.entries, thinking_prefix=carry)
        carry = None
        for c in turn.tool_calls:
            _remember(seen, c)
        steps.append(st)
    if n_calls > max_calls:
        raise TaskSkip("localizer_over_calls", f"{n_calls} > {max_calls}")
    final_thinking = TEMPLATES["localizer_answer"]
    if carry:
        final_thinking = f"{carry}\n\n{final_thinking}"
    steps.append(ScriptTurn(thinking=final_thinking, text=z_to_json(z),
                            meta={"turn": None, "synthetic": "localizer_answer", "weight": 1.0}))
    return steps, {"localizer_calls": n_calls, "localizer_dropped_calls": dropped}


_BACKTICK = re.compile(r"`([^`\n]{2,80})`")
_PATHLIKE = re.compile(r"(?<![\w/.])((?:[\w.-]+/)+[\w.-]+\.\w+|[\w-]+\.py)\b")
_IDENT = re.compile(r"^[A-Za-z_][\w.]*(?:\(\))?$|^[\w./-]+\.\w+$")
_IGNORE = {"self", "cls", "None", "True", "False", "return", "raise", "import", "def", "class",
           "str", "int", "dict", "list", "tuple", "set", "bool", "float", "object", "type", "kwargs",
           "args", "len", "isinstance", "print"}


def references(text: str) -> set[str]:
    """Code names and paths a piece of reasoning refers to: backticked identifiers/paths and bare
    file paths."""
    out: set[str] = set()
    for m in _BACKTICK.finditer(text or ""):
        t = m.group(1).strip()
        if _IDENT.match(t):
            out.add(t.removesuffix("()"))
    for m in _PATHLIKE.finditer(text or ""):
        out.add(m.group(1))
    return {t for t in out if t not in _IGNORE and len(t) >= 3}


def dangling(refs: set[str], seen_text: str) -> list[str]:
    missing = []
    for r in sorted(refs):
        last = r.rsplit("/", 1)[-1] if "/" in r else r.rsplit(".", 1)[-1] if not r.endswith(".py") else r
        if r in seen_text or (last and last in seen_text):
            continue
        missing.append(r)
    return missing


def _nameable(loc: Location, seen_text: str) -> bool:
    if loc.path in seen_text:
        return True
    return not loc.symbol.startswith("<") and loc.symbol.rsplit(".", 1)[-1] in seen_text


def _what(locs: list[Location], limit: int = 4) -> str:
    names = [f"`{x.symbol}` in `{x.path}`" if not x.symbol.startswith("<") else f"`{x.path}`"
             for x in locs]
    names = list(dict.fromkeys(names))
    if len(names) > limit:
        return ", ".join(names[:limit]) + f" and {len(names) - limit} more"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def fixer_script(task: Task, tl: TaskLocations, z: list[Location], requests: bool,
                 walker: BudgetWalker, renderer: ObsRenderer, time_scale: float = 1.0
                 ) -> tuple[list[Step], dict[str, Any]]:
    steps: list[Step] = []
    seen_parts = [task.trace.task_prompt, task.problem_statement, task.hints or "", z_to_json(z)]
    meta: dict[str, Any] = {"requests": [], "unrecovered_context": []}
    t_last_pre = max([c.elapsed_s for t in tl.pre_turns for c in t.tool_calls] or [0.0]) * time_scale

    steps.append(ScriptTurn(thinking=TEMPLATES["fixer_call_localizer"],
                            calls=[ScriptCall(LOCALIZER, {"request": TEMPLATES["first_request"]}, "")],
                            meta={"turn": None, "synthetic": "call_localizer", "weight": 1.0}))

    def reads_turn(locs: list[Location], thinking: str, synthetic: str) -> ScriptTurn | None:
        rc = read_calls_for(locs, task.files, renderer)
        if not rc:
            return None
        calls = []
        for call, out in rc:
            calls.append(ScriptCall("read_file", call.args,
                                    walker.response("read_file", out, elapsed_seconds=t_last_pre)))
            seen_parts.append(call.filepath)
            seen_parts.append(out)
        return ScriptTurn(thinking=thinking, calls=calls,
                          meta={"turn": None, "synthetic": synthetic, "weight": 1.0})

    st = reads_turn(z, TEMPLATES["fixer_read_z"].format(n=len(z)), "read_z")
    if st is not None:
        steps.append(st)

    missing = [x for x in minus(tl.needed, z) if normalize_path(x.path) in task.files]
    if missing and requests:
        # Repeat: read every missing location the fixer can already name (its path or symbol name
        # appears in what it has seen) in one turn; otherwise run a recorded teacher search whose
        # output names the next missing file, then read. What stays unreachable is dropped
        # (context) or makes the record impossible (edit).
        while missing:
            seen_text = "\n".join(seen_parts)
            nameable = [x for x in missing if _nameable(x, seen_text)]
            if nameable:
                st = reads_turn(nameable, TEMPLATES["fixer_read_missing"].format(what=_what(nameable)),
                                "read_missing")
                if st is not None:
                    steps.append(st)
                meta["requests"].extend({"loc": x.to_json(), "via": "read"} for x in nameable)
                missing = minus(missing, nameable)
                continue
            loc = missing[0]
            grep = next((c for c in tl.searches if loc.path in (c.result or "")), None)
            if grep is not None:
                out = walker.response("run_command", grep.result or "", elapsed_seconds=t_last_pre)
                steps.append(ScriptTurn(thinking=TEMPLATES["fixer_grep_missing"].format(what=_what([loc])),
                                        calls=[ScriptCall("run_command", grep.args, out)],
                                        meta={"turn": None, "synthetic": "grep_missing", "weight": 1.0}))
                seen_parts.append(grep.result or "")
                meta["requests"].append({"loc": loc.to_json(), "via": "grep", "entry": grep.entry_index})
                continue
            if loc.role == "edit":
                raise TaskSkip("impossible_input", f"{loc.path}:{loc.symbol} not in z and not recoverable")
            meta["unrecovered_context"].append(loc.to_json())
            missing = missing[1:]
    elif missing:
        lost_edit = [x for x in missing if x.role == "edit"]
        if lost_edit:
            raise TaskSkip("impossible_input", f"{len(lost_edit)} edit location(s) missing, no requests")
        meta["unrecovered_context"] = [x.to_json() for x in missing]

    # the teacher's trajectory from its first edit on
    seen: dict[str, str] = {}
    for t in tl.pre_turns:
        for c in t.tool_calls:
            _remember(seen, c)
    dangling_refs: list[dict[str, Any]] = []
    for ev in tl.post_events:
        if isinstance(ev, Nudge):
            steps.append(ScriptNudge(ev.content))
            continue
        refs = references(ev.thinking_text + "\n" + ev.visible_text)
        miss = dangling(refs, "\n".join(seen_parts))
        if miss:
            dangling_refs.append({"turn": ev.index, "refs": miss})
        calls = []
        for c in ev.tool_calls:
            out = walker.response(c.tool, c.result or "", elapsed_seconds=c.elapsed_s * time_scale)
            calls.append(ScriptCall(c.tool, c.args, out))
        st = _script_turn(ev, calls, seen, task.trace.entries)
        for c in ev.tool_calls:
            _remember(seen, c)
            seen_parts.append(json.dumps(c.args, ensure_ascii=False))
            seen_parts.append(c.result or "")
        steps.append(st)
    meta["dangling_refs"] = dangling_refs
    return steps, meta


def build_scripts(task: Task, tl: TaskLocations, v: Variant, budget: Budget, em_cfg: dict[str, Any],
                  time_scale: float = 1.0) -> Scripts:
    walker = BudgetWalker(max_tool_calls=budget.tool_calls, time_minutes=budget.time_minutes,
                          turns=budget.turns)
    loc_steps, loc_meta = localizer_session(tl, task.trace, v.z, walker,
                                            em_cfg.get("localizer_max_calls", 12), time_scale)
    with ObsRenderer(task.files) as r:
        fix_steps, fix_meta = fixer_script(task, tl, v.z, v.requests, walker, r, time_scale)
    meta = {**loc_meta, **fix_meta, "counted_calls": walker.used}
    return Scripts(fix_steps, loc_steps, meta)


# --------------------------------------------------------------------------- records
@dataclass
class LocFixConfig:
    agent_dir: str = "agents/locfix"
    model_alias: str = "gemma-4-31b-it-qat-w4a16-ct"
    budget: Budget = field(default_factory=lambda: Budget(5, 40, 300))
    declared_tools: list[str] = field(default_factory=lambda: [*DEFAULT_TOOLS, LOCALIZER])
    compaction: dict[str, Any] | None = None
    layout: str = "segments"
    time_scale: float = 1.0
    em: dict[str, Any] = field(default_factory=dict)
    seed: int = 1234


def replay_variant(task: Task, tl: TaskLocations, v: Variant, cfg: LocFixConfig, summarizer=None,
                   prompt_token_counter: Callable[[dict], int] | None = None
                   ) -> tuple[ReplayResult, Scripts]:
    sc = build_scripts(task, tl, v, cfg.budget, cfg.em, cfg.time_scale)
    prompt = rebuild_with_budget(task.trace.task_prompt, cfg.budget, cfg.declared_tools,
                                 problem_statement=task.problem_statement)
    hints = parse_task_prompt(task.trace.task_prompt).hints or None
    res = replay({FIXER: sc.fixer, LOCALIZER: sc.localizer}, cfg.agent_dir, prompt,
                 task.problem_statement, cfg.model_alias, hints=hints, compaction=cfg.compaction,
                 summarizer=summarizer, prompt_token_counter=prompt_token_counter)
    return res, sc


def records_for_variant(task: Task, tl: TaskLocations, v: Variant, cfg: LocFixConfig, **kw: Any
                        ) -> list[dict[str, Any]]:
    res, sc = replay_variant(task, tl, v, cfg, **kw)
    common = {"variant": v.name, "z": [x.to_json() for x in v.z], "requests_enabled": v.requests,
              "l_edit": [x.to_json() for x in tl.l_edit], "l_view": [x.to_json() for x in tl.l_view],
              "budget": cfg.budget.__dict__, "trace": Path(task.trace.path).name, **sc.meta}
    fixer_only = ("requests", "unrecovered_context", "dangling_refs")
    out: list[dict[str, Any]] = []
    for stream, stage, role in ((FIXER, v.fixer_stage, "fixer"), (LOCALIZER, v.localizer_stage, "localizer")):
        if stage is None:
            continue
        meta = common if role == "fixer" else {k: x for k, x in common.items() if k not in fixer_only}
        recs = records_from_replay(res.of_stream(stream), stage=stage, role=role,
                                   instance_id=task.instance_id, source=task.source,
                                   agent_config=cfg.agent_dir, layout=cfg.layout, meta=meta)
        for r in recs:
            r["id"] = f"{r['id']}__{v.name}"
            r["z"] = common["z"]
        out.extend(recs)
    return out


def build_task_records(task: Task, cfg: LocFixConfig, counts: dict[str, int] | None = None,
                       **kw: Any) -> list[dict[str, Any]]:
    counts = counts if counts is not None else {}
    rng = random.Random(f"{cfg.seed}:{task.instance_id}")
    tl = task_locations(task)
    out: list[dict[str, Any]] = []
    for v in solved_variants(tl, task.files, cfg.em, rng):
        if v.fixer_stage is None and v.localizer_stage is None:
            continue
        try:
            recs = records_for_variant(task, tl, v, cfg, **kw)
        except TaskSkip as e:
            counts[f"drop:{v.name}:{e.reason}"] = counts.get(f"drop:{v.name}:{e.reason}", 0) + 1
            continue
        except BudgetExceeded:
            counts[f"drop:{v.name}:over_budget"] = counts.get(f"drop:{v.name}:over_budget", 0) + 1
            continue
        for r in recs:
            k = f"records:{r['stage']}"
            counts[k] = counts.get(k, 0) + 1
            if r["meta"].get("requests"):
                counts[f"with_requests:{r['stage']}"] = counts.get(f"with_requests:{r['stage']}", 0) + 1
        out.extend(recs)
    return out


# --------------------------------------------------------------------------- CLI
def add_args_cmd_locfix_records(p: argparse.ArgumentParser) -> None:
    p.add_argument("traces", help="folder of cleaned traces")
    p.add_argument("--solved", required=True, help="solved.jsonl (teacher.patch, problem_statement, fold)")
    p.add_argument("--code", required=True, help="folder of code/<instance_id>.json")
    p.add_argument("--out", required=True, help="output folder (one JSONL per stage)")
    p.add_argument("--limit", type=int, default=0)


def cmd_locfix_records(args: argparse.Namespace, cfg: dict) -> int:
    from ..runmeta import write_run_meta

    b = cfg["budget_locfix"]
    lc = LocFixConfig(
        model_alias=cfg["models"]["31b"]["serve_id"],
        budget=Budget(b["max_time_minutes"], b["max_tool_calls"], b["max_turns"]),
        compaction=cfg["harness"]["compaction"] if cfg["data"].get("replay_compaction") else None,
        layout="segments" if cfg["data"].get("training_layout", "auto") in ("auto", "one_sequence") else "per_turn",
        em=cfg["em"], seed=cfg["seed"],
    )
    rows: dict[str, dict] = {}
    with open(args.solved) as f:
        for line in f:
            r = json.loads(line)
            rows[r["instance_id"]] = r
    exclude = [x.lower() for x in cfg["data"].get("exclude_repos", [])]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"traces": 0}
    handles: dict[str, Any] = {}
    try:
        for i, p in enumerate(sorted(Path(args.traces).glob("*.json"))):
            if args.limit and i >= args.limit:
                break
            counts["traces"] += 1
            try:
                tr = load_trace(p)
                repo = parse_task_prompt(tr.task_prompt).repo.lower()
                if any(repo.split("/")[-1] == x for x in exclude):
                    raise TaskSkip("dev_repo")
                row = rows.get(tr.instance_id)
                if row is None:
                    raise TaskSkip("not_in_solved")
                code_path = Path(args.code) / f"{tr.instance_id}.json"
                if not code_path.exists():
                    raise TaskSkip("no_code_file")
                task = Task(tr, json.loads(code_path.read_text()), row["teacher"]["patch"],
                            row["problem_statement"], source={"set": "solved", "dataset": row.get("verifier", "other"),
                                                              "fold": row.get("fold", "A"), "attempt": tr.attempt or 1})
                recs = build_task_records(task, lc, counts)
            except TaskSkip as e:
                counts[f"skip:{e.reason}"] = counts.get(f"skip:{e.reason}", 0) + 1
                continue
            except Exception as e:  # noqa: BLE001 - count and continue, reported below
                key = f"error:{type(e).__name__}"
                counts[key] = counts.get(key, 0) + 1
                print(f"[{p.name}] {type(e).__name__}: {e}")
                continue
            for r in recs:
                if r["stage"] not in handles:
                    handles[r["stage"]] = open(out / f"{r['stage']}.jsonl", "w")
                handles[r["stage"]].write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        for h in handles.values():
            h.close()
    write_run_meta(out, cfg, {"counts": counts, "templates": TEMPLATES})
    (out / "manifest.json").write_text(json.dumps(counts, indent=2, sort_keys=True))
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0
