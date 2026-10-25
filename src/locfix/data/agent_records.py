"""Stage 1 agent records: teacher trajectory -> scripted replay -> training records (spec 03).

Pipeline per trajectory:
  1. `script_from_trace`: turns + recorded tool outputs, rewritten for our budget with the
     harness's own warning/status functions (`BudgetWalker`); per-turn loss weights.
  2. `replay`: run the script through the real agent stack and capture every model request.
  3. `records_from_replay`: cut the captured requests into training records. Consecutive requests
     that extend each other form one segment (one record, loss on every response in the segment);
     compaction or any other break starts a new segment. `layout="per_turn"` gives one record per
     model call instead.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..harness.budget import BudgetExceeded, BudgetWalker
from ..harness.prompts import Budget, parse_task_prompt, rebuild_with_budget
from ..harness.replay import ReplayResult, ScriptCall, ScriptNudge, ScriptTurn, Step, replay
from .traces import Nudge, Trace, Turn, load_trace

DEFAULT_TOOLS = ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"]


class RecordError(RuntimeError):
    pass


# --------------------------------------------------------------------------- script
@dataclass
class ScriptInfo:
    n_counted_calls: int
    teacher_seconds: float
    zero_weight_turns: dict[int, str] = field(default_factory=dict)


def _turn_weight(turn: Turn, seen: dict[str, str], entries: list[dict]) -> tuple[float, str | None]:
    """Loss weight of a teacher turn.

    - cleaning v2 may mark entries with `train_weight` (e.g. 0 for reference-seeking calls);
    - a turn without thinking whose calls all repeat an earlier call with the same arguments and
      got the same output (e.g. `read_file` with an `end_line` past the 150-line cap) gets 0.
    """
    marked = [entries[i].get("train_weight") for i in turn.entry_indices if "train_weight" in entries[i]]
    if marked:
        return float(min(marked)), "cleaning_mark"
    if turn.tool_calls and not turn.thinking and not turn.text:
        repeats = []
        for c in turn.tool_calls:
            key = c.tool + json.dumps(c.args, sort_keys=True)
            norm_key = c.tool + json.dumps({k: v for k, v in c.args.items() if k != "end_line"}, sort_keys=True)
            repeats.append(seen.get(key) == c.result or seen.get(norm_key) == c.result)
        if all(repeats):
            return 0.0, "repeated_call"
    return 1.0, None


def script_from_trace(trace: Trace, budget: Budget, time_scale: float = 1.0) -> tuple[list[Step], ScriptInfo]:
    """Teacher trace -> replay script under `budget`. Raises BudgetExceeded if the trajectory
    needs more counted tool calls than the budget allows.

    `time_scale` multiplies the teacher's clock for `get_status` time fields (projected Gemma time
    / teacher time; 1.0 until timing runs exist)."""
    walker = BudgetWalker(max_tool_calls=budget.tool_calls, time_minutes=budget.time_minutes,
                          turns=budget.turns)
    steps: list[Step] = []
    seen: dict[str, str] = {}
    zero: dict[int, str] = {}
    for ev in trace.events:
        if isinstance(ev, Nudge):
            steps.append(ScriptNudge(ev.content))
            continue
        if not isinstance(ev, Turn):
            raise RecordError(f"{trace.path}: compaction entries in teacher traces are not supported")
        weight, reason = _turn_weight(ev, seen, trace.entries)
        if weight != 1.0 and reason:
            zero[ev.index] = reason
        calls = []
        for c in ev.tool_calls:
            out = walker.response(c.tool, c.result or "", elapsed_seconds=c.elapsed_s * time_scale)
            calls.append(ScriptCall(c.tool, c.args, out))
            seen[c.tool + json.dumps(c.args, sort_keys=True)] = c.result or ""
            seen[c.tool + json.dumps({k: v for k, v in c.args.items() if k != "end_line"}, sort_keys=True)] = c.result or ""
        thinking, text = ev.thinking_text or None, ev.visible_text or None
        merged = False
        if calls and text:
            # Gemma 4's template renders a tool-call turn's content after the tool responses, so a
            # turn cannot carry visible text at the position the model wrote it. Fold the teacher's
            # narration into the thinking channel (the model's native order: thought -> calls).
            thinking = f"{thinking}\n\n{text}" if thinking else text
            text, merged = None, True
        steps.append(ScriptTurn(thinking=thinking, text=text, calls=calls,
                                meta={"turn": ev.index, "entries": ev.entry_indices,
                                      "weight": weight, "reason": reason, "text_merged": merged}))
    return steps, ScriptInfo(walker.used, trace.duration_s or 0.0, zero)


# --------------------------------------------------------------------------- records
def history_echo(resp_message: dict[str, Any]) -> dict[str, Any]:
    """The assistant message as ADK/LiteLLM put it into the next request (verified against real
    echoes in tests): role, content if any, tool_calls (type, id, function), reasoning fields."""
    out: dict[str, Any] = {"role": "assistant"}
    if resp_message.get("content"):
        out["content"] = resp_message["content"]
    if resp_message.get("tool_calls"):
        out["tool_calls"] = [{"type": "function", "id": tc["id"],
                              "function": {"name": tc["function"]["name"],
                                           "arguments": tc["function"]["arguments"]}}
                             for tc in resp_message["tool_calls"]]
    reasoning = resp_message.get("reasoning_content") or resp_message.get("reasoning")
    if reasoning:
        out["reasoning_content"] = reasoning
        out["reasoning"] = reasoning
    return out


def _extends(prev_req: dict, prev_resp: dict, cur_req: dict) -> bool:
    """Does `cur_req` continue `prev_req` in a way that keeps the token rendering a prefix?

    Besides message-level extension, no new user message may appear: Gemma 4's template renders
    an assistant turn's thinking only after the last user message, so a nudge (or a compaction
    summary) removes earlier thoughts from later prompts and must start a new record."""
    pm, cm = prev_req["messages"], cur_req["messages"]
    if len(cm) <= len(pm) or cm[: len(pm)] != pm:
        return False
    if cm[len(pm)] != history_echo(prev_resp["choices"][0]["message"]):
        return False
    return all(m.get("role") != "user" for m in cm[len(pm) + 1:])


REQUEST_META_KEYS = ("chat_template_kwargs", "thinking_token_budget", "max_completion_tokens",
                     "max_tokens", "temperature", "top_p", "top_k", "seed")


def records_from_replay(res: ReplayResult, *, stage: str, role: str, instance_id: str,
                        source: dict[str, Any], agent_config: str, layout: str = "segments",
                        meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    ex = res.agent_exchanges
    if not ex:
        raise RecordError("replay produced no model requests")
    if len(res.turn_meta) != len(ex):
        raise RecordError("turn metadata does not match the captured requests")

    if layout == "per_turn":
        groups = [[k] for k in range(len(ex))]
    elif layout == "segments":
        groups, cur = [], [0]
        for k in range(1, len(ex)):
            if _extends(ex[k - 1].request, ex[k - 1].response, ex[k].request):
                cur.append(k)
            else:
                groups.append(cur)
                cur = [k]
        groups.append(cur)
    else:
        raise ValueError(f"unknown layout {layout!r}")

    records = []
    for gi, g in enumerate(groups):
        last = ex[g[-1]]
        messages = list(last.request["messages"]) + [history_echo(last.response["choices"][0]["message"])]
        loss = []
        for k in g:
            idx = len(ex[k].request["messages"])
            tm = res.turn_meta[k]
            entry = {"message_index": idx, "train": tm.get("weight", 1.0) > 0, "weight": tm.get("weight", 1.0)}
            if tm.get("reason"):
                entry["reason"] = tm["reason"]
            loss.append(entry)
        rec = {
            "id": f"{stage}__{instance_id}__a{source.get('attempt', 0)}__w{gi}",
            "stage": stage,
            "role": role,
            "instance_id": instance_id,
            "source": source,
            "agent_config": agent_config,
            "tools": last.request.get("tools", []),
            "messages": messages,
            "loss": loss,
            "weight": 1.0,
            "spans": None,
            "meta": {
                "request_params": {k: last.request[k] for k in REQUEST_META_KEYS if k in last.request},
                "layout": layout,
                "window": gi,
                "n_windows": len(groups),
                "turns": [res.turn_meta[k].get("turn") for k in g],
                **(meta or {}),
            },
        }
        records.append(rec)
    return records


# --------------------------------------------------------------------------- per trace
@dataclass
class BuildConfig:
    agent_dir: str = "agents/single"
    model_alias: str = "gemma-4-31b-it-qat-w4a16-ct"
    budget: Budget = field(default_factory=lambda: Budget(5, 30, 300))
    declared_tools: list[str] = field(default_factory=lambda: list(DEFAULT_TOOLS))
    compaction: dict[str, Any] | None = None
    layout: str = "segments"
    time_scale: float = 1.0
    stage: str = "s1_agent"


def build_agent_records(trace: Trace, bc: BuildConfig, source: dict[str, Any],
                        problem_statement: str | None = None, summarizer=None,
                        prompt_token_counter=None) -> list[dict[str, Any]]:
    steps, info = script_from_trace(trace, bc.budget, bc.time_scale)
    parts = parse_task_prompt(trace.task_prompt)
    prompt = rebuild_with_budget(trace.task_prompt, bc.budget, bc.declared_tools,
                                 problem_statement=problem_statement)
    res = replay(steps, bc.agent_dir, prompt, problem_statement or parts.problem_statement,
                 bc.model_alias, hints=parts.hints or None, compaction=bc.compaction,
                 summarizer=summarizer, prompt_token_counter=prompt_token_counter)
    return records_from_replay(
        res, stage=bc.stage, role="agent", instance_id=trace.instance_id, source=source,
        agent_config=bc.agent_dir, layout=bc.layout,
        meta={"n_counted_calls": info.n_counted_calls, "teacher_seconds": info.teacher_seconds,
              "budget": bc.budget.__dict__, "zero_weight_turns": info.zero_weight_turns,
              "trace": Path(trace.path).name},
    )


# --------------------------------------------------------------------------- CLI
def add_args_cmd_agent_records(p: argparse.ArgumentParser) -> None:
    p.add_argument("traces", help="folder of cleaned traces")
    p.add_argument("--out", required=True, help="output JSONL")
    p.add_argument("--solved", help="solved.jsonl (instance_id, fold, verifier, problem_statement)")
    p.add_argument("--limit", type=int, default=0)


def cmd_agent_records(args: argparse.Namespace, cfg: dict) -> int:
    from ..runmeta import write_run_meta

    b = cfg["budget"]
    bc = BuildConfig(
        model_alias=cfg["models"]["31b"]["serve_id"],
        budget=Budget(b["max_time_minutes"], b["max_tool_calls"], b["max_turns"]),
        compaction=cfg["harness"]["compaction"] if cfg["data"].get("replay_compaction") else None,
        layout="segments" if cfg["data"].get("training_layout", "auto") in ("auto", "one_sequence") else "per_turn",
    )
    rows: dict[str, dict] = {}
    if args.solved:
        with open(args.solved) as f:
            for line in f:
                r = json.loads(line)
                rows[r["instance_id"]] = r
    exclude = [x.lower() for x in cfg["data"].get("exclude_repos", [])]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"traces": 0, "records": 0}
    with open(out, "w") as f:
        for i, p in enumerate(sorted(Path(args.traces).glob("*.json"))):
            if args.limit and i >= args.limit:
                break
            counts["traces"] += 1
            try:
                tr = load_trace(p)
                repo = parse_task_prompt(tr.task_prompt).repo.lower()
                if any(repo.split("/")[-1] == x for x in exclude):
                    counts["skip_dev_repo"] = counts.get("skip_dev_repo", 0) + 1
                    continue
                row = rows.get(tr.instance_id, {})
                source = {"set": "solved", "dataset": row.get("verifier", "other"),
                          "fold": row.get("fold", "A"), "attempt": tr.attempt or 1}
                recs = build_agent_records(tr, bc, source, row.get("problem_statement"))
            except BudgetExceeded:
                counts["skip_over_budget"] = counts.get("skip_over_budget", 0) + 1
                continue
            except Exception as e:  # noqa: BLE001 - count and continue, reported below
                key = f"error:{type(e).__name__}"
                counts[key] = counts.get(key, 0) + 1
                print(f"[{p.name}] {type(e).__name__}: {e}")
                continue
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            counts["records"] += len(recs)
    write_run_meta(out.parent, cfg, {"counts": counts, "output": str(out)})
    print(json.dumps(counts, indent=2))
    return 0
