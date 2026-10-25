"""Loading teacher traces and grouping their entries into model turns.

A trace (`traces/<id>__a<N>.json`) is a list of entries; see context/DATA_PIPELINE.md §4. This
module turns it into:

    Trace
      system_instruction, task_prompt
      events: [Turn | Nudge | Compaction]      in order
    Turn       one model call: thinking parts, text parts, tool calls (parallel if several),
               the tool responses that answered them, the final text if it is the last turn

Grouping rule (raises when the trace does not fit it): a model-output entry (`thinking`, `text`,
`tool_call`, `final`) starts a new turn when the previous event was not model output of the same
turn, i.e. after a tool response / nudge / prompt, or when it carries `usage` while the current
turn already has content, or when its timestamp differs from the current turn's.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

COUNTED_TOOLS = frozenset({
    "run_command", "read_file", "edit_file", "write_file",
    "get_code_neighbors", "search_similar_code", "get_code_subgraph",
})
FREE_TOOLS = frozenset({"submit_patch", "get_status"})
MODEL_TYPES = frozenset({"thinking", "text", "tool_call", "final"})
KNOWN_TYPES = MODEL_TYPES | {"system_instruction", "task_prompt", "tool_response", "usage",
                             "continuation_nudge", "compaction"}

class TraceError(ValueError):
    pass


@dataclass
class ToolCall:
    entry_index: int
    tool: str
    args: dict[str, Any]
    elapsed_s: float
    response_index: int | None = None
    result: str | None = None          # raw JSON string as recorded
    response_elapsed_s: float | None = None

    @property
    def counted(self) -> bool:
        return self.tool in COUNTED_TOOLS

    def result_json(self) -> dict[str, Any]:
        if self.result is None:
            raise TraceError(f"tool call at entry {self.entry_index} has no response")
        try:
            out = json.loads(self.result)
        except json.JSONDecodeError:
            return {"status": "ok", "result": self.result}
        return out if isinstance(out, dict) else {"status": "ok", "result": out}

    @property
    def path(self) -> str | None:
        p = self.args.get("filepath") or self.args.get("file_path") or self.args.get("path")
        return str(p) if p else None


@dataclass
class Turn:
    index: int                                    # turn number (0-based)
    entry_indices: list[int] = field(default_factory=list)
    timestamp: float | None = None
    elapsed_s: float | None = None
    thinking: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    final: str | None = None
    usage: dict[str, Any] | None = None
    thinking_cleaned: bool = False                # an entry of this turn was rewritten by cleaning

    @property
    def is_parallel(self) -> bool:
        return len(self.tool_calls) > 1

    @property
    def thinking_text(self) -> str:
        return "\n\n".join(self.thinking)

    @property
    def visible_text(self) -> str:
        parts = list(self.text) + ([self.final] if self.final else [])
        return "\n\n".join(parts)


@dataclass
class Nudge:
    entry_index: int
    content: str


@dataclass
class Compaction:
    entry_index: int
    entry: dict[str, Any]


Event = Turn | Nudge | Compaction


@dataclass
class Trace:
    path: str
    instance_id: str
    attempt: int | None
    entries: list[dict[str, Any]]
    system_instruction: str
    task_prompt: str
    events: list[Event]
    duration_s: float | None

    # ------------------------------------------------------------------ views
    @property
    def turns(self) -> list[Turn]:
        return [e for e in self.events if isinstance(e, Turn)]

    def tool_calls(self) -> Iterator[ToolCall]:
        for t in self.turns:
            yield from t.tool_calls

    @property
    def n_counted_calls(self) -> int:
        return sum(1 for c in self.tool_calls() if c.counted)

    @property
    def final_text(self) -> str | None:
        for t in reversed(self.turns):
            if t.final is not None:
                return t.final
        return None

    def first_edit_call(self) -> ToolCall | None:
        """First call that modifies files inside /workspace (edit_file, write_file, or a
        run_command that looks like a write outside /tmp)."""
        for c in self.tool_calls():
            if c.tool in ("edit_file", "write_file"):
                return c
            if c.tool == "run_command" and is_workspace_write_command(c.args.get("command", "")):
                return c
        return None

    def calls_before_first_edit(self) -> list[ToolCall]:
        first = self.first_edit_call()
        out = []
        for c in self.tool_calls():
            if first is not None and c.entry_index >= first.entry_index:
                break
            out.append(c)
        return out

    def turn_model_seconds(self) -> list[float]:
        """Teacher wall time spent generating each turn (time since the previous event ended)."""
        out, prev = [], 0.0
        for ev in self.events:
            if isinstance(ev, Turn):
                out.append(max(0.0, (ev.elapsed_s or prev) - prev))
                ends = [c.response_elapsed_s for c in ev.tool_calls if c.response_elapsed_s is not None]
                prev = max(ends) if ends else (ev.elapsed_s or prev)
            elif isinstance(ev, Nudge):
                prev = self.entries[ev.entry_index].get("elapsed_s", prev)
        return out

    def turn_tool_seconds(self) -> list[float]:
        """Wall time of each turn's tool execution (parallel calls: until the last response)."""
        out = []
        for t in self.turns:
            ends = [c.response_elapsed_s for c in t.tool_calls if c.response_elapsed_s is not None]
            out.append(max(0.0, max(ends) - (t.elapsed_s or 0.0)) if ends else 0.0)
        return out


_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n.*?\n\s*\1\b", re.S)
_SAFE_PREFIXES = ("/tmp", "/dev/null", "/dev/stdout", "/dev/stderr")


def _is_safe_target(path: str) -> bool:
    return path.startswith(_SAFE_PREFIXES) or path.startswith("&")


def is_workspace_write_command(command: str) -> bool:
    """Heuristic: does a shell command modify repository files? Writes under /tmp are ignored.

    Detected: `sed -i`/`perl -i` on files, output redirection (`>`, `>>`), `tee`, `cp`/`mv` to a
    destination, `rm`, `git apply`, `patch`. Heredoc bodies are ignored.
    """
    import shlex

    if not command:
        return False
    cmd = _HEREDOC.sub("<<HEREDOC", command)
    for seg in re.split(r"&&|\|\||;|\n|(?<![>&])\|(?!\|)", cmd):
        seg = seg.strip()
        if not seg:
            continue
        # redirections anywhere in the segment
        for m in re.finditer(r"(?<![0-9&<])>{1,2}\s*([^\s;&|]+)", seg):
            if not _is_safe_target(m.group(1)):
                return True
        try:
            toks = shlex.split(seg, posix=True)
        except ValueError:
            toks = seg.split()
        if not toks:
            continue
        while toks and re.fullmatch(r"\w+=\S*", toks[0]):   # env assignments
            toks = toks[1:]
        if toks and toks[0] in ("cd", "sudo", "env", "timeout", "nohup") and len(toks) > 1:
            toks = toks[2:] if toks[0] in ("cd", "timeout") else toks[1:]
        if not toks:
            continue
        prog, args = toks[0], [t for t in toks[1:] if t != "<<HEREDOC"]
        files = [a for a in args if not a.startswith("-")]
        if prog in ("sed", "perl") and any(a.startswith("-i") or a.startswith("-pi") for a in args):
            targets = files[1:] if prog == "sed" else files[1:]
            if any(not _is_safe_target(t) for t in targets):
                return True
        elif prog == "tee":
            if any(not _is_safe_target(t) for t in files):
                return True
        elif prog in ("cp", "mv", "install") and files:
            if not _is_safe_target(files[-1]):
                return True
        elif prog in ("rm", "rmdir", "unlink", "truncate", "touch") and files:
            if any(not _is_safe_target(t) for t in files):
                return True
        elif prog == "git" and args[:1] and args[0] in ("apply", "am", "checkout", "restore", "reset", "stash", "mv", "rm"):
            if args[0] == "checkout" and "--" not in args and not any(a.endswith(".py") for a in args):
                continue
            return True
        elif prog == "patch":
            return True
    return False


# ---------------------------------------------------------------------- parsing
_ID_RE = re.compile(r"^(?:\d+_)?(?P<id>.+?)__a(?P<attempt>\d+)\.json$")


def parse_trace_name(path: str | Path) -> tuple[str, int | None]:
    name = Path(path).name
    m = _ID_RE.match(name)
    if not m:
        return Path(path).stem, None
    return m.group("id"), int(m.group("attempt"))


def load_trace(path: str | Path) -> Trace:
    path = Path(path)
    raw = json.loads(path.read_text())
    entries = raw["entries"] if isinstance(raw, dict) else raw
    instance_id, attempt = parse_trace_name(path)
    return build_trace(entries, str(path), instance_id, attempt,
                       raw.get("duration_s") if isinstance(raw, dict) else None)


def build_trace(entries: list[dict[str, Any]], path: str = "<memory>", instance_id: str = "",
                attempt: int | None = None, duration_s: float | None = None) -> Trace:
    system_instruction = task_prompt = None
    events: list[Event] = []
    cur: Turn | None = None
    pending: list[ToolCall] = []          # calls waiting for a response, in order
    last_kind = "start"                   # "model" | "response" | "nudge" | "prompt" | ...

    def close_turn() -> None:
        nonlocal cur
        if cur is not None:
            events.append(cur)
            cur = None

    for i, e in enumerate(entries):
        t = e.get("type")
        if t not in KNOWN_TYPES:
            raise TraceError(f"{path}: unknown entry type {t!r} at {i}")

        if t == "system_instruction":
            system_instruction = e["content"]
            last_kind = "prompt"
            continue
        if t == "task_prompt":
            task_prompt = e["content"]
            last_kind = "prompt"
            continue
        if t == "usage":
            if cur is not None and cur.usage is None:
                cur.usage = {k: v for k, v in e.items() if k not in ("type",)}
            continue
        if t == "continuation_nudge":
            if pending:
                raise TraceError(f"{path}: nudge at {i} while tool calls are unanswered")
            close_turn()
            events.append(Nudge(i, e["content"]))
            last_kind = "nudge"
            continue
        if t == "compaction":
            close_turn()
            events.append(Compaction(i, e))
            last_kind = "compaction"
            continue
        if t == "tool_response":
            if not pending:
                raise TraceError(f"{path}: tool_response at {i} without a pending call")
            call = pending.pop(0)
            if call.tool != e.get("tool"):
                raise TraceError(f"{path}: response at {i} is for {e.get('tool')!r}, "
                                 f"expected {call.tool!r} (call at {call.entry_index})")
            call.response_index = i
            call.result = e.get("result")
            call.response_elapsed_s = e.get("elapsed_s")
            last_kind = "response"
            continue

        # model output
        new_turn = (
            cur is None
            or last_kind != "model"
            or ("usage" in e and cur.entry_indices and cur.usage is not None)
            or (cur.timestamp is not None and e.get("timestamp") != cur.timestamp)
        )
        if new_turn:
            if pending:
                raise TraceError(f"{path}: new model turn at {i} while tool calls are unanswered")
            close_turn()
            cur = Turn(index=sum(isinstance(x, Turn) for x in events), timestamp=e.get("timestamp"),
                       elapsed_s=e.get("elapsed_s"))
        assert cur is not None
        cur.entry_indices.append(i)
        if "usage" in e and cur.usage is None:
            cur.usage = e["usage"]
        if "cleaned_from" in e:
            cur.thinking_cleaned = True
        if t == "thinking":
            cur.thinking.append(e["content"])
        elif t == "text":
            cur.text.append(e["content"])
        elif t == "final":
            cur.final = e["content"]
        elif t == "tool_call":
            call = ToolCall(i, e["tool"], dict(e.get("args") or {}), e.get("elapsed_s") or 0.0)
            cur.tool_calls.append(call)
            pending.append(call)
        last_kind = "model"

    close_turn()
    if pending:
        raise TraceError(f"{path}: {len(pending)} tool call(s) without a response at the end")
    if system_instruction is None or task_prompt is None:
        raise TraceError(f"{path}: missing system_instruction or task_prompt")
    return Trace(path, instance_id, attempt, entries, system_instruction, task_prompt, events,
                 duration_s)


def iter_traces(folder: str | Path) -> Iterator[Trace]:
    for p in sorted(Path(folder).glob("*.json")):
        yield load_trace(p)


# ---------------------------------------------------------------------- CLI
def add_args_cmd_stats(p: argparse.ArgumentParser) -> None:
    p.add_argument("folder", help="folder of trace JSON files")
    p.add_argument("--out", help="write the per-trace table as JSONL")


def trace_row(tr: Trace) -> dict[str, Any]:
    tools = Counter(c.tool for c in tr.tool_calls())
    first = tr.first_edit_call()
    return {
        "trace": Path(tr.path).name,
        "instance_id": tr.instance_id,
        "turns": len(tr.turns),
        "counted_calls": tr.n_counted_calls,
        "parallel_turns": sum(t.is_parallel for t in tr.turns),
        "nudges": sum(isinstance(e, Nudge) for e in tr.events),
        "compactions": sum(isinstance(e, Compaction) for e in tr.events),
        "calls_before_first_edit": len(tr.calls_before_first_edit()),
        "first_edit_tool": first.tool if first else None,
        "duration_s": tr.duration_s,
        "tools": dict(tools),
        "cleaned_turns": sum(t.thinking_cleaned for t in tr.turns),
    }


def cmd_stats(args: argparse.Namespace, cfg: dict) -> int:
    rows, errors = [], []
    for p in sorted(Path(args.folder).glob("*.json")):
        try:
            rows.append(trace_row(load_trace(p)))
        except TraceError as e:
            errors.append(str(e))
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    calls = [r["counted_calls"] for r in rows]
    summary = {
        "traces": len(rows),
        "errors": len(errors),
        "counted_calls": {
            "median": statistics.median(calls) if calls else None,
            "p90": sorted(calls)[int(0.9 * (len(calls) - 1))] if calls else None,
            "max": max(calls) if calls else None,
        },
        "with_parallel_calls": sum(r["parallel_turns"] > 0 for r in rows),
        "with_nudges": sum(r["nudges"] > 0 for r in rows),
        "with_compaction": sum(r["compactions"] > 0 for r in rows),
    }
    print(json.dumps(summary, indent=2))
    for e in errors[:20]:
        print("ERROR", e)
    return 1 if errors else 0
