"""Offline fixer rollouts for the Stage 3 signal test (spec 06 §8, adapted).

The Kaggle dev tasks have no teacher labels and the training tasks are not evaluated in the harness
(no Docker on the training node), so the signal test cannot measure "resolved". Instead the fixer
**writes its own edit** for a training task of the other fold, and that edit is compared with the
verified patch y*.

One rollout runs the real agent stack (`adk_submission.compile_submission` of `agents/locfix`, ADK,
LiteLLM, the harness's tool declarations), exactly as `score.prepare` replays it, with:

- the localizer sub-agent scripted: the teacher's exploration, then the answer **z** under test;
- the fixer's first turns scripted as in the training records: call the localizer, read z
  (mode `none`: nothing else);
- then the fixer **live**: every further fixer request goes to vLLM serving the fold fixer (greedy,
  temperature 0), and its tool calls are executed offline:
    read_file / edit_file / write_file   the harness's own tools on the task's files (`LocalWorkspace`;
                                         edits change the files, later reads see them)
    get_status                           the harness's own, from the shared budget walker
    run_command                          before any edit, only a command the teacher ran before its
                                         first edit (its recorded output); anything else cannot run
                                         offline
  Budget warnings come from the same walker as in training.

A rollout stops at the first call that cannot run offline (a command, an unknown file, a second
localizer call, `submit_patch`, a budget limit), at a text-only answer, or after `max_model_calls`
live fixer turns. Edits of the stopping turn that can be applied are applied (for the metrics only).
The fixer usually edits right after reading z and then runs tests, so the edits made before the
first command are what is measured.

Metrics, against y* (`patch_metrics`):
    edited          the fixer changed at least one file
    first_edit_loc  its first edit changes lines of a file y* changes, within 3 lines of a y* hunk
    hunks_matched   y* hunks reproduced exactly (same base lines replaced, same new lines; trailing
                    whitespace ignored); hunks = changed-line groups of the diff (gap ≤ 2 lines)
    all_hunks       every y* hunk reproduced  (the primary measure)
    any_hunk        at least one y* hunk reproduced
    added_recall    share of y*'s added lines that appear among the fixer's added lines, per file
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from ..data.edits import _change_groups, apply_patch
from ..data.loc_fix_records import FIXER, LocFixConfig, Task, TaskLocations
from ..data.locations import Location, normalize_path
from ..data.traces import is_workspace_write_command
from ..harness.replay import ReplayError, _Ctx, _normalize_args, _Player, agent_name_of
from ..harness.scripted_server import chat_response

STOP_TEXT = "(rollout stopped)"
LIVE_TOOLS = ("read_file", "edit_file", "write_file", "get_status", "run_command")


# --------------------------------------------------------------------------- offline tools
class OfflineTools:
    def __init__(self, task: Task, tl: TaskLocations, walker, elapsed_seconds: float):
        from ..harness.local_sandbox import LocalWorkspace

        self.ws = LocalWorkspace(task.files)
        self.files = set(task.files)
        self.walker = walker
        self.elapsed = elapsed_seconds
        self.recorded = {c.args.get("command", ""): c.result or "" for t in tl.pre_turns for c in t.tool_calls
                         if c.tool == "run_command" and not is_workspace_write_command(c.args.get("command", ""))}
        self.edits: list[dict[str, Any]] = []       # {tool, args, ok, applied_only}
        self.calls: list[dict[str, Any]] = []       # every live call: {tool, args?, served}
        self.touched: list[str] = []

    def close(self) -> None:
        self.ws.close()

    @property
    def edited(self) -> bool:
        return any(e["ok"] for e in self.edits)

    def unservable(self, tool: str, args: dict[str, Any]) -> str | None:
        """Why this call cannot run offline (None: it can)."""
        if tool not in LIVE_TOOLS:
            return "localizer_again" if tool == "localizer" else f"tool:{tool}"
        if self.walker.exceeds(tool):
            return "budget"
        if tool in ("read_file", "edit_file"):
            p = normalize_path(str(args.get("filepath", "")))
            if p not in self.files:
                return "unknown_file"
        if tool == "write_file" and not str(args.get("filepath", "")).strip():
            return "bad_args"
        if tool == "run_command":
            cmd = args.get("command", "")
            if self.edited:
                return "command_after_edit"
            if cmd not in self.recorded:
                return "command"
        return None

    def _apply(self, tool: str, args: dict[str, Any]) -> str:
        out = self.ws.call(tool, args)
        ok = False
        try:
            ok = json.loads(out).get("status") == "ok"
        except (json.JSONDecodeError, AttributeError):
            pass
        p = normalize_path(str(args["filepath"]))
        if ok:
            self.files.add(p)
            if p not in self.touched:
                self.touched.append(p)
        self.edits.append({"tool": tool, "args": args, "ok": ok})
        return out

    def run(self, tool: str, args: dict[str, Any]) -> str:
        self.calls.append({"tool": tool, "served": True})
        if tool in ("edit_file", "write_file"):
            out = self._apply(tool, args)
        elif tool == "read_file":
            out = self.ws.call(tool, args)
        elif tool == "run_command":
            out = self.recorded[args.get("command", "")]
        else:                                      # get_status: produced by the walker
            out = ""
        return self.walker.response(tool, out, elapsed_seconds=self.elapsed)

    def apply_only(self, tool: str, args: dict[str, Any]) -> None:
        """An edit of the stopping turn: applied for the metrics, never shown to the model."""
        try:
            self._apply(tool, args)
            self.edits[-1]["applied_only"] = True
        except Exception:  # noqa: BLE001 - a malformed edit counts as not applied
            self.edits.append({"tool": tool, "args": args, "ok": False, "applied_only": True})

    def text(self, path: str) -> str | None:
        return self.ws.read_text(path) if self.ws.exists(path) else None


# --------------------------------------------------------------------------- player
class RolloutPlayer(_Player):
    """Scripted streams first; once the root (fixer) script is used up, fixer requests go to
    `model_fn` and its tool calls run on `tools`."""

    def __init__(self, steps: dict[str, list], model_fn: Callable[[dict], dict], tools: OfflineTools,
                 root: str = FIXER, max_model_calls: int = 8):
        from swegemma.tools import create_tools

        super().__init__(steps, None, None, harness_tools=set(create_tools(_Ctx())))
        self.model_fn = model_fn
        self.tools = tools
        self.root = root
        self.max_model_calls = max_model_calls
        self.live_pending: deque[tuple[str, dict[str, Any]]] = deque()
        self.live_requests: list[dict[str, Any]] = []
        self.live_responses: list[dict[str, Any]] = []
        self.stop: str | None = None
        self.error: str | None = None

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        name = agent_name_of(req)
        if name == self.root and not self.pending_calls and not self.queue(name):
            return self._live(req)
        return super().handle(req)

    def _stopped(self, req: dict[str, Any], reason: str) -> dict[str, Any]:
        if self.stop is None:
            self.stop = reason
        return chat_response(req.get("model", "rollout"), STOP_TEXT, None, None)

    def _live(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.stop is not None:
            return self._stopped(req, self.stop)
        if self.live_pending:
            return self._stopped(req, "tool_not_run")
        if len(self.live_requests) >= self.max_model_calls:
            return self._stopped(req, "max_model_calls")
        self.live_requests.append(req)
        try:
            resp = self.model_fn(req)
        except Exception as e:  # noqa: BLE001 - recorded in the result
            self.error = f"{type(e).__name__}: {e}"[:500]
            return self._stopped(req, "model_error")
        msg = resp["choices"][0]["message"]
        r = msg.get("reasoning") or msg.get("reasoning_content")
        if r:                                     # the scripted server streams `reasoning`
            msg["reasoning"] = msg["reasoning_content"] = r
        self.live_responses.append(msg)
        if resp["choices"][0].get("finish_reason") == "length":
            return self._stopped(req, "length")
        calls = msg.get("tool_calls") or []
        if not calls:
            self.stop = "text_only"
            return resp
        parsed = []
        for c in calls:
            try:
                a = c["function"].get("arguments") or "{}"
                parsed.append((c["function"]["name"], json.loads(a) if isinstance(a, str) else a))
            except (KeyError, json.JSONDecodeError):
                return self._stopped(req, "bad_args")
        why = [self.tools.unservable(n, a) for n, a in parsed]
        if any(why):
            for (n, a), w in zip(parsed, why, strict=True):
                if w is None and n in ("edit_file", "write_file"):
                    self.tools.apply_only(n, a)
            for (n, a), w in zip(parsed, why, strict=True):
                self.tools.calls.append({"tool": n, "served": w is None and n in ("edit_file", "write_file"),
                                         "stopped": w, **({"command": a.get("command")} if n == "run_command" else {})})
            return self._stopped(req, next(w for w in why if w))
        self.live_pending.extend(parsed)
        return resp

    def tool_result(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.pending_calls:
            return super().tool_result(name, args)
        from swegemma.tools.base import parse_tool_response

        for i, (n, a) in enumerate(self.live_pending):
            if n == name and _normalize_args(a) == _normalize_args(args):
                del self.live_pending[i]
                return parse_tool_response(self.tools.run(name, args))
        raise ReplayError(f"live tool call {name}({args}) was not requested by the model")


# --------------------------------------------------------------------------- metrics
def _norm(lines: list[str]) -> tuple[str, ...]:
    return tuple(x.rstrip() for x in lines)


def hunks(path: str, old: str | None, new: str) -> list[tuple]:
    """(path, i1, i2, new lines) change groups of old -> new (a new file is one hunk)."""
    if old is None:
        return [(path, 0, 0, _norm(new.splitlines()))]
    a, b = old.splitlines(), new.splitlines()
    return [(path, i1, i2, _norm(b[j1:j2])) for i1, i2, j1, j2 in _change_groups(a, b)]


def _added(old: str | None, new: str) -> list[str]:
    a = set(x.strip() for x in (old or "").splitlines())
    return [x.strip() for x in new.splitlines() if x.strip() and x.strip() not in a]


def patch_metrics(task: Task, tools: OfflineTools) -> dict[str, Any]:
    after = apply_patch(task.files, task.patch)
    y = {p: t for p, t in after.items() if t is not None}
    y_h = [h for p, t in y.items() for h in hunks(p, task.files.get(p), t)]
    final = {p: tools.text(p) for p in tools.touched}
    f_h = {h for p, t in final.items() if t is not None for h in hunks(p, task.files.get(p), t)}
    matched = sum(1 for h in y_h if h in f_h)
    # first edit: its own change groups on the base files
    first_loc = False
    first = next((e for e in tools.edits if e["ok"]), None)
    if first is not None:
        from ..harness.local_sandbox import LocalWorkspace

        p = normalize_path(str(first["args"]["filepath"]))
        with LocalWorkspace(task.files) as ws:
            ws.call(first["tool"], first["args"])
            new = ws.read_text(p) if ws.exists(p) else None
        if new is not None and p in y:
            fh = hunks(p, task.files.get(p), new)
            yr = [(h[1], h[2]) for h in y_h if h[0] == p]
            first_loc = any(ya - 3 <= fb and fa <= yb + 3 for _, fa, fb, _ in fh for ya, yb in yr)
    y_add = {p: _added(task.files.get(p), t) for p, t in y.items()}
    f_add = {p: set(_added(task.files.get(p), t)) for p, t in final.items() if t is not None}
    n_add = sum(len(v) for v in y_add.values())
    hit = sum(1 for p, v in y_add.items() for x in v if x in f_add.get(p, set()))
    return {"edited": tools.edited, "n_edits": sum(e["ok"] for e in tools.edits),
            "first_edit_loc": first_loc, "y_hunks": len(y_h), "hunks_matched": matched,
            "all_hunks": bool(y_h) and matched == len(y_h), "any_hunk": matched > 0,
            "added_recall": round(hit / n_add, 4) if n_add else None,
            "files_touched": list(tools.touched)}


# --------------------------------------------------------------------------- one rollout
@dataclass
class RolloutResult:
    stop: str
    metrics: dict[str, Any]
    model_calls: int
    calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    first_request: dict[str, Any] | None = None

    def to_json(self, keep_request: bool = False) -> dict[str, Any]:
        d = {"stop": self.stop, "model_calls": self.model_calls, "calls": self.calls, "error": self.error,
             **self.metrics}
        if keep_request:
            d["first_request"] = self.first_request
        return d


def rollout(task: Task, tl: TaskLocations, z: list[Location], lc: LocFixConfig, model_fn: Callable[[dict], dict],
            max_model_calls: int = 8) -> RolloutResult:
    """The fold fixer's own attempt given z (see the module docstring)."""
    from ..score.sequences import SequenceError, build_steps, run_replay

    try:
        scripts, meta = build_steps(task, tl, z, "none", lc, edits=[], with_edits=False)
    except SequenceError as e:
        return RolloutResult(f"prefix:{e.reason}", {}, 0, error=str(e)[:300])
    t_last = max([c.elapsed_s for t in tl.pre_turns for c in t.tool_calls] or [0.0]) * lc.time_scale
    tools = OfflineTools(task, tl, meta["walker"], t_last)
    player = RolloutPlayer(scripts, model_fn, tools, FIXER, max_model_calls)
    try:
        run_replay(task, scripts, lc, player=player)
        stop = player.stop or "agent_ended"
        err = player.error
    except Exception as e:  # noqa: BLE001 - a broken rollout is reported, not fatal
        stop, err = "replay_error", f"{type(e).__name__}: {e}"[:500]
    try:
        m = patch_metrics(task, tools)
    finally:
        tools.close()
    return RolloutResult(stop, m, len(player.live_requests), tools.calls, err,
                         player.live_requests[0] if player.live_requests else None)


def http_model_fn(base_url: str, model: str, seed: int = 0, timeout: float = 1800) -> Callable[[dict], dict]:
    """Forward a fixer request (as LiteLLM sent it) to the vLLM server: the fold fixer's adapter,
    greedy decoding."""
    import urllib.request

    def fn(req: dict[str, Any]) -> dict[str, Any]:
        body = dict(req, model=model, temperature=0.0, seed=seed)
        for k in ("stream", "stream_options"):
            body.pop(k, None)
        r = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read())
    return fn
