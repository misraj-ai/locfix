"""Scripted replay of a trajectory through the real agent stack.

The agent YAML is compiled by `adk_submission.compile_submission`, the model is ADK's `LiteLlm`
(with the harness's reasoning patch) pointed at a `ScriptedServer`, and the tools are the harness's
own tool callables (same names, signatures and docstrings, hence the same declarations) whose
bodies return recorded outputs instead of touching a sandbox. The harness's outer loop (run until a
text-only turn, then nudge or stop) is reproduced. Every request the model would receive is
captured; those requests are the training contexts.
"""
from __future__ import annotations

import asyncio
import functools
import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

from .scripted_server import Exchange, ScriptedServer, chat_response

SUMMARIZER_PREFIX = "The following is a conversation history between a user and an AI agent."


class ReplayError(RuntimeError):
    pass


@dataclass
class ScriptCall:
    tool: str
    args: dict[str, Any]
    result: str                              # raw JSON string the tool returns (already budget-adjusted)


@dataclass
class ScriptTurn:
    thinking: str | None = None
    text: str | None = None
    calls: list[ScriptCall] = field(default_factory=list)
    finish_reason: str | None = None         # e.g. "length" to reproduce a truncation nudge
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScriptNudge:
    content: str


Step = ScriptTurn | ScriptNudge


@dataclass
class ReplayResult:
    exchanges: list[Exchange]                # every request/response, in order
    agent_exchanges: list[Exchange]          # only the agents' own model calls (no summarizer)
    turn_meta: list[dict[str, Any]]          # meta of the ScriptTurn answering each agent exchange
    initial_prompt: str
    streams: list[str | None] = field(default_factory=list)   # agent name per agent exchange

    def of_stream(self, name: str | None) -> ReplayResult:
        """The exchanges of one agent (e.g. only the localizer sub-agent's model calls)."""
        keep = [i for i, s in enumerate(self.streams) if s == name]
        return ReplayResult(self.exchanges, [self.agent_exchanges[i] for i in keep],
                            [self.turn_meta[i] for i in keep], self.initial_prompt, [name] * len(keep))


def estimate_prompt_tokens(req: dict[str, Any]) -> int:
    """Rough prompt size (4 characters per token, like ADK's own estimate). Use the real Gemma
    tokenizer (`locfix.render.chat.prompt_token_counter`) whenever compaction matters."""
    chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in req.get("messages", []))
    chars += sum(len(json.dumps(t, ensure_ascii=False)) for t in req.get("tools", []))
    return chars // 4


def static_summarizer(text: str = "Summary of the earlier work.") -> Callable[[dict], dict]:
    """Summarizer for tests: always returns `text`."""
    def handle(req: dict[str, Any]) -> dict[str, Any]:
        return chat_response(req.get("model", "scripted"), text, None, None)
    return handle


def forwarding_summarizer(base_url: str, model: str, timeout: float = 600,
                          **sampling: Any) -> Callable[[dict], dict]:
    """Summarizer that forwards the summary request to a real OpenAI-compatible server (vLLM
    serving the base model or the current adapter) with fixed sampling for reproducibility."""
    import urllib.request

    def handle(req: dict[str, Any]) -> dict[str, Any]:
        body = dict(req, model=model, **({"temperature": 0.0, "seed": 0} | sampling))
        body.pop("stream", None)
        r = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                   data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read())
    return handle


def _is_summarizer_request(req: dict[str, Any]) -> bool:
    for m in req.get("messages", []):
        c = m.get("content")
        text = c if isinstance(c, str) else json.dumps(c)
        if SUMMARIZER_PREFIX in text:
            return True
    return False


_AGENT_NAME = re.compile(r'Your internal name is "([^"]+)"')


def agent_name_of(req: dict[str, Any]) -> str | None:
    """Name of the agent that sent a model request (ADK appends it to the system instruction)."""
    for m in req.get("messages", []):
        if m.get("role") != "system":
            continue
        c = m.get("content")
        text = c if isinstance(c, str) else json.dumps(c)
        found = _AGENT_NAME.findall(text)
        if found:
            return found[-1]
    return None


def _normalize_args(args: dict[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False)


class _Player:
    """Serves scripted turns to the model endpoint and recorded results to the tools.

    `steps` is one script (any agent may ask) or {agent name: script} when sub-agents run as tools:
    requests are routed by the agent name ADK puts in the system instruction. Execution is
    sequential (a parent waits for its `agent_tool` call), so one queue of pending tool calls
    suffices. Calls to tools that are not harness tools are `agent_tool` calls: ADK runs the
    sub-agent itself, so they are not served here; their `result` is ignored."""

    def __init__(self, steps: list[Step] | dict[str, list[Step]], summarizer: Callable[[dict], dict] | None,
                 prompt_token_counter: Callable[[dict], int] | None = None,
                 harness_tools: set[str] | None = None):
        if isinstance(steps, dict):
            self.streams: dict[str | None, deque[Step]] = {k: deque(v) for k, v in steps.items()}
        else:
            self.streams = {None: deque(steps)}
        self.summarizer = summarizer
        self.count_tokens = prompt_token_counter or estimate_prompt_tokens
        self.harness_tools = harness_tools
        self.pending_calls: deque[ScriptCall] = deque()
        self.turn_meta: list[dict[str, Any]] = []
        self.stream_of_turn: list[str | None] = []
        self.agent_calls: list[tuple[str, dict[str, Any]]] = []
        self.n_turns = 0

    def queue(self, name: str | None) -> deque[Step]:
        if None in self.streams:
            return self.streams[None]
        if name not in self.streams:
            raise ReplayError(f"model request from agent {name!r}, which has no script "
                              f"(scripts: {sorted(map(str, self.streams))})")
        return self.streams[name]

    @property
    def steps(self) -> deque[Step]:
        """The root script (single-script replays)."""
        return next(iter(self.streams.values()))

    def is_agent_call(self, tool: str) -> bool:
        return self.harness_tools is not None and tool not in self.harness_tools

    # model side --------------------------------------------------------------------
    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        if _is_summarizer_request(req):
            if self.summarizer is None:
                raise ReplayError("compaction summary requested but no summarizer is configured")
            return self.summarizer(req)
        if self.pending_calls:
            raise ReplayError(f"model called again with {len(self.pending_calls)} tool call(s) unanswered")
        name = agent_name_of(req)
        steps = self.queue(name)
        if not steps or not isinstance(steps[0], ScriptTurn):
            nxt = type(steps[0]).__name__ if steps else "end of script"
            raise ReplayError(f"model request #{self.n_turns + 1} ({name}) but the script expects {nxt}")
        turn: ScriptTurn = steps.popleft()  # type: ignore[assignment]
        tool_calls = []
        agent_calls = [c for c in turn.calls if self.is_agent_call(c.tool)]
        if agent_calls and len(turn.calls) > 1:
            raise ReplayError("an agent_tool call must be the only call of its turn")
        for j, c in enumerate(turn.calls):
            tool_calls.append({"id": f"call_{self.n_turns}_{j}", "type": "function",
                               "function": {"name": c.tool, "arguments": json.dumps(c.args, ensure_ascii=False)}})
            if self.is_agent_call(c.tool):
                self.agent_calls.append((c.tool, c.args))
            else:
                self.pending_calls.append(c)
        self.turn_meta.append(turn.meta)
        self.stream_of_turn.append(name)
        self.n_turns += 1
        n = int(self.count_tokens(req))
        return chat_response(req.get("model", "scripted"), turn.text, turn.thinking, tool_calls or None,
                             finish_reason=turn.finish_reason,
                             usage={"prompt_tokens": n, "completion_tokens": 0, "total_tokens": n})

    # tool side ---------------------------------------------------------------------
    def tool_result(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        from swegemma.tools.base import parse_tool_response

        if not self.pending_calls:
            raise ReplayError(f"unexpected tool call {name}({args})")
        call = self.pending_calls.popleft()
        if call.tool != name or _normalize_args(call.args) != _normalize_args(args):
            raise ReplayError(f"tool call mismatch: expected {call.tool}({call.args}), got {name}({args})")
        return parse_tool_response(call.result)


class _Ctx:
    """Placeholder context: tool factories only capture it; replayed tools never use it."""


def replay_tools(player: _Player, names: list[str] | None = None) -> dict[str, Callable]:
    """Harness tool callables with their real signatures/docstrings, returning recorded results."""
    from swegemma.tools import create_tools

    real = create_tools(_Ctx())
    out: dict[str, Callable] = {}
    for name, fn in real.items():
        if names is not None and name not in names:
            continue

        def make(tool_name: str, real_fn: Callable) -> Callable:
            @functools.wraps(real_fn)
            def replayed(*args: Any, **kwargs: Any) -> dict[str, Any]:
                if args:
                    raise ReplayError(f"{tool_name} called with positional args")
                return player.tool_result(tool_name, kwargs)
            return replayed

        out[name] = make(name, fn)
    return out


async def replay_async(
    steps: list[Step] | dict[str, list[Step]],
    agent_dir: str | Path,
    initial_prompt: str,
    problem_statement: str,
    model_alias: str,
    hints: str | None = None,
    compaction: dict[str, Any] | None = None,
    summarizer: Callable[[dict], dict] | None = None,
    max_llm_calls: int = 500,
    prompt_token_counter: Callable[[dict], int] | None = None,
) -> ReplayResult:
    from adk_submission import compile_submission
    from adk_submission.registry import ModelRegistry
    from adk_submission.resolvers.generation import install_litellm_reasoning_patch
    from google.adk.agents.run_config import RunConfig
    from google.adk.apps.app import App
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types
    from swegemma.harness.agent_runner import ToolErrorPlugin, resolve_swegemma_adapter

    from swegemma.tools import create_tools

    player = _Player(steps, summarizer, prompt_token_counter, harness_tools=set(create_tools(_Ctx())))
    with ScriptedServer(player.handle) as server:
        install_litellm_reasoning_patch()
        models = ModelRegistry()
        models.register(model_alias, LiteLlm(model="openai/scripted", api_base=server.base_url,
                                             api_key="EMPTY"))
        agent = compile_submission(
            submission_dir=str(agent_dir),
            tool_registry=replay_tools(player),
            model_registry=models,
            adapter_resolver_fn=lambda base, info: resolve_swegemma_adapter(base, info, models),
        )
        app_kwargs: dict[str, Any] = {"name": "swegemma_eval", "root_agent": agent,
                                      "plugins": [ToolErrorPlugin()]}
        if compaction:
            from google.adk.apps.app import EventsCompactionConfig

            app_kwargs["events_compaction_config"] = EventsCompactionConfig(**compaction)
        runner = Runner(app=App(**app_kwargs), session_service=InMemorySessionService())
        state = {"problem_description": problem_statement}
        if hints and hints.strip():
            state["hints"] = hints.strip()
        session = await runner.session_service.create_session(app_name="swegemma_eval",
                                                              user_id="eval_user", state=state)
        message = genai_types.Content(role="user", parts=[genai_types.Part(text=initial_prompt)])
        while True:
            try:
                async for _ in runner.run_async(user_id="eval_user", session_id=session.id,
                                                new_message=message,
                                                run_config=RunConfig(max_llm_calls=max_llm_calls)):
                    pass
            except Exception as e:
                # the scripted server answers a script error with HTTP 500, which LiteLLM raises as
                # InternalServerError: report the script error itself
                errors = [x.error for x in server.exchanges if x.error]
                if errors:
                    raise ReplayError(errors[0]) from e
                raise
            errors = [e.error for e in server.exchanges if e.error]
            if errors:
                raise ReplayError(errors[0])
            root = player.queue(agent.name)
            if not root:
                break
            step = root.popleft()
            if not isinstance(step, ScriptNudge):
                raise ReplayError("agent stopped (text-only turn) but the script continues with a model turn")
            message = genai_types.Content(role="user", parts=[genai_types.Part(text=step.content)])
        if player.pending_calls:
            raise ReplayError(f"{len(player.pending_calls)} scripted tool call(s) were never made")
        left = {k: len(v) for k, v in player.streams.items() if v}
        if left:
            raise ReplayError(f"script steps never used: {left}")
        exchanges = list(server.exchanges)

    agent_ex = [e for e in exchanges if not _is_summarizer_request(e.request)]
    return ReplayResult(exchanges, agent_ex, player.turn_meta, initial_prompt, player.stream_of_turn)


def replay(*args: Any, **kwargs: Any) -> ReplayResult:
    """Synchronous wrapper. Inside a running event loop (Jupyter/Colab) the replay runs in a worker
    thread with its own loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(replay_async(*args, **kwargs))
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, replay_async(*args, **kwargs)).result()
