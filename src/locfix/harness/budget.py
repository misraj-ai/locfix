"""Budget-dependent parts of tool responses, produced by the harness's own functions.

- `attach_budget_warning`: runs `swegemma.tools.base._attach_low_budget_warning` on a recorded
  tool response, as if `used` counted calls had been made out of `max_calls`.
- `get_status_response`: runs `swegemma.tools.execution.get_status` on a stand-in context.
- `BudgetWalker`: walks the tool calls of a trajectory in order and rewrites each response the way
  the harness would have produced it under a different budget.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from ..data.traces import COUNTED_TOOLS


def attach_budget_warning(raw_response: str, used: int, max_calls: int | None) -> str:
    from swegemma.tools.base import _attach_low_budget_warning

    ctx = SimpleNamespace(budget=SimpleNamespace(tool_calls=max_calls), tool_calls_used=used)
    return _attach_low_budget_warning(ctx, raw_response)


@dataclass
class StatusContext:
    """Attributes `swegemma.tools.execution.get_status` reads from its context."""

    tool_calls_used: int
    max_tool_calls: int | None
    time_minutes: float | None
    turns: int | None
    elapsed_seconds: float
    submitted_patch: str | None = None
    command_timeout_seconds: int | None = 300
    budget: Any = field(init=False)
    harness: Any = field(init=False)

    def __post_init__(self) -> None:
        self.budget = SimpleNamespace(tool_calls=self.max_tool_calls, time_minutes=self.time_minutes,
                                      turns=self.turns, cost_usd=None, total_tokens=None)
        self.harness = SimpleNamespace(command_timeout_seconds=self.command_timeout_seconds)
        self.tool_lock = None

    @property
    def remaining_time_seconds(self) -> float | None:
        if self.time_minutes is None:
            return None
        return max(0.0, self.time_minutes * 60.0 - self.elapsed_seconds)

    @property
    def agent_elapsed_seconds(self) -> float:
        return max(0.0, self.elapsed_seconds)

    def check_budget(self, *, count_tool_call: bool = True) -> str | None:  # noqa: ARG002
        return None


def get_status_response(ctx: StatusContext) -> str:
    from swegemma.tools.execution import get_status

    return get_status(ctx)


@dataclass
class BudgetWalker:
    """Rewrites recorded tool responses for a new budget, call by call, in trajectory order."""

    max_tool_calls: int | None
    time_minutes: float | None
    turns: int | None
    command_timeout_seconds: int | None = 300
    used: int = 0
    submitted_patch: str | None = None

    def exceeds(self, tool: str) -> bool:
        """Would this call be refused (budget exhausted) under the new budget?"""
        return (tool in COUNTED_TOOLS and self.max_tool_calls is not None
                and self.used >= self.max_tool_calls)

    def response(self, tool: str, recorded: str, elapsed_seconds: float) -> str:
        """The response the harness would return for this call. Raises if over budget."""
        if self.exceeds(tool):
            raise BudgetExceeded(f"call {self.used + 1} exceeds max_tool_calls={self.max_tool_calls}")
        if tool in COUNTED_TOOLS:
            self.used += 1
            # recorded responses may already carry a warning for the teacher's budget: drop it
            return attach_budget_warning(strip_budget_warning(recorded), self.used, self.max_tool_calls)
        if tool == "get_status":
            ctx = StatusContext(self.used, self.max_tool_calls, self.time_minutes, self.turns,
                                elapsed_seconds, self.submitted_patch, self.command_timeout_seconds)
            return get_status_response(ctx)
        if tool == "submit_patch":
            # get_status reports len(submitted_patch); keep a placeholder of the recorded size
            try:
                size = int(json.loads(recorded).get("patch_size", 0))
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                size = 0
            self.submitted_patch = "x" * size
        return recorded


class BudgetExceeded(RuntimeError):
    pass


def strip_budget_warning(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(payload, dict) and "budget_warning" in payload:
        payload.pop("budget_warning")
        return json.dumps(payload, ensure_ascii=False)
    return raw
