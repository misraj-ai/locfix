"""The harness's initial user prompt, rebuilt with the harness's own `build_agent_prompt`.

`parse_task_prompt` splits a recorded task prompt into its inputs (repo, problem statement, hints,
budget, sandbox-testing flag, workspace tree); `build_task_prompt` calls
`swegemma.harness.agent_runner.build_agent_prompt` with those inputs and a (possibly different)
budget and tool set. Rebuilding with the original inputs must give the recorded prompt byte for
byte (tested), which is what makes rebuilding with our budget trustworthy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from collections.abc import Iterable

HEADER_RE = re.compile(r"\AYou are evaluating a software engineering task for repository (?P<repo>.+?)\.\n\nProblem Statement:\n")
TREE_INTRO = "## Workspace Layout\nThe repository is located at `/workspace`. Here is the directory tree (up to 3 levels):\n```\n"
SANDBOX_TESTING_OFF = "3. The `pytest` and `unittest` test frameworks are intentionally disabled"


class PromptError(ValueError):
    pass


@dataclass
class Budget:
    time_minutes: float | None = None
    tool_calls: int | None = None
    turns: int | None = None
    cost_usd: float | None = None


@dataclass
class HarnessLimits:
    command_timeout_seconds: int | None = 300
    max_stdout_chars: int | None = 5000
    max_file_lines: int | None = 150
    max_file_chars: int | None = 10000


@dataclass
class TaskPromptParts:
    repo: str
    problem_statement: str
    hints: str = ""
    budget: Budget = field(default_factory=Budget)
    enable_sandbox_testing: bool = True
    workspace_tree: str = ""
    has_code_intelligence: bool = False


def _num(s: str) -> float | int:
    return float(s) if "." in s else int(s)


def parse_task_prompt(text: str) -> TaskPromptParts:
    m = HEADER_RE.match(text)
    if not m:
        raise PromptError("not a swegemma task prompt (header not found)")
    rest = text[m.end():]
    # the problem statement runs until the first section the builder may add after it
    cut = len(rest)
    for marker in ("\n\n## Hints:\n", "\n\n## Task Budget", "\n\n## Execution Environment Rules",
                   "\n\n## Instructions:"):
        i = rest.find(marker)
        if i != -1:
            cut = min(cut, i)
    problem = rest[:cut]
    tail = rest[cut:]

    hints = ""
    hm = re.search(r"\n\n## Hints:\n(.*?)\n\n## ", tail, re.S)
    if hm:
        hints = hm.group(1)

    budget = Budget()
    bm = re.search(r"## Task Budget \(Session terminates when any budget is exhausted\)\n(.*?)\n\n", tail, re.S)
    if bm:
        for line in bm.group(1).splitlines():
            if mm := re.match(r"- Time allowance: ([\d.]+) minutes", line):
                budget.time_minutes = _num(mm.group(1))
            elif mm := re.match(r"- Tool calls allowance: (\d+) calls", line):
                budget.tool_calls = int(mm.group(1))
            elif mm := re.match(r"- Max loop iterations: (\d+) turns", line):
                budget.turns = int(mm.group(1))
            elif mm := re.match(r"- Cost budget: \$([\d.]+) USD", line):
                budget.cost_usd = float(mm.group(1))

    tree = ""
    ti = tail.find(TREE_INTRO)
    if ti != -1:
        body = tail[ti + len(TREE_INTRO):]
        end = body.rfind("\n```\n")
        if end == -1:
            raise PromptError("unterminated workspace tree block")
        tree = body[:end]

    return TaskPromptParts(
        repo=m.group("repo"),
        problem_statement=problem,
        hints=hints,
        budget=budget,
        enable_sandbox_testing=SANDBOX_TESTING_OFF not in tail,
        workspace_tree=tree,
        has_code_intelligence="## Code Intelligence Tools" in tail,
    )


def build_task_prompt(
    parts: TaskPromptParts,
    budget: Budget | None = None,
    declared_tools: Iterable[str] | None = None,
    limits: HarnessLimits | None = None,
) -> str:
    """Call the harness's `build_agent_prompt` (graph tools section never added: graph data
    paths are not configured, matching agents without graph tools)."""
    from swegemma.harness.agent_runner import build_agent_prompt

    b = budget or parts.budget
    lim = limits or HarnessLimits()
    task = SimpleNamespace(repo=parts.repo, problem_statement=parts.problem_statement,
                           hints_text=parts.hints, base_commit=None)
    config = SimpleNamespace(
        budget=SimpleNamespace(time_minutes=b.time_minutes, tool_calls=b.tool_calls,
                               turns=b.turns, cost_usd=b.cost_usd),
        harness=SimpleNamespace(**lim.__dict__),
        graph_dir=None, embeddings_dir=None,
        enable_sandbox_testing=parts.enable_sandbox_testing,
    )
    return build_agent_prompt(task, config, parts.workspace_tree,
                              enable_sandbox_testing=parts.enable_sandbox_testing,
                              declared_tools=list(declared_tools) if declared_tools is not None else None)


def rebuild_with_budget(recorded_prompt: str, budget: Budget, declared_tools: Iterable[str],
                        problem_statement: str | None = None, hints: str | None = None) -> str:
    """Recorded teacher prompt -> the prompt our agent would get under `budget`."""
    parts = parse_task_prompt(recorded_prompt)
    if parts.has_code_intelligence:
        raise PromptError("recorded prompt has a Code Intelligence section; graph tools are not supported here")
    if problem_statement is not None:
        parts.problem_statement = problem_statement
    if hints is not None:
        parts.hints = hints
    return build_task_prompt(parts, budget=budget, declared_tools=declared_tools)
