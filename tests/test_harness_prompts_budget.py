import json

import pytest

from locfix.data.traces import load_trace
from locfix.harness.budget import (
    BudgetExceeded,
    BudgetWalker,
    StatusContext,
    attach_budget_warning,
    get_status_response,
)
from locfix.harness.prompts import Budget, build_task_prompt, parse_task_prompt, rebuild_with_budget

TEACHER_TOOLS = ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"]


def test_prompt_rebuild_is_byte_exact(all_trace_paths):
    for p in all_trace_paths:
        tr = load_trace(p)
        parts = parse_task_prompt(tr.task_prompt)
        assert parts.budget.tool_calls == 100 and parts.budget.turns == 300
        assert parts.budget.time_minutes in (7.0, 8.0)
        assert parts.workspace_tree.startswith("./")
        assert build_task_prompt(parts, declared_tools=TEACHER_TOOLS) == tr.task_prompt, p.name


def test_rebuild_changes_only_budget(astropy_path):
    tr = load_trace(astropy_path)
    new = rebuild_with_budget(tr.task_prompt, Budget(time_minutes=5, tool_calls=30, turns=300), TEACHER_TOOLS)
    assert "- Time allowance: 5 minutes\n- Tool calls allowance: 30 calls\n" in new
    old_lines = tr.task_prompt.splitlines()
    new_lines = new.splitlines()
    assert len(old_lines) == len(new_lines)
    changed = [(a, b) for a, b in zip(old_lines, new_lines, strict=True) if a != b]
    assert len(changed) == 2 and all("allowance" in a for a, _ in changed)


def test_rebuild_without_read_file_drops_its_rules(astropy_path):
    tr = load_trace(astropy_path)
    new = rebuild_with_budget(tr.task_prompt, Budget(5, 30, 300), ["run_command", "submit_patch"])
    assert "File view limit" not in new and "Command output limit" in new


def test_budget_warning_matches_harness_rule():
    raw = json.dumps({"status": "ok", "stdout": "x"}, ensure_ascii=False)
    assert attach_budget_warning(raw, used=19, max_calls=30) == raw            # 11 remaining
    w = json.loads(attach_budget_warning(raw, used=20, max_calls=30))
    assert w["budget_warning"] == ("Only 10 tool call(s) remaining (20/30 used). "
                                   "Finalize your edits and call submit_patch soon.")
    assert attach_budget_warning(raw, used=18, max_calls=19) == raw            # limit < 20: never


def test_walker_on_trace(gunicorn_path):
    tr = load_trace(gunicorn_path)
    walker = BudgetWalker(max_tool_calls=22, time_minutes=5, turns=300)
    warned = []
    for c in tr.tool_calls():
        out = walker.response(c.tool, c.result, elapsed_seconds=c.elapsed_s)
        if "budget_warning" in json.loads(out):
            warned.append(walker.used)
        if c.tool not in ("get_status", "submit_patch"):
            assert json.loads(out)["status"] == json.loads(c.result)["status"]
    assert walker.used == 12
    assert warned == [12]                                                      # 22 - 12 = 10 left


def test_walker_refuses_over_budget(exceptiongroup_path):
    tr = load_trace(exceptiongroup_path)
    walker = BudgetWalker(max_tool_calls=20, time_minutes=5, turns=300)
    with pytest.raises(BudgetExceeded):
        for c in tr.tool_calls():
            walker.response(c.tool, c.result, c.elapsed_s)


def test_get_status_fields_order():
    out = json.loads(get_status_response(StatusContext(12, 30, 5, 300, elapsed_seconds=100.0)))
    assert list(out) == ["status", "tool_calls_used", "patch_submitted", "patch_size",
                         "tool_calls_remaining", "max_tool_calls", "time_seconds_remaining",
                         "max_time_minutes", "agent_elapsed_seconds", "max_turns",
                         "command_timeout_seconds"]
    assert out["tool_calls_remaining"] == 18 and out["time_seconds_remaining"] == 200.0
