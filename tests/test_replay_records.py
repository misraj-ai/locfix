"""Scripted replay through the real ADK + LiteLLM + harness stack (CPU, no model)."""
import json
from pathlib import Path

import jsonschema
import pytest

from locfix.data.agent_records import (
    BuildConfig,
    build_agent_records,
    history_echo,
    script_from_trace,
)
from locfix.data.traces import load_trace
from locfix.harness.budget import BudgetExceeded
from locfix.harness.prompts import Budget
from locfix.harness.replay import ReplayError, ScriptCall, ScriptNudge, ScriptTurn, replay

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas" / "record.schema.json").read_text())
ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
AGENT = str(ROOT / "agents" / "single")


DEFAULT_BUDGET = Budget(5, 30, 300)


def build(path, budget=DEFAULT_BUDGET):
    tr = load_trace(path)
    return tr, build_agent_records(tr, BuildConfig(agent_dir=AGENT, budget=budget),
                                   {"set": "solved", "dataset": "other", "fold": "A", "attempt": 1})


def test_records_for_all_fixtures(all_trace_paths):
    for p in all_trace_paths:
        tr, recs = build(p)
        assert len(recs) == 1, p.name                       # every request extends the previous one
        rec = recs[0]
        jsonschema.validate(rec, SCHEMA)
        assert len(rec["loss"]) == len(tr.turns)
        assert all(rec["messages"][e["message_index"]]["role"] == "assistant" for e in rec["loss"])
        assert rec["messages"][0]["role"] == "system"
        assert rec["messages"][0]["content"].startswith("You are an expert software engineer")
        assert "- Time allowance: 5 minutes" in rec["messages"][1]["content"]
        assert [t["function"]["name"] for t in rec["tools"]] == [
            "run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"]
        assert rec["meta"]["request_params"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_tool_outputs_reach_the_model_unchanged(astropy_path):
    tr, (rec,) = build(astropy_path)
    tool_msgs = [m["content"] for m in rec["messages"] if m["role"] == "tool"]
    assert tool_msgs == [c.result for c in tr.tool_calls()]


def test_thinking_and_parallel_calls(gunicorn_path):
    tr, (rec,) = build(gunicorn_path)
    assistants = [m for m in rec["messages"] if m["role"] == "assistant"]
    assert assistants[0]["reasoning_content"] == tr.turns[0].thinking_text
    parallel = assistants[3]
    assert [tc["function"]["name"] for tc in parallel["tool_calls"]] == ["read_file", "read_file"]
    assert rec["messages"][-1]["content"] == tr.final_text


def test_repeated_call_gets_zero_weight(exceptiongroup_path, arrow_path):
    _, (eg,) = build(exceptiongroup_path)
    zero = [e for e in eg["loss"] if e["weight"] == 0]
    assert len(zero) == 1 and zero[0]["reason"] == "repeated_call"
    _, (ar,) = build(arrow_path)
    assert all(e["weight"] == 1 for e in ar["loss"])           # second read shows new lines


def test_budget_warning_in_replayed_outputs(gunicorn_path):
    _, (rec,) = build(gunicorn_path, Budget(5, 22, 300))
    tool_msgs = [json.loads(m["content"]) for m in rec["messages"] if m["role"] == "tool"]
    warned = [m for m in tool_msgs if "budget_warning" in m]
    assert len(warned) == 1 and "(12/22 used)" in warned[0]["budget_warning"]


def test_over_budget_trajectory_is_rejected(exceptiongroup_path):
    with pytest.raises(BudgetExceeded):
        script_from_trace(load_trace(exceptiongroup_path), Budget(5, 20, 300))


def test_nudge_and_final_text():
    steps = [
        ScriptTurn(thinking="look", calls=[ScriptCall("run_command", {"command": "ls"},
                                                      '{"status": "ok", "stdout": "a\\n", "stderr": "", "exit_code": 0}')]),
        ScriptTurn(text="I think I am done."),                 # text-only: ADK ends the run
        ScriptNudge("Please continue your work using the available tools, or call submit_patch when "
                    "you have completed and verified your changes."),
        ScriptTurn(calls=[ScriptCall("submit_patch", {}, '{"status": "ok", "patch_size": 10, "files_changed": 1}')]),
        ScriptTurn(text="Fixed."),
    ]
    res = replay(steps, AGENT, "Task prompt", "problem", ALIAS)
    assert len(res.agent_exchanges) == 4
    msgs = res.agent_exchanges[-1].request["messages"]
    assert msgs[-1]["role"] == "tool"
    assert any(m["role"] == "user" and "Please continue" in str(m.get("content")) for m in msgs)


def test_mismatch_is_loud():
    steps = [ScriptTurn(calls=[ScriptCall("run_command", {"command": "ls"}, '{"status": "ok"}')])]
    steps2 = steps + [ScriptTurn(text="done"), ScriptTurn(text="extra turn never requested")]
    with pytest.raises(ReplayError):
        replay(steps2, AGENT, "Task prompt", "problem", ALIAS)


def test_history_echo_shape():
    msg = {"role": "assistant", "content": "x", "reasoning": "r", "reasoning_content": "r",
           "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    assert history_echo(msg) == {"role": "assistant", "content": "x",
                                 "tool_calls": [{"type": "function", "id": "c1",
                                                 "function": {"name": "f", "arguments": "{}"}}],
                                 "reasoning_content": "r", "reasoning": "r"}


def test_compaction_windows_train_each_turn_once(exceptiongroup_path):
    from locfix.harness.replay import static_summarizer

    tr = load_trace(exceptiongroup_path)
    bc = BuildConfig(agent_dir=AGENT, compaction=dict(compaction_interval=15, overlap_size=2,
                                                      token_threshold=6000, event_retention_size=5))
    recs = build_agent_records(tr, bc, {"set": "solved", "dataset": "other", "fold": "A", "attempt": 1},
                               summarizer=static_summarizer("SUMMARY-X"))
    assert len(recs) > 1
    turns = [t for r in recs for t in r["meta"]["turns"]]
    assert sorted(turns) == list(range(len(tr.turns)))
    later = recs[1]["messages"]
    assert any(m["role"] == "user" and "SUMMARY-X" in json.dumps(m.get("content")) for m in later)
    for r in recs:
        jsonschema.validate(r, SCHEMA)
