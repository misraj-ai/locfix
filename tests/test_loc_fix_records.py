import json
import random
from pathlib import Path

import pytest
import yaml

from locfix.config import load_config
from locfix.data.edits import patch_to_edit_calls
from locfix.data.loc_fix_records import (
    LocFixConfig,
    Task,
    TaskSkip,
    Variant,
    build_task_records,
    patch_from_edits,
    records_for_variant,
    references,
    solved_variants,
    task_locations,
)
from locfix.data.locations import Location
from locfix.data.render_obs import ObsRenderer, check_recorded_reads, merge_ranges, read_calls_for
from locfix.data.traces import load_trace
from locfix.data.zloc import parse_z, prune, union, z_to_json
from locfix.harness.prompts import parse_task_prompt

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).parent / "fixtures" / "code"


def make_task(path) -> Task:
    tr = load_trace(path)
    files = json.loads((CODE / f"{tr.instance_id}.json").read_text())
    return Task(tr, files, patch_from_edits(tr, files, strict=False),
                parse_task_prompt(tr.task_prompt).problem_statement,
                source={"set": "solved", "dataset": "fixture", "fold": "A", "attempt": 1})


@pytest.fixture(scope="module")
def em_cfg():
    return load_config()["em"]


# --------------------------------------------------------------------------- inputs
def test_recorded_reads_reproduce_from_code(all_trace_paths):
    """Spec 04 §7 acceptance: every teacher read_file before the first edit re-renders exactly."""
    total = 0
    for p in all_trace_paths:
        t = make_task(p)
        ok, bad = check_recorded_reads(t.trace, t.files)
        assert bad == [], (p.name, bad)
        total += ok
    assert total == 7


def test_teacher_patches_convert_to_exact_edit_calls(all_trace_paths):
    for p in all_trace_paths:
        t = make_task(p)
        calls = patch_to_edit_calls(t.files, t.patch, filepath_prefix="/workspace/")   # validated
        assert calls and all(c.tool == "edit_file" for c in calls)


def test_locations_acceptance(arrow_path, gunicorn_path):
    tl = task_locations(make_task(arrow_path))
    assert [x.symbol for x in tl.l_edit] == ["ArrowFactory.get"]
    tl = task_locations(make_task(gunicorn_path))
    assert [x.symbol for x in tl.l_edit] == ["Worker.init_process"]
    viewed = {x.path for x in tl.l_view}
    assert {"gunicorn/reloader.py", "tests/test_reload.py"} <= viewed
    assert len(tl.pre_turns) == 4 and tl.searches


# --------------------------------------------------------------------------- z
def test_z_roundtrip_and_prune():
    files = {"a.py": "def f():\n    return 1\n\n\ndef g():\n" + "    x = 1\n" * 300}
    locs = [Location("a.py", "g", 5, 305, "context"), Location("a.py", "f", 1, 2, "edit")]
    assert parse_z(z_to_json(locs)) == locs
    assert json.loads(z_to_json(locs))["locations"][0] == {"path": "a.py", "symbol": "g",
                                                           "lines": [5, 305], "role": "context"}
    kept = prune(locs, files, max_locations=8, budget_tokens=100)
    assert [x.symbol for x in kept] == ["f"]                     # edit kept, big context dropped
    assert [x.role for x in union([Location("a.py", "f", 1, 2)], locs[1:])] == ["edit"]


def test_read_calls_merge_split_and_continue():
    assert merge_ranges([(10, 20), (22, 30), (40, 45)]) == [(10, 30), (40, 45)]
    long_lines = "".join(f"x{i} = '{'y' * 200}'\n" for i in range(1, 401))
    files = {"m.py": long_lines}
    calls = read_calls_for([Location("m.py", "<module>", 1, 400)], files)
    # reads sized under the 10,000-character cap: every output is complete and they tile 1..400
    assert calls[0][0].start_line == 1 and len(calls) == 9
    for (c, out), (nxt, _) in zip(calls, calls[1:], strict=False):
        assert not json.loads(out)["is_truncated"] and c.end_line + 1 == nxt.start_line
    assert calls[-1][0].end_line == 400
    with ObsRenderer(files) as r:
        assert read_calls_for([Location("new.py", "<file>", 1, 1, "edit")], files, r) == []


def test_references():
    refs = references("The bug is in `ArrowFactory.get()`; see arrow/factory.py and `tzinfo`. `None`")
    assert refs == {"ArrowFactory.get", "arrow/factory.py", "tzinfo"}


# --------------------------------------------------------------------------- records
def test_variants_gunicorn(gunicorn_path, em_cfg):
    t = make_task(gunicorn_path)
    vs = solved_variants(task_locations(t), t.files, em_cfg, random.Random(0))
    names = {v.name: v for v in vs}
    assert names["gold"].fixer_stage == "s2_fix_gold" and not names["gold"].requests
    assert [x.symbol for x in names["edit_only"].z] == ["Worker.init_process"]
    assert all(x.role == "edit" for x in names["view_only"].z if x.symbol == "Worker.init_process")


@pytest.fixture(scope="module")
def gunicorn_records(em_cfg):
    from conftest import trace_path

    t = make_task(trace_path("benoitc__gunicorn"))
    tl = task_locations(t)
    z = list(tl.l_edit)
    recs = records_for_variant(t, tl, Variant("edit_only", z, True, "s2_fix_noisy", "s2_loc_edit"),
                               LocFixConfig(em=em_cfg))
    return t, tl, recs


def test_fixer_record_structure(gunicorn_records):
    t, tl, recs = gunicorn_records
    (fx,) = [r for r in recs if r["role"] == "fixer"]
    msgs = fx["messages"]
    # one user message: with skip_summarization false the fixer continues after the localizer
    # call (true would end the run and the harness would inject a nudge)
    assert [m["role"] for m in msgs].count("user") == 1
    call = msgs[2]["tool_calls"][0]["function"]
    assert call["name"] == "localizer"
    assert json.loads(call["arguments"]) == {"request": "Find the code locations relevant to this issue."}
    wrapped = json.loads(msgs[3]["content"])
    assert parse_z(wrapped["result"]) == tl.l_edit                   # AgentTool wraps the text
    assert msgs[4]["tool_calls"][0]["function"]["name"] == "read_file"
    assert {q["via"] for q in fx["meta"]["requests"]} == {"read", "grep"}
    names = [tc["function"]["name"] for m in msgs if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
    assert names[-1] == "submit_patch" and msgs[-1]["content"]          # teacher's ending kept
    assert fx["meta"]["dangling_refs"] == []
    assert "localizer" in [x["function"]["name"] for x in fx["tools"]]


def test_localizer_record_structure(gunicorn_records):
    t, tl, recs = gunicorn_records
    (lc,) = [r for r in recs if r["role"] == "localizer"]
    msgs = lc["messages"]
    assert msgs[1] == {"role": "user", "content": "Find the code locations relevant to this issue."}
    sys = msgs[0]["content"] if isinstance(msgs[0]["content"], str) else msgs[0]["content"][0]["text"]
    assert t.problem_statement.strip()[:200] in sys and '{"locations": [{"path"' in sys
    assert {x["function"]["name"] for x in lc["tools"]} == {"run_command", "read_file"}
    assert parse_z(msgs[-1]["content"]) == tl.l_edit
    assert len(lc["loss"]) == len(tl.pre_turns) + 1
    assert "requests" not in lc["meta"]


def test_shared_budget_counts_localizer_calls(gunicorn_records):
    t, tl, recs = gunicorn_records
    (fx,) = [r for r in recs if r["role"] == "fixer"]
    loc_calls = sum(len(x.tool_calls) for x in tl.pre_turns)
    fixer_calls = sum(1 for m in fx["messages"] if m["role"] == "assistant"
                      for tc in m.get("tool_calls", [])
                      if tc["function"]["name"] in ("run_command", "read_file", "edit_file", "write_file"))
    assert fx["meta"]["counted_calls"] == loc_calls + fixer_calls


def test_impossible_input_is_dropped(gunicorn_path, em_cfg):
    t = make_task(gunicorn_path)
    tl = task_locations(t)
    ctx_only = [x for x in tl.l_view if x.path == "tests/test_reload.py"]
    with pytest.raises(TaskSkip, match="impossible_input"):
        records_for_variant(t, tl, Variant("bad", ctx_only, False, "s2_fix_noisy", None),
                            LocFixConfig(em=em_cfg))


def test_all_fixtures_build_and_render(all_trace_paths, em_cfg):
    from locfix.render.chat import load_tokenizer, render_record

    counts: dict[str, int] = {}
    recs = []
    for p in all_trace_paths:
        recs += build_task_records(make_task(p), LocFixConfig(em=em_cfg), counts)
    assert not [k for k in counts if k.startswith("drop:")], counts
    stages = {r["stage"] for r in recs}
    assert stages == {"s2_fix_gold", "s2_fix_noisy", "s2_loc_edit", "s2_loc_view", "s3_loc_r0"}
    tok_dir = ROOT / "work" / "model_meta" / "e4b"
    if not (tok_dir / "tokenizer.json").exists():
        pytest.skip("tokenizer not available")
    tok = load_tokenizer(tok_dir)
    for r in recs:
        out = render_record(r, tok)
        assert len(out.input_ids) < 32768


def test_locfix_agent_config(em_cfg):
    agent = yaml.safe_load((ROOT / "agents/locfix/agent.yaml").read_text().replace("!include", ""))
    (entry,) = [x for x in agent["tools"] if isinstance(x, dict)]
    assert entry["agent_tool"]["skip_summarization"] is False
    ev = yaml.safe_load((ROOT / "agents/locfix/eval_config.yaml").read_text())["evaluation"]
    b = load_config()["budget_locfix"]
    assert (ev["max_time_minutes"], ev["max_tool_calls"], ev["max_turns"]) == (
        b["max_time_minutes"], b["max_tool_calls"], b["max_turns"])
