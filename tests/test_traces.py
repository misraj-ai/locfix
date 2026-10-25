import copy
import json

import pytest

from locfix.data.traces import (
    Nudge,
    TraceError,
    build_trace,
    is_workspace_write_command,
    load_trace,
    parse_trace_name,
)


def test_turns_match_model_calls(all_trace_paths):
    # every model call carries `usage` on its first entry: one turn per usage-carrying entry
    for p in all_trace_paths:
        tr = load_trace(p)
        assert len(tr.turns) == sum("usage" in e for e in tr.entries), p.name
        for c in tr.tool_calls():
            assert c.result is not None


def test_counts_on_fixtures(exceptiongroup_path, arrow_path, astropy_path, gunicorn_path):
    eg, ar, ap, gu = (load_trace(p) for p in (exceptiongroup_path, arrow_path, astropy_path, gunicorn_path))
    assert (eg.n_counted_calls, ar.n_counted_calls, ap.n_counted_calls, gu.n_counted_calls) == (28, 13, 3, 12)
    assert [t.index for t in gu.turns if t.is_parallel] == [3]           # two read_file calls
    assert [c.tool for c in gu.turns[3].tool_calls] == ["read_file", "read_file"]
    assert gu.turns[3].tool_calls[0].path.endswith("tests/test_reload.py")
    assert ap.final_text.startswith("Fixed the root cause")
    assert eg.turns[[t.entry_indices[0] for t in eg.turns].index(58)].thinking_cleaned


def test_first_edit(arrow_path, gunicorn_path, astropy_path):
    ar, gu, ap = load_trace(arrow_path), load_trace(gunicorn_path), load_trace(astropy_path)
    assert ar.first_edit_call().entry_index == 8 and ar.first_edit_call().tool == "edit_file"
    assert [c.tool for c in ar.calls_before_first_edit()] == ["read_file", "read_file"]
    assert gu.first_edit_call().entry_index == 16
    assert {c.path.split("/workspace/")[-1] for c in gu.calls_before_first_edit() if c.path} == {
        "gunicorn/reloader.py", "tests/test_reload.py", "gunicorn/workers/base.py"}
    assert len(ap.calls_before_first_edit()) == 1


def test_tool_timing_nonnegative(all_trace_paths):
    for p in all_trace_paths:
        tr = load_trace(p)
        assert len(tr.turn_tool_seconds()) == len(tr.turns)
        assert all(x >= 0 for x in tr.turn_tool_seconds() + tr.turn_model_seconds())


def test_name_parsing():
    assert parse_trace_name("x/astropy__astropy-13803__a1.json") == ("astropy__astropy-13803", 1)
    assert parse_trace_name("1792827328459_astropy__astropy-13803__a2.json") == ("astropy__astropy-13803", 2)


def test_errors_are_loud(astropy_path):
    entries = json.loads(astropy_path.read_text())["entries"]
    bad = copy.deepcopy(entries)
    bad[3]["type"] = "mystery"
    with pytest.raises(TraceError, match="unknown entry type"):
        build_trace(bad)
    bad = [e for e in copy.deepcopy(entries) if e["type"] != "tool_response"]
    with pytest.raises(TraceError):
        build_trace(bad)


def test_nudge_events():
    entries = [
        {"type": "system_instruction", "content": "s"},
        {"type": "task_prompt", "content": "p"},
        {"type": "text", "content": "thinking out loud", "timestamp": 1, "elapsed_s": 1, "usage": {}},
        {"type": "continuation_nudge", "content": "Please continue"},
        {"type": "tool_call", "tool": "submit_patch", "args": {}, "timestamp": 2, "elapsed_s": 2, "usage": {}},
        {"type": "tool_response", "tool": "submit_patch", "result": "{}", "elapsed_s": 2.1},
    ]
    tr = build_trace(entries)
    assert [type(e).__name__ for e in tr.events] == ["Turn", "Nudge", "Turn"]
    assert isinstance(tr.events[1], Nudge)


@pytest.mark.parametrize("cmd,expected", [
    ("cat > /tmp/repro.py <<'EOF'\nprint(1)\nEOF\npython /tmp/repro.py", False),
    ("rm /tmp/repro.py", False),
    ("grep -rn foo src/ > /tmp/out.txt", False),
    ("python -m pytest tests -q 2>&1 | tail -3", False),
    ("cd /workspace && git stash && pytest -q; git stash pop", True),
    ("sed -i 's/a/b/' src/pkg/mod.py", True),
    ("sed -n '150,400p' /workspace/src/x.py", False),
    ("echo x >> src/pkg/mod.py", True),
    ("cp /tmp/fixed.py src/pkg/mod.py", True),
    ("rm src/pkg/old.py", True),
    ("git apply /tmp/fix.patch", True),
    ("ls /usr/lib/python3*/site-packages 2>/dev/null; pip show exceptiongroup 2>/dev/null | head -3", False),
])
def test_workspace_write_heuristic(cmd, expected):
    assert is_workspace_write_command(cmd) is expected
