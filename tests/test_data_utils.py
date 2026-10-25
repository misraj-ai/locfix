import json

import pytest

from locfix.data.edits import (
    PatchError,
    apply_patch,
    make_unified_diff,
    parse_patch,
    patch_to_edit_calls,
)
from locfix.data.locations import SymbolIndex, edit_locations, view_locations, view_ranges
from locfix.data.traces import load_trace
from locfix.harness.local_sandbox import LocalWorkspace


def arrow_files(arrow_path):
    """arrow/factory.py at the base commit, reconstructed from the teacher's two reads."""
    tr = load_trace(arrow_path)
    calls = list(tr.tool_calls())
    r1, r2 = calls[0].result_json(), calls[1].result_json()
    text = "\n".join(r1["content"].split("\n") + r2["content"].split("\n")[1:]) + "\n"
    return tr, {"arrow/factory.py": text}


def arrow_teacher_patch(arrow_path):
    tr, before = arrow_files(arrow_path)
    with LocalWorkspace(before) as ws:
        for c in tr.tool_calls():
            if c.tool == "edit_file":
                ws.call(c.tool, c.args)
        after = ws.read_text("arrow/factory.py")
    return tr, before, after, make_unified_diff("arrow/factory.py", before["arrow/factory.py"], after)


def test_local_harness_tools_reproduce_recorded_outputs(arrow_path):
    tr, before = arrow_files(arrow_path)
    with LocalWorkspace(before) as ws:
        for c in tr.tool_calls():
            if c.tool in ("read_file", "edit_file"):
                assert ws.call(c.tool, c.args) == c.result, c.entry_index


def test_read_file_truncation_rules():
    text = "".join(f"line {i}\n" for i in range(1, 401))
    with LocalWorkspace({"a.py": text}) as ws:
        out = json.loads(ws.read_file("a.py"))
        assert (out["start_line"], out["end_line"], out["total_lines"], out["is_truncated"]) == (1, 150, 400, True)
        out = json.loads(ws.read_file("/workspace/a.py", 10, 598))      # end_line beyond the cap is ignored
        assert (out["start_line"], out["end_line"], out["is_truncated"]) == (10, 159, True)
        out = json.loads(ws.read_file("missing.py"))
        assert out["status"] == "error" and out["error_type"] == "FileReadError"


def test_patch_roundtrip_and_edit_calls(arrow_path):
    tr, before, after, patch = arrow_teacher_patch(arrow_path)
    assert apply_patch(before, patch)["arrow/factory.py"] == after
    calls = patch_to_edit_calls(before, patch, filepath_prefix="/workspace/")   # validated by the harness
    assert all(c.tool == "edit_file" for c in calls)
    assert all(c.args["filepath"] == "/workspace/arrow/factory.py" for c in calls)
    assert sum(len(c.added_lines) for c in calls) > 0


def test_new_file_and_multi_file_patch():
    before = {"pkg/a.py": "def f():\n    return 1\n\n\ndef g():\n    return 2\n"}
    after_a = "def f():\n    return 10\n\n\ndef g():\n    return 2\n"
    patch = make_unified_diff("pkg/a.py", before["pkg/a.py"], after_a)
    patch += ("diff --git a/pkg/new.py b/pkg/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/pkg/new.py\n"
              "@@ -0,0 +1,2 @@\n+X = 1\n+Y = 2\n")
    calls = patch_to_edit_calls(before, patch)
    assert [c.tool for c in calls] == ["edit_file", "write_file"]
    assert calls[0].added_lines == ["    return 10"]
    assert calls[1].args["content"] == "X = 1\nY = 2\n"


def test_non_unique_context_is_extended():
    before = {"m.py": "x = 1\ny = 0\nx = 1\ny = 0\nz = 9\n"}
    after = "x = 1\ny = 0\nx = 1\ny = 5\nz = 9\n"
    calls = patch_to_edit_calls(before, make_unified_diff("m.py", before["m.py"], after))
    assert len(calls) == 1 and before["m.py"].count(calls[0].args["old_string"]) == 1


def test_context_mismatch_is_loud():
    before = {"m.py": "a\nb\nc\n"}
    patch = make_unified_diff("m.py", "a\nB\nc\n", "a\nX\nc\n")
    with pytest.raises(PatchError, match="context mismatch"):
        apply_patch(before, patch)


def test_changed_lines_include_insertion_anchor():
    fp = parse_patch(make_unified_diff("m.py", "a\nb\nc\n", "a\nb\nNEW\nc\n"))[0]
    assert fp.changed_old_lines() == [2]


def test_edit_locations_arrow(arrow_path):
    tr, before, after, patch = arrow_teacher_patch(arrow_path)
    locs = edit_locations(before, patch)
    assert [(loc.symbol, loc.role) for loc in locs] == [("ArrowFactory.get", "edit")]


def test_view_locations_arrow(arrow_path):
    tr, before = arrow_files(arrow_path)
    names = [loc.symbol for loc in view_locations(tr, before)]
    assert "ArrowFactory.get" in names and "ArrowFactory.now" in names
    assert "ArrowFactory" not in names                       # class replaced by its methods


def test_view_ranges_gunicorn_and_truncated_sed(gunicorn_path, exceptiongroup_path):
    gu = [(v.path, v.start, v.end) for v in view_ranges(load_trace(gunicorn_path))]
    assert gu == [("gunicorn/reloader.py", 1, 131), ("tests/test_reload.py", 1, 68),
                  ("gunicorn/workers/base.py", 85, 140)]
    eg = [(v.start, v.end, v.via) for v in view_ranges(load_trace(exceptiongroup_path))]
    assert (150, 275, "sed") in eg                          # 'sed -n 150,400p' cut at 5,000 chars


def test_symbol_index_decorators_and_nesting():
    src = "import os\n\n@dec\ndef f():\n    pass\n\nclass C:\n    def m(self):\n        return 1\n"
    idx = SymbolIndex("x.py", src)
    assert [(s.name, s.start, s.end, s.depth) for s in idx.symbols] == [
        ("f", 3, 5, 0), ("C", 7, 9, 0), ("C.m", 8, 9, 1)]
    assert idx.innermost(9).name == "C.m"
    assert [loc.symbol for loc in idx.location_for_lines([1, 9], "edit")] == ["<module>", "C.m"]
