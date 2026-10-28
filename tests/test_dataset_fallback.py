import json
import shutil
from pathlib import Path

import pytest

from locfix.data.code_from_trace import seen_code
from locfix.data.dataset import SolvedData
from locfix.data.loc_fix_records import TaskSkip, task_locations
from locfix.data.locations import SymbolIndex
from locfix.data.render_obs import ObsRenderer, UnseenCode, read_calls_for
from locfix.data.locations import Location
from locfix.data.traces import load_trace

FIX = Path(__file__).parent / "fixtures"


def test_seen_code_matches_true_code(all_trace_paths):
    for p in all_trace_paths:
        tr = load_trace(p)
        true = json.loads((FIX / "code" / f"{tr.instance_id}.json").read_text())
        s = seen_code(tr)
        files = s.files()
        for path, known in s.known().items():
            assert len(files[path].splitlines()) == len(true[path].splitlines())
            t, f = true[path].splitlines(), files[path].splitlines()
            assert all(t[n - 1] == f[n - 1] for n in known), (p.name, path)


def test_partial_file_symbols(gunicorn_path):
    s = seen_code(load_trace(gunicorn_path))
    path = "gunicorn/workers/base.py"
    assert s.coverage()[path] < 0.3
    idx = SymbolIndex(path, s.files()[path], s.known()[path])
    names = [(x.name, x.start, x.end) for x in idx.symbols]
    assert ("init_process", 86, 114) in names                 # class header unseen: no "Worker."
    with ObsRenderer(s.files(), s.known()) as r:
        read_calls_for([Location(path, "init_process", 86, 114)], s.files(), r)
        with pytest.raises(UnseenCode):
            read_calls_for([Location(path, "<module>", 1, 20)], s.files(), r)


@pytest.fixture
def data_dir(tmp_path, all_trace_paths):
    from test_loc_fix_records import make_task

    (tmp_path / "cleaned" / "traces").mkdir(parents=True)
    with open(tmp_path / "cleaned" / "solved.jsonl", "w") as f:
        for p in all_trace_paths:
            shutil.copy(p, tmp_path / "cleaned" / "traces" / p.name)
            t = make_task(p)
            f.write(json.dumps({"instance_id": t.instance_id, "problem_statement": t.problem_statement,
                                "fold": "B", "verifier": "swe_smith", "teacher": {"patch": t.patch}}) + "\n")
    return tmp_path


def test_solved_data_code_sources(data_dir):
    data = SolvedData(data_dir)
    assert len(data.trace_paths()) == 4
    astropy = next(p for p in data.trace_paths() if p.name.startswith("astropy"))
    task, info = data.task(astropy)
    assert info.source == "trace"
    with pytest.raises(TaskSkip, match="missing_code"):     # only grep|sed output: no file text
        task_locations(task)
    (data_dir / "experiment" / "code").mkdir(parents=True)
    shutil.copy(FIX / "code" / "astropy__astropy-13803.json", data_dir / "experiment" / "code")
    task, info = SolvedData(data_dir).task(astropy)
    assert info.source == "code_json" and task.known_lines is None
    assert [x.symbol for x in task_locations(task).l_edit] == ["Latitude._validate_angles"]
    assert task.source == {"set": "solved", "dataset": "swe_smith", "fold": "B", "attempt": 1}


def test_code_json_is_checked_against_the_trace(data_dir, all_trace_paths):
    """A code file that does not reproduce the teacher's reads is not used; a code file missing a
    file the teacher read is completed from the trace."""
    from locfix.data.code_files import check_code, needed_paths
    from locfix.data.edits import parse_patch
    from locfix.data.locations import normalize_path
    from locfix.data.loc_fix_records import LocFixConfig, build_task_records
    from test_loc_fix_records import make_task

    code = data_dir / "cleaned" / "code"
    code.mkdir(parents=True)
    for p in all_trace_paths:
        t = make_task(p)
        assert check_code(t.trace, t.files, t.patch)["status"] in ("ok", "unverified"), p.name
        assert set(t.files) & set(needed_paths(t.trace, t.patch)) >= {fp for fp in t.files if fp in t.patch}
    gun = next(p for p in all_trace_paths if p.name.startswith("benoitc"))
    t = make_task(gun)
    edited = {fp.old_path for fp in parse_patch(t.patch) if fp.old_path}
    read = {normalize_path(c.args["filepath"]) for c in t.trace.calls_before_first_edit() if c.tool == "read_file"}
    only_read = sorted(read - edited)
    assert only_read and edited & read, "fixture should read an edited file and a file it does not edit"
    target = code / f"{t.instance_id}.json"

    # 1) a read-only file missing from the code file -> completed from the trace
    target.write_text(json.dumps({k: v for k, v in t.files.items() if k != only_read[0]}))
    task, info = SolvedData(data_dir).task(gun)
    assert info.source == "code_json+trace" and only_read[0] in task.files
    assert task.known_lines[sorted(edited)[0]] == set(range(1, len(t.files[sorted(edited)[0]].splitlines()) + 1))
    assert build_task_records(task, LocFixConfig(em={"localizer_max_calls": 15, "max_locations": 8,
                                                      "ctx_budget_tokens": 6000, "no_request_fraction": 0.4}))

    # 2) an edited file whose text differs from what the teacher read -> the code file is not used
    path = sorted(edited & read)[0]
    target.write_text(json.dumps({**t.files, path: "# inserted line shifts everything\n" + t.files[path]}))
    task, info = SolvedData(data_dir).task(gun)
    assert info.source == "trace:code_json_differs" and task.known_lines is not None
