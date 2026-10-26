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
