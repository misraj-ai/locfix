"""Record-builder CLI: --shard + data.merge-records, the length limit, and the shared task list."""
import json
from pathlib import Path

import pytest

from locfix.cli import main

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "work" / "model_meta" / "e4b"
pytestmark = pytest.mark.skipif(not (TOK / "tokenizer.json").exists(), reason="tokenizer not available")


@pytest.fixture
def data(tmp_path, all_trace_paths):
    from test_loc_fix_records import make_task

    traces = tmp_path / "traces"
    traces.mkdir()
    with open(tmp_path / "solved.jsonl", "w") as f:
        for p in all_trace_paths:
            (traces / p.name).write_bytes(p.read_bytes())
            t = make_task(p)
            f.write(json.dumps({"instance_id": t.instance_id, "problem_statement": t.problem_statement,
                                "fold": "A", "verifier": "fixture", "teacher": {"patch": t.patch}}) + "\n")
    return tmp_path


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# every fixture trace fits in 100 calls; the exclusion list would drop none of them anyway
BIG = ["--set", "budget.max_tool_calls=100", "--set", "budget_locfix.max_tool_calls=100", "--set", "data.exclude_repos=[]"]


def agent(data, out, *extra):
    return main(["data.agent-records", str(data / "traces"), "--solved", str(data / "solved.jsonl"),
                 "--out", str(out), "--tokenizer", str(TOK), *BIG, *extra])


def test_shards_merge_to_the_full_build(data, tmp_path):
    agent(data, tmp_path / "full" / "s1_agent.jsonl")
    agent(data, tmp_path / "sh0" / "s1_agent.jsonl", "--shard", "0/2")
    agent(data, tmp_path / "sh1" / "s1_agent.jsonl", "--shard", "1/2")
    main(["data.merge-records", str(tmp_path / "sh0"), str(tmp_path / "sh1"), "--out", str(tmp_path / "merged")])
    full = lines(tmp_path / "full" / "s1_agent.jsonl")
    merged = lines(tmp_path / "merged" / "s1_agent.jsonl")
    assert sorted(r["id"] for r in full) == sorted(r["id"] for r in merged) and len(full) >= 4
    assert {r["id"]: r for r in full} == {r["id"]: r for r in merged}           # identical records
    assert all(r["meta"]["n_tokens"] > 1000 for r in full)
    full_tasks = (tmp_path / "full" / "s1_agent.tasks.txt").read_text().split()
    assert sorted(full_tasks) == (tmp_path / "merged" / "s1_agent.tasks.txt").read_text().split()
    m = json.loads((tmp_path / "merged" / "manifest.json").read_text())
    assert m["counts"]["traces"] == 4 and m["counts"]["traces_kept"] == len(full_tasks)
    with pytest.raises(SystemExit):           # the same shard twice: duplicate ids
        main(["data.merge-records", str(tmp_path / "sh0"), str(tmp_path / "sh0"), "--out", str(tmp_path / "bad")])


def test_length_limit_drops_whole_trajectories_and_sets_the_task_list(data, tmp_path):
    agent(data, tmp_path / "full" / "s1_agent.jsonl")
    full = lines(tmp_path / "full" / "s1_agent.jsonl")
    longest: dict[str, int] = {}
    for r in full:
        longest[r["meta"]["trace"]] = max(longest.get(r["meta"]["trace"], 0), r["meta"]["n_tokens"])
    limit = sorted(longest.values())[1]                       # keeps the two shortest trajectories
    out = tmp_path / "cut" / "s1_agent.jsonl"
    agent(data, out, "--set", f"data.max_trace_tokens={limit}")
    kept = (tmp_path / "cut" / "s1_agent.tasks.txt").read_text().split()
    assert sorted(kept) == sorted(t for t, n in longest.items() if n <= limit) and len(kept) == 2
    assert {r["meta"]["trace"] for r in lines(out)} == set(kept)
    counts = json.loads((tmp_path / "cut" / "run_meta.json").read_text())["counts"]
    assert counts["skip_too_long"] == 2

    # two-agent records restricted to the same trajectories
    lf = tmp_path / "locfix"
    main(["data.locfix-records", str(data / "traces"), "--solved", str(data / "solved.jsonl"),
          "--code", str(ROOT / "tests" / "fixtures" / "code"), "--out", str(lf), "--tokenizer", str(TOK),
          "--tasks", str(tmp_path / "cut" / "s1_agent.tasks.txt"), *BIG])
    recs = [r for f in lf.glob("*.jsonl") for r in lines(f)]
    ids = {t.rsplit("__a", 1)[0] for t in kept}               # <instance_id>__a<N>.json
    assert recs and {r["instance_id"] for r in recs} == ids
    assert all("n_tokens" in r["meta"] for r in recs)
    man = json.loads((lf / "manifest.json").read_text())
    assert man["skip:not_in_task_list"] == 2


def test_limit_needs_a_tokenizer(data, tmp_path):
    with pytest.raises(SystemExit, match="--tokenizer"):
        main(["data.agent-records", str(data / "traces"), "--solved", str(data / "solved.jsonl"),
              "--out", str(tmp_path / "x.jsonl"), *BIG])


def test_split_by_task(tmp_path):
    from locfix.data.build_common import is_val

    ids = [f"repo__task-{i}" for i in range(2000)]
    val = [i for i in ids if is_val(i, 0.03, 1234)]
    assert 30 < len(val) < 90 and val == [i for i in ids if is_val(i, 0.03, 1234)]
    src = tmp_path / "src"
    (src / "locfix").mkdir(parents=True)
    rows = [{"id": f"r{n}", "instance_id": ids[n % 200]} for n in range(600)]
    (src / "s1_agent.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (src / "locfix" / "s2_fix_gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:300]))
    main(["data.split", str(src / "s1_agent.jsonl"), str(src / "locfix"), "--out", str(tmp_path / "rec"),
          "--fraction", "0.2"])
    for rel in ("s1_agent.jsonl", "locfix/s2_fix_gold.jsonl"):
        tr, va = lines(tmp_path / "rec" / rel), lines(tmp_path / "rec" / "val" / rel)
        assert va and {r["instance_id"] for r in tr}.isdisjoint({r["instance_id"] for r in va})
        assert sorted(r["id"] for r in tr + va) == sorted(r["id"] for r in (rows if rel.startswith("s1") else rows[:300]))
        assert all(is_val(r["instance_id"], 0.2, 1234) for r in va)


def test_code_paths_and_check_code(data, tmp_path):
    main(["data.code-paths", str(data / "traces"), "--solved", str(data / "solved.jsonl"),
          "--out", str(tmp_path / "paths.jsonl")])
    rows = {r["instance_id"]: r for r in lines(tmp_path / "paths.jsonl")}
    assert len(rows) == 4
    code_dir = ROOT / "tests" / "fixtures" / "code"
    for iid, r in rows.items():
        files = json.loads((code_dir / r["code_file"]).read_text())
        # every file the fixture code holds for the patch is requested
        from locfix.data.code_files import patch_paths
        solved = {x["instance_id"]: x for x in lines(data / "solved.jsonl")}
        assert set(patch_paths(solved[iid]["teacher"]["patch"])) <= set(r["paths"]) and files
    main(["data.check-code", str(data / "traces"), "--solved", str(data / "solved.jsonl"),
          "--code", str(code_dir), "--out", str(tmp_path / "check.json")])
    rep = json.loads((tmp_path / "check.json").read_text())
    assert rep["counts"] == {"ok": 3, "unverified": 1}            # astropy: the teacher printed no file text
