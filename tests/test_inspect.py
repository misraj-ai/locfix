import json
from pathlib import Path

import pytest

from locfix.render import inspect as I

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "tests" / "fixtures" / "records"


def test_tables_without_tokenizer(gunicorn_path):
    trace = json.loads(gunicorn_path.read_text())
    rows = I.trace_rows(trace["entries"])
    assert rows[3]["type"] == "tool_call" and rows[3]["content"].startswith("run_command(")
    rec = json.loads((RECORDS / "s1_fixtures.jsonl").read_text().splitlines()[0])
    mrows = I.message_rows(rec)
    assert len(mrows) == len(rec["messages"])
    assert sum(r["trained"].startswith("yes") for r in mrows) == I.record_summary(rec)["trained turns"]


def test_rendered_views():
    tok_dir = ROOT / "work" / "model_meta" / "e4b"
    if not (tok_dir / "tokenizer.json").exists():
        pytest.skip("tokenizer not available")
    from locfix.render.chat import load_tokenizer, render_record

    tok = load_tokenizer(tok_dir)
    rec = json.loads((RECORDS / "locfix_fixtures.jsonl").read_text().splitlines()[0])
    r = render_record(rec, tok)
    pieces = I._segments(r)
    assert "".join(t for t, _ in pieces) == r.text                    # the view covers the text exactly
    assert sum(1 for _, w in pieces if w is not None) == len(r.target_spans)
    assert "<span" in I.rendered_html(r) and "⟦" in I.rendered_ascii(r)
    end = I.token_rows(r, tok, span=0, at="end", before=2, after=2)
    assert end[1]["piece"] == repr("<|tool_response>") and end[1]["in loss"] == "✓"
    assert end[2]["in loss"] == ""


def test_check_records_cli(capsys):
    tok_dir = ROOT / "work" / "model_meta" / "e4b"
    if not (tok_dir / "tokenizer.json").exists():
        pytest.skip("tokenizer not available")
    from locfix.cli import main

    assert main(["data.check-records", str(RECORDS), "--tokenizer", str(tok_dir), "--sample", "2"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out[out.index("{"):])
    assert report["s1_fixtures.jsonl"]["errors"] == 0 and report["locfix_fixtures.jsonl"]["rendered"] == 2


def test_data_stats_cli(tmp_path, capsys):
    tok_dir = ROOT / "work" / "model_meta" / "e4b"
    if not (tok_dir / "tokenizer.json").exists():
        pytest.skip("tokenizer not available")
    from locfix.cli import main

    assert main(["data.stats", "--traces", str(ROOT / "tests/fixtures/traces"), "--records", str(RECORDS),
                 "--tokenizer", str(tok_dir), "--out", str(tmp_path), "--world-size", "2",
                 "--global-batch", "4", "--tokens-per-s", "2000"]) == 0
    js = json.loads((tmp_path / "stats.json").read_text())
    sweep = {r["max_tool_calls"]: r["traces kept"] for r in js["traces"]["budget_sweep"]}
    assert sweep[24] == 3 and sweep[30] == 4                  # exceptiongroup needs 28 calls
    dist = js["traces"]["distributions"]
    assert 8000 < dist["approx_tokens"]["min"] < 9000          # astropy, rendered 8,381
    stages = {r["stage"]: r for r in js["records"]["stages"]}
    assert stages["s1_agent"]["records"] == 4 and stages["s2_fix_noisy"]["with requests"] == 4
    sim = js["records"]["step_simulation"]
    rnd = next(r for r in sim if r["records"] == "all" and r["mode"] == "random" and r["max_seq_len"] == "32K")
    grp = next(r for r in sim if r["records"] == "all" and r["mode"] == "grouped" and r["max_seq_len"] == "32K")
    assert grp["gpu_efficiency %"] >= rnd["gpu_efficiency %"] and "epoch hours" in rnd
    rows = [json.loads(x) for x in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert all(0.95 < sum(r["categories"].values()) / r["tokens"] < 1.05 for r in rows)   # breakdown ±2%


def test_length_grouping_plan():
    import numpy as np

    from locfix.train.data import Example, Source, make_plan

    lens = [100, 9000, 120, 8800, 150, 9500, 90, 9100]
    exs = [Example(np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.ones(n, dtype=np.float32),
                   f"r{i}", f"t{i}", "s") for i, n in enumerate(lens)]
    plan = make_plan(exs, [Source("x")], 1, rows_per_step=4, max_tokens=10000, packing=False, seed=0,
                     length_grouping=True)
    steps = [plan.rows[i:i + 4] for i in range(0, 8, 4)]
    for st in steps:                                             # each step all short or all long
        sizes = [lens[r[0]] for r in st]
        assert max(sizes) < 200 or min(sizes) > 8000
    assert sorted(i for r in plan.rows for i in r) == list(range(8))
