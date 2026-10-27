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
