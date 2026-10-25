"""Rendering records to tokens with the real Gemma 4 tokenizer and chat template.

Needs the tokenizer files (tokenizer.json, tokenizer_config.json, chat_template.jinja) of the
31B `-w4a16-ct` checkpoint in $LOCFIX_TOKENIZER_DIR (default: work/model_meta/31b); skipped if absent.
"""
import os
from pathlib import Path

import pytest

from locfix.data.agent_records import BuildConfig, build_agent_records, records_from_replay
from locfix.data.traces import load_trace
from locfix.harness.replay import ScriptCall, ScriptNudge, ScriptTurn, replay
from locfix.render.chat import prompt_token_counter, render_record, render_text

ROOT = Path(__file__).resolve().parents[1]
TOK_DIR = Path(os.environ.get("LOCFIX_TOKENIZER_DIR", ROOT / "work" / "model_meta" / "31b"))
AGENT = str(ROOT / "agents" / "single")
SRC = {"set": "solved", "dataset": "other", "fold": "A", "attempt": 1}

pytestmark = pytest.mark.skipif(not (TOK_DIR / "tokenizer.json").exists(),
                                reason="Gemma 4 tokenizer files not available")


@pytest.fixture(scope="module")
def tok():
    from locfix.render.chat import load_tokenizer
    return load_tokenizer(TOK_DIR)


def test_fixture_records_render(tok, all_trace_paths):
    for p in all_trace_paths:
        tr = load_trace(p)
        (rec,) = build_agent_records(tr, BuildConfig(agent_dir=AGENT), SRC)
        out = render_record(rec, tok)
        trained = [e for e in rec["loss"] if e["weight"] > 0]
        assert len(out.target_spans) == len(trained), p.name
        for s, e, _ in out.target_spans:
            piece = out.text[s:e]
            assert piece.endswith(("<|tool_response>", "<turn|>")), piece[-40:]
        # labels decode to exactly the target texts
        lab = [t for t, w in zip(out.input_ids, out.weights, strict=True) if w > 0]
        assert tok.decode(lab, skip_special_tokens=False) == "".join(out.text[s:e] for s, e, _ in out.target_spans)
        assert len(out.input_ids) < 32768
        assert out.text.startswith("<bos><|turn>system\n<|think|>\n")
        assert 'Your internal name is "swe_agent".' in out.text      # added by ADK


def test_final_text_turn_keeps_thinking_in_generation_format(tok):
    steps = [ScriptTurn(thinking="check", calls=[ScriptCall("submit_patch", {}, '{"status": "ok", "patch_size": 1, "files_changed": 1}')]),
             ScriptTurn(thinking="all good", text="Fixed it.")]
    res = replay(steps, AGENT, "Task prompt", "problem", "gemma-4-31b-it-qat-w4a16-ct")
    (rec,) = records_from_replay(res, stage="s1_agent", role="agent", instance_id="x", source=SRC,
                                 agent_config=AGENT)
    out = render_record(rec, tok)
    s, e, _ = out.target_spans[-1]
    assert out.text[s:e] == "<|channel>thought\nall good\n<channel|>Fixed it.<turn|>"


def test_nudge_starts_a_new_record(tok):
    ok = '{"status": "ok", "stdout": "", "stderr": "", "exit_code": 0}'
    steps = [ScriptTurn(thinking="t1", calls=[ScriptCall("run_command", {"command": "ls"}, ok)]),
             ScriptTurn(thinking="t2", text="I am stuck."),
             ScriptNudge("Please continue your work using the available tools, or call submit_patch when "
                         "you have completed and verified your changes."),
             ScriptTurn(thinking="t3", calls=[ScriptCall("submit_patch", {}, '{"status": "ok", "patch_size": 1, "files_changed": 1}')]),
             ScriptTurn(text="Done.")]
    res = replay(steps, AGENT, "Task prompt", "problem", "gemma-4-31b-it-qat-w4a16-ct")
    recs = records_from_replay(res, stage="s1_agent", role="agent", instance_id="x", source=SRC,
                               agent_config=AGENT)
    assert [len(r["loss"]) for r in recs] == [2, 2]
    for r in recs:
        render_record(r, tok)                                       # each record is consistent
    # after the nudge, the earlier thought is no longer in the prompt
    assert "t1" not in render_text(tok, recs[1]["messages"][:-1], recs[1]["tools"],
                                   {"enable_thinking": True}, True)


def test_prompt_token_counter(tok, astropy_path):
    (rec,) = build_agent_records(load_trace(astropy_path), BuildConfig(agent_dir=AGENT), SRC)
    req = {"messages": rec["messages"][:-1], "tools": rec["tools"],
           "chat_template_kwargs": {"enable_thinking": True}}
    n = prompt_token_counter(tok)(req)
    out = render_record(rec, tok)
    assert 0 < n < len(out.input_ids)
