"""Stage 3 round 0 and the signal test (spec 06): folds, candidates, scoring sequences, weights,
offline rollouts."""
import json
from pathlib import Path

import pytest

from locfix.em.folds import fold_split, hash_fold, load_folds, other

ROOT = Path(__file__).resolve().parents[1]


def _rec(iid: str, fold: str | None, stage: str = "s3_loc_r0") -> dict:
    return {"id": f"{stage}__{iid}", "stage": stage, "instance_id": iid,
            "source": {"set": "solved", "fold": fold} if fold else {"set": "solved"}}


# --------------------------------------------------------------------------- folds
def test_fold_split_by_source(tmp_path):
    d = tmp_path / "locfix"
    d.mkdir()
    rows = [_rec(f"t{i}", "A" if i % 2 else "B") for i in range(10)]
    (d / "s3_loc_r0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "s2_fix_gold.jsonl").write_text("".join(json.dumps(_rec(r["instance_id"], r["source"]["fold"],
                                                                 "s2_fix_gold")) + "\n" for r in rows * 2))
    s = fold_split([str(d)], tmp_path / "folds")
    assert s["tasks"] == {"A": 5, "B": 5}
    assert s["files"]["locfix/s2_fix_gold.jsonl"] == {"A": 10, "B": 10}
    a = [json.loads(x) for x in (tmp_path / "folds/A/locfix/s3_loc_r0.jsonl").read_text().splitlines()]
    assert {r["source"]["fold"] for r in a} == {"A"} and len(a) == 5
    folds = load_folds(tmp_path / "folds")
    assert folds["t1"] == "A" and folds["t2"] == "B"
    assert other("A") == "B" and other("B") == "A"


def test_fold_split_refuses_missing_or_one_sided_folds(tmp_path):
    f = tmp_path / "s3_loc_r0.jsonl"
    f.write_text("".join(json.dumps(_rec(f"t{i}", "A")) + "\n" for i in range(10)))
    with pytest.raises(SystemExit, match="smaller fold"):
        fold_split([str(f)], tmp_path / "o1")
    f.write_text("".join(json.dumps(_rec(f"t{i}", None)) + "\n" for i in range(10)))
    with pytest.raises(SystemExit, match="--by hash"):
        fold_split([str(f)], tmp_path / "o2")
    s = fold_split([str(f)], tmp_path / "o3", by="hash", min_share=0.1)
    assert sum(s["tasks"].values()) == 10
    assert all(load_folds(tmp_path / "o3")[f"t{i}"] == hash_fold(f"t{i}", 1234) for i in range(10))


def test_round0_run_configs_resolve():
    from locfix.config import load_config
    from locfix.train.sft import TrainSettings

    for name, role, n_src in (("S3r0-e4b-loc-A", "localizer", 1), ("S3r0-e4b-loc-B", "localizer", 1),
                              ("S3r0-e4b-fix-A", "fixer", 2), ("S3r0-e4b-fix-B", "fixer", 2),
                              ("S3r0-e4b-localizer", "localizer", 1)):
        cfg = load_config([str(ROOT / "configs/runs" / f"{name}.yaml")])
        t = TrainSettings.from_config(cfg)
        assert cfg["train"]["role"] == role and len(t.sources) == n_src, name
        fold = name[-1] if name[-2] == "-" else None
        if fold:
            assert all(f"/folds/{fold}/" in s.path for s in t.sources), name
            assert t.output_dir.endswith(("loc_" + fold, "fix_" + fold)), name


# --------------------------------------------------------------------------- shared fixtures
import os  # noqa: E402

from locfix.config import load_config  # noqa: E402
from locfix.data.loc_fix_records import Task, build_task_records, patch_from_edits  # noqa: E402
from locfix.data.locations import Location  # noqa: E402
from locfix.data.traces import load_trace  # noqa: E402
from locfix.data.zloc import z_to_json  # noqa: E402
from locfix.harness.prompts import parse_task_prompt  # noqa: E402

CODE = Path(__file__).parent / "fixtures" / "code"
TOK_DIR = Path(os.environ.get("LOCFIX_TOKENIZER_DIR", ROOT / "work" / "model_meta" / "e4b"))
needs_tok = pytest.mark.skipif(not (TOK_DIR / "tokenizer.json").exists(), reason="Gemma 4 tokenizer not available")


def make_task(path, fold: str = "B") -> Task:
    tr = load_trace(path)
    files = json.loads((CODE / f"{tr.instance_id}.json").read_text())
    return Task(tr, files, patch_from_edits(tr, files, strict=False),
                parse_task_prompt(tr.task_prompt).problem_statement,
                source={"set": "solved", "dataset": "fixture", "fold": fold, "attempt": 1})


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def lc(cfg):
    from locfix.em.tasks import locfix_config

    return locfix_config(cfg)


@pytest.fixture(scope="module")
def gunicorn(lc):
    from conftest import trace_path
    from locfix.data.loc_fix_records import task_locations

    t = make_task(trace_path("benoitc__gunicorn"))
    recs = build_task_records(t, lc)
    return t, task_locations(t), recs


# --------------------------------------------------------------------------- candidates
def test_parse_sample():
    from locfix.em.candidates import parse_sample

    z = [Location("a.py", "f", 1, 3, "edit")]
    ok = "<|channel>thought\nI have found it.\n<channel|>" + z_to_json(z) + "<turn|>"
    assert parse_sample(ok) == (z, "ok")
    assert parse_sample(z_to_json(z))[1] == "ok"
    assert parse_sample("Here: " + z_to_json(z) + " done")[1] == "ok"
    assert parse_sample("<|channel>thought\nlet me look<channel|><|tool_call>call:read_file{}")[1] == "tool_call"
    assert parse_sample("<|channel>thought\nstill thinking")[1] == "unfinished_thinking"
    assert parse_sample("<|channel>thought\nx<channel|>not json")[1] == "bad_json"
    assert parse_sample('{"locations": [{"path": "a.py"}]}')[1] == "bad_json"


def test_clean_z_and_assemble():
    from locfix.em.candidates import assemble, clean_z

    files = {"a.py": "\n".join(f"line{i}" for i in range(1, 41)) + "\n", "b.py": "x = 1\ny = 2\n"}
    em = {"K": 4, "max_locations": 8, "ctx_budget_tokens": 6000}
    z, st = clean_z([Location("/workspace/a.py", "f", 30, 99, "edit"), Location("c.py", "g", 1, 2),
                     Location("./b.py", "<file>", 2, 1, "weird")], files, em)
    assert z == [Location("a.py", "f", 30, 40, "edit"), Location("b.py", "<file>", 1, 2, "context")]
    assert st["unknown_path"] == 1 and st["clamped"] == 2

    zt = [Location("a.py", "f", 1, 10, "edit"), Location("b.py", "<file>", 1, 2)]
    s = lambda z: "<|channel>thought\nok\n<channel|>" + z_to_json(z)  # noqa: E731
    samples = [s(zt), s(zt[:1]), s(zt[:1]), "garbage", s([Location("zz.py", "q", 1, 2)]),
               s([Location("b.py", "<file>", 1, 2), Location("a.py", "f", 1, 10, "edit")])]
    pert = {"edit_only": zt[:1], "view_only": [Location("b.py", "<file>", 1, 2, "context")], "gold_minus_one": None}
    a = assemble(zt, samples, pert, files, em)
    srcs = [c.source for c in a.candidates]
    # the reordered sample becomes z_T again (edit locations first); edit_only == the first sample
    assert srcs == ["teacher", "sample", "view_only"]
    assert a.candidates[0].z == zt and [c.cand_id for c in a.candidates] == [0, 1, 2]
    assert a.stats["n_samples_equal_teacher"] == 2 and a.stats["duplicate"] == 3
    assert a.stats["parse"] == {"ok": 5, "bad_json": 1} and a.stats["empty"] == 1
    assert len(assemble(zt, samples * 3, pert, files, {**em, "K": 1}).candidates) == 2   # K + 1 cap


@needs_tok
def test_stage3_tasks_and_localizer_prompt_match_training_render(tmp_path, gunicorn):
    """The candidate prompt is exactly the context the round-0 localizer was trained to answer in."""
    from locfix.em.candidates import render_prompt_ids
    from locfix.em.tasks import read_stage3_tasks
    from locfix.render.chat import load_tokenizer, render_record

    task, tl, recs = gunicorn
    loc = [r for r in recs if r["stage"] == "s3_loc_r0"]
    p = tmp_path / "s3_loc_r0.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in loc))
    (st,) = read_stage3_tasks(p, {task.instance_id: "B"})
    assert st.fold == "B" and st.trace == Path(task.trace.path).name and st.z_teacher == loc[0]["z"]
    tok = load_tokenizer(TOK_DIR)
    r = render_record(loc[-1], tok)
    start = r.target_spans[-1][0]
    ids = render_prompt_ids(tok, st.prompt)
    assert tok(r.text[:start], add_special_tokens=False)["input_ids"] == ids
    with pytest.raises(ValueError, match="folds.json"):
        read_stage3_tasks(p, {})


# --------------------------------------------------------------------------- scoring sequences
def test_added_line_spans_and_overlap():
    from locfix.score.spans import SpanError, added_line_offsets, tokens_overlapping, value_span

    args = {"old_string": "a = 1\nb = 2", "new_string": "a = 1\nb = 3\n\nc = 4"}
    offs = added_line_offsets("edit_file", args)
    assert [args["new_string"][a:b] for a, b in offs] == ["b = 3", "c = 4"]
    assert [x for a, b in added_line_offsets("write_file", {"content": "x\n\ny"}) for x in ["x\n\ny"[a:b]]] == ["x", "y"]
    t = 'call:edit_file{filepath:<|"|>/w/a.py<|"|>,new_string:<|"|>NEW<|"|>,old_string:<|"|>OLD<|"|>}'
    s, e = value_span(t, "new_string", "NEW", base=100)
    assert t[s - 100:e - 100] == "NEW"
    with pytest.raises(SpanError):
        value_span(t, "new_string", "MISSING")
    assert tokens_overlapping([(0, 3), (3, 6), (6, 6), (6, 9), (9, 12)], [(4, 7), (11, 20)]) == [1, 3, 4]


@needs_tok
def test_scoring_sequence_gold_matches_training_records(gunicorn, lc):
    """For z = z_T without requests, the scoring context up to the first edit is the s2_fix_gold
    training record's context up to the teacher's first edit; y*₊ tokens decode to the added lines."""
    from locfix.em.candidates import Candidate
    from locfix.render.chat import load_tokenizer
    from locfix.score.sequences import canonical_edits, sequences_for_task
    from locfix.score.spans import added_line_offsets

    task, tl, recs = gunicorn
    tok = load_tokenizer(TOK_DIR)
    gold = next(r for r in recs if r["stage"] == "s2_fix_gold")
    zt = [Location(e["path"], e["symbol"], *e["lines"], e["role"]) for e in gold["z"]]
    edit_only = [x for x in zt if x.role == "edit"]
    items, fails = sequences_for_task(task, tl, [Candidate(0, "teacher", zt), Candidate(1, "x", edit_only)], lc, tok)
    assert fails == []
    by = {(i["cand_id"], i["mode"]): i for i in items}
    from locfix.score.sequences import misses
    assert {(0, "none"), (1, "none"), (1, "recover")} <= set(by)
    assert ((0, "recover") in by) == misses(tl, zt, task.files)       # only when z_T was pruned

    # the context the canonical edits are scored in == the training record's context
    teacher_first_edit = next(e["message_index"] for e in gold["loss"]
                              if gold["messages"][e["message_index"]].get("tool_calls")
                              and gold["messages"][e["message_index"]]["tool_calls"][0]["function"]["name"]
                              in ("edit_file", "write_file"))
    from locfix.render.chat import render_text
    kw = gold["meta"]["request_params"].get("chat_template_kwargs") or {}
    ctx = render_text(tok, gold["messages"][:teacher_first_edit], gold["tools"], kw, True)
    ids = tok(ctx, add_special_tokens=False)["input_ids"]
    it = by[(0, "none")]
    assert it["input_ids"][: len(ids)] == ids

    # y*₊ tokens decode to exactly the added lines (in order)
    added = []
    for tool, args, _ in canonical_edits(task):
        text = args["content"] if tool == "write_file" else args["new_string"]
        added += [text[a:b] for a, b in added_line_offsets(tool, args)]
    dec = tok.decode([it["input_ids"][t] for t in it["positions"]])
    pos = 0
    for line in added:
        j = dec.find(line, pos)
        assert j >= 0, line
        pos = j + len(line)
    assert it["n_plus_tokens"] == len(it["positions"]) > 0
    # same patch, same y*₊ token count, for every candidate; requests add context
    assert len({i["n_plus_tokens"] for i in items}) == 1
    assert by[(1, "none")]["n_z_tokens"] < by[(0, "none")]["n_z_tokens"]
    assert by[(1, "recover")]["n_requests"] > 0 and by[(1, "none")]["n_requests"] == 0
    assert by[(1, "recover")]["n_ctx_tokens"] > by[(1, "none")]["n_ctx_tokens"]


# --------------------------------------------------------------------------- end-to-end data (CPU)
@pytest.fixture(scope="module")
def stage3_data(tmp_path_factory, lc, cfg):
    """A miniature Stage 3 input set from the fixture traces: solved.jsonl, records, folds, samples."""
    from conftest import TRACES

    from locfix.em.folds import fold_split

    d = tmp_path_factory.mktemp("s3")
    (d / "records" / "locfix").mkdir(parents=True)
    solved, recs = [], []
    for i, p in enumerate(sorted(TRACES.glob("*.json"))):
        fold = "AB"[i % 2]
        t = make_task(p, fold)
        solved.append({"instance_id": t.instance_id, "problem_statement": t.problem_statement, "fold": fold,
                       "verifier": "fixture", "teacher": {"patch": t.patch}})
        try:
            recs += build_task_records(t, lc)
        except Exception:  # noqa: BLE001 - tasks the builder skips have no records
            pass
    (d / "solved.jsonl").write_text("".join(json.dumps(r) + "\n" for r in solved))
    for st in ("s3_loc_r0", "s2_fix_gold", "s2_fix_noisy"):
        (d / "records/locfix" / f"{st}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs if r["stage"] == st))
    fold_split([str(d / "records/locfix")], d / "records/folds", min_share=0.2)
    loc = [r for r in recs if r["stage"] == "s3_loc_r0"]
    rows = []
    for r in loc:
        zt = [Location(e["path"], e["symbol"], *e["lines"], e["role"]) for e in r["z"]]
        texts = ["<|channel>thought\nFound.\n<channel|>" + z_to_json(zt[:1]),
                 "<|channel>thought\nFound.\n<channel|>" + z_to_json(zt), "not json"]
        rows.append({"instance_id": r["instance_id"], "fold": r["source"]["fold"], "localizer": "loc_x",
                     "texts": texts})
    (d / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


@needs_tok
def test_prepare_cli(stage3_data, cfg):
    from locfix.cli import main

    d = stage3_data
    out = d / "prepare"
    args = ["score.prepare", "--loc-records", str(d / "records/locfix/s3_loc_r0.jsonl"),
            "--folds", str(d / "records/folds"), "--samples", str(d / "samples.jsonl"),
            "--traces", str(ROOT / "tests/fixtures/traces"), "--solved", str(d / "solved.jsonl"),
            "--code", str(CODE), "--tokenizer", str(TOK_DIR), "--out", str(out), "--workers", "1"]
    assert main(args) == 0
    cands = [json.loads(x) for x in (out / "candidates.jsonl").read_text().splitlines()]
    items = [json.loads(x) for x in (out / "items.jsonl").read_text().splitlines()]
    counts = json.loads((out / "prepare_counts.json").read_text())
    n_tasks = len((d / "records/locfix/s3_loc_r0.jsonl").read_text().splitlines())
    assert len(cands) == counts["tasks"] == n_tasks >= 2 and not any(k.startswith("skip") for k in counts)
    for c in cands:
        srcs = [x["source"] for x in c["candidates"]]
        assert srcs[0] == "teacher" and c["assemble"]["n_samples_equal_teacher"] == 1
        assert c["assemble"]["parse"] == {"ok": 2, "bad_json": 1}
    ids = {(i["instance_id"], i["cand_id"], i["mode"]) for i in items}
    for c in cands:
        assert {(c["instance_id"], x["cand_id"], "none") for x in c["candidates"]} <= ids
    assert all(i["fold"] in "AB" and i["n_plus_tokens"] > 0 for i in items)


# --------------------------------------------------------------------------- scorer (fake vLLM)
def install_fake_vllm(monkeypatch, logprob_fn):
    """A stand-in for vllm (LLM.generate with prompt_logprobs) so the scorer runs on CPU.
    logprob_fn(lora_name, token_ids, t) -> log p of token t."""
    import sys
    import types

    class LP:
        def __init__(self, v):
            self.logprob = v

    class Out:
        def __init__(self, plp):
            self.prompt_logprobs = plp
            self.outputs = []

    class LLM:
        kwargs = None

        def __init__(self, **kw):
            LLM.kwargs = kw

        def generate(self, prompts, sp, lora_request=None, use_tqdm=True):
            assert sp.prompt_logprobs == 0 and sp.max_tokens == 1
            return [Out([None] + [{p.ids[t]: LP(logprob_fn(lora_request.name, p.ids, t))}
                                  for t in range(1, len(p.ids))]) for p in prompts]

    class SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class TokensPrompt:
        def __init__(self, prompt_token_ids):
            self.ids = prompt_token_ids

    class LoRARequest:
        def __init__(self, name, i, path):
            self.name, self.i, self.path = name, i, path

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = TokensPrompt
    lora = types.ModuleType("vllm.lora")
    req = types.ModuleType("vllm.lora.request")
    req.LoRARequest = LoRARequest
    for name, mod in (("vllm", vllm), ("vllm.inputs", inputs), ("vllm.lora", lora), ("vllm.lora.request", req)):
        monkeypatch.setitem(sys.modules, name, mod)
    return LLM


def test_scorer_routes_folds_and_averages_plus_tokens(tmp_path, monkeypatch):
    from locfix.score.vllm_scorer import iter_items, score

    LLM = install_fake_vllm(monkeypatch, lambda lora, ids, t: -1.0 if lora == "fix_A" else -2.0)
    items = [{"instance_id": f"t{i}", "fold": "AB"[i % 2], "cand_id": c, "source": "x", "mode": "none",
              "input_ids": list(range(10 + c)), "positions": [3, 4, 5 + c], "n_plus_tokens": 3}
             for i in range(4) for c in range(2)]
    p = tmp_path / "items.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in items))
    ads = {}
    for f in "AB":
        (tmp_path / f).mkdir()
        (tmp_path / f / "adapter_config.json").write_text(json.dumps({"r": 16}))
        ads[f] = str(tmp_path / f)
    rows = list(score(iter_items([str(p)], None), ct="ct", adapters=ads, max_model_len=11, gpu_util=0.5,
                      tp=1, chunk=3, text_only=True))
    by = {(r["instance_id"], r["cand_id"]): r for r in rows}
    assert len(rows) == 8 and sum("error" in r for r in rows) == 4 and LLM.kwargs["enable_lora"] and LLM.kwargs["max_loras"] == 2
    assert by[("t0", 0)]["fixer"] == "fix_B" and by[("t0", 0)]["logp_plus_mean"] == -2.0   # fold A task
    assert by[("t1", 0)]["fixer"] == "fix_A" and by[("t1", 0)]["logp_plus_sum"] == -3.0    # fold B task
    assert by[("t1", 1)]["error"] == "too_long"                     # 11 tokens >= max_model_len
    assert len(iter_items([str(p)], (0, 2))) + len(iter_items([str(p)], (1, 2))) == 8
    assert {json.loads(x)["instance_id"] for _, x in iter_items([str(p)], (0, 2))} == {"t0", "t2"}


# --------------------------------------------------------------------------- weights
def test_tau_bisection_hits_ess_target():
    import random

    from locfix.score.weights import ess, weights_for_ess

    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(2, 12)
        sc = [rng.gauss(-0.5, 0.3) for _ in range(n)]
        for target in (1, 2, 4):
            w, tau, e = weights_for_ess(sc, target)
            assert abs(sum(w) - 1) < 1e-9
            if target == 1:
                assert w == [1.0 if k == max(range(n), key=lambda i: sc[i]) else 0.0 for k in range(n)]
            elif target >= n:
                assert e == n
            else:
                assert abs(ess(w) - target) <= 0.01 * target, (sc, target, e)
                assert max(range(n), key=lambda i: w[i]) == max(range(n), key=lambda i: sc[i])
    assert weights_for_ess([-1.0, -1.0, -1.0], 2)[0] == [1 / 3] * 3


def _score_rows(iid, fold, vals):
    """vals: [(cand_id, mode, logp_mean, n_z_tokens, n_requests)]"""
    return [{"instance_id": iid, "fold": fold, "cand_id": c, "mode": m, "logp_plus_mean": lp,
             "n_z_tokens": zt, "n_requests": nr} for c, m, lp, zt, nr in vals]


def test_weights_calibration_and_z_star():
    from locfix.score.weights import compute_weights

    z = lambda *names: [{"path": "a.py", "symbol": s, "lines": [1, 2], "role": "edit" if s == "f" else "context"}  # noqa: E731
                        for s in names]
    cands = [{"instance_id": "t1", "fold": "A", "l_edit": z("f"),
              "candidates": [{"cand_id": 0, "source": "teacher", "z": z("f", "g", "h")},
                             {"cand_id": 1, "source": "sample", "z": z("f")},
                             {"cand_id": 2, "source": "sample", "z": z("g")}]},
             {"instance_id": "t2", "fold": "B", "l_edit": z("f"),
              "candidates": [{"cand_id": 0, "source": "teacher", "z": z("f", "g")},
                             {"cand_id": 1, "source": "edit_only", "z": z("f")}]}]
    scores = (_score_rows("t1", "A", [(0, "none", -0.50, 3000, 0), (1, "none", -0.52, 1000, 0),
                                      (2, "none", -1.40, 1000, 0), (2, "recover", -0.55, 1000, 2)])
              + _score_rows("t2", "B", [(0, "none", -0.30, 2000, 0), (1, "none", -0.90, 1000, 0),
                                        (1, "recover", -0.31, 1000, 1)])
              + [{"instance_id": "t2", "cand_id": 1, "mode": "x", "error": "too_long"}])
    em = {"ctx_cost_sigma": 0.25, "req_cost_sigma": 0.25, "ess_target": 2}
    rows, s = compute_weights(scores, cands, em)
    import statistics
    sig = statistics.median([statistics.pstdev([-0.50, -0.52, -1.40]), statistics.pstdev([-0.30, -0.90])])
    assert abs(s["sigma"] - sig) < 1e-5 and s["median_z_tokens"] == 1000
    assert abs(s["lambda_ctx"] - 0.25 * sig) < 1e-5 and abs(s["lambda_req"] - 0.25 * sig) < 1e-5
    by = {r["instance_id"]: r for r in rows}
    # t1: the small z (cand 1) nearly matches z_T's log p with a third of the context -> z*
    assert by["t1"]["z_star"] == 1 and not by["t1"]["z_star_is_teacher"] and by["t1"]["z_star_tokens"] == 1000
    assert abs(sum(c["w_z"] for c in by["t1"]["candidates"]) - 1) < 1e-5
    assert abs(by["t1"]["ess"] - 2) <= 0.02
    # t2: z_T wins; cand 1 recovers by requests but pays λ_req
    assert by["t2"]["z_star"] == 0 and by["t2"]["z_star_is_teacher"]
    assert s["skipped"] == {"pair:too_long": 1} and s["z_star_is_teacher"] == 0.5
    hard, s1 = compute_weights(scores, cands, em, ess_target=1)
    assert all(sorted(p["w"] for p in r["pairs"])[-1] == 1.0 for r in hard)
    fixed, s2 = compute_weights(scores, cands, em, lambda_ctx=0.0, lambda_req=0.0)
    assert s2["lambda_ctx"] == 0.0 and {r["instance_id"]: r["z_star"] for r in fixed}["t1"] == 0


# --------------------------------------------------------------------------- signal-test rollouts
def _calls(*calls):
    return [{"id": f"live_{i}", "type": "function",
             "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]


def scripted_model(turns):
    """A stand-in for the vLLM fixer: returns the given turns in order (list of tool-call lists or
    a text), recording each request."""
    from locfix.harness.scripted_server import chat_response

    seen = []

    def fn(req):
        seen.append(req)
        t = turns[len(seen) - 1]
        if isinstance(t, str):
            return chat_response("fix_A", t, "thinking", None)
        return chat_response("fix_A", None, "Let me do it.", _calls(*t))
    fn.seen = seen
    return fn


def test_rollout_oracle_edit_reproduces_patch_and_prompt_equals_scoring_context(gunicorn, lc):
    from locfix.em.rollout import rollout
    from locfix.score.sequences import build_steps, canonical_edits, run_replay

    task, tl, recs = gunicorn
    gold = next(r for r in recs if r["stage"] == "s2_fix_gold")
    zt = [Location(e["path"], e["symbol"], *e["lines"], e["role"]) for e in gold["z"]]
    edits = canonical_edits(task)
    model = scripted_model([[(t, a) for t, a, _ in edits], [("run_command", {"command": "python -m pytest -q"})]])
    r = rollout(task, tl, zt, lc, model)
    assert r.stop == "command_after_edit" and r.model_calls == 2 and r.error is None
    assert r.metrics["all_hunks"] and r.metrics["any_hunk"] and r.metrics["first_edit_loc"]
    assert r.metrics["added_recall"] == 1.0 and r.metrics["n_edits"] == len(edits)

    # the fixer's first live request == the context the scorer used for the same z (mode none)
    scripts, _ = build_steps(task, tl, zt, "none", lc, edits)
    res = run_replay(task, scripts, lc)
    fx = res.of_stream("fixer")
    k = next(i for i, m in enumerate(fx.turn_meta) if m.get("synthetic") == "canonical_edit")
    assert model.seen[0]["messages"] == fx.agent_exchanges[k].request["messages"]
    assert model.seen[0]["tools"] == fx.agent_exchanges[k].request["tools"]
    # the second live request carries the harness's edit outputs (with budget warnings) back
    assert len(model.seen[1]["messages"]) == len(model.seen[0]["messages"]) + 1 + len(edits)


def test_rollout_stops_on_what_cannot_run_offline(gunicorn, lc):
    from locfix.em.rollout import rollout

    task, tl, _ = gunicorn
    z = [x for x in tl.l_edit]
    path = "/workspace/" + z[0].path
    # read a known file, then a command nobody recorded
    r = rollout(task, tl, z, lc, scripted_model([[("read_file", {"filepath": path, "start_line": 1, "end_line": 5})],
                                                  [("run_command", {"command": "ls /nonexistent"})]]))
    assert r.stop == "command" and r.model_calls == 2 and not r.metrics["edited"] and not r.metrics["all_hunks"]
    # a teacher command before any edit is served from the trace
    cmd = next(iter(c.args["command"] for t in tl.pre_turns for c in t.tool_calls if c.tool == "run_command"))
    r = rollout(task, tl, z, lc, scripted_model([[("run_command", {"command": cmd})], "I am done."]))
    assert r.stop == "text_only" and r.model_calls == 2
    # unknown file, second localizer call, model error, call cap
    assert rollout(task, tl, z, lc, scripted_model([[("read_file", {"filepath": "/workspace/nope.py"})]])).stop == "unknown_file"
    assert rollout(task, tl, z, lc, scripted_model([[("localizer", {"request": "more"})]])).stop == "localizer_again"

    def boom(req):
        raise ConnectionError("server down")
    r = rollout(task, tl, z, lc, boom)
    assert r.stop == "model_error" and "server down" in r.error
    loop = scripted_model([[("read_file", {"filepath": path})]] * 10)
    assert rollout(task, tl, z, lc, loop, max_model_calls=3).stop == "max_model_calls"


def test_patch_metrics_wrong_edit(gunicorn, lc):
    from locfix.em.rollout import rollout
    from locfix.score.sequences import canonical_edits

    task, tl, _ = gunicorn
    tool, args, _ = canonical_edits(task)[0]
    wrong = dict(args, new_string=args["new_string"].replace("\n", "\n# changed\n", 1))
    r = rollout(task, tl, tl.l_edit, lc, scripted_model([[(tool, wrong)], [("run_command", {"command": "pytest"})]]))
    assert r.metrics["edited"] and r.metrics["first_edit_loc"] and not r.metrics["all_hunks"]
    assert 0 < (r.metrics["added_recall"] or 0) <= 1


# --------------------------------------------------------------------------- signal test report
def _w(iid, star_teacher, zt=3000, zs=1000):
    return {"instance_id": iid, "fold": "A", "z_star_is_teacher": star_teacher, "z_star_source": "teacher" if star_teacher else "sample",
            "teacher_tokens": zt, "z_star_tokens": zt if star_teacher else zs}


def _r(iid, cond, ok, edited=True, stop="command_after_edit"):
    return {"instance_id": iid, "condition": cond, "stop": stop, "all_hunks": ok, "any_hunk": ok,
            "first_edit_loc": edited, "edited": edited, "added_recall": 1.0 if ok else 0.2}


def test_signal_report_and_decisions():
    from locfix.em.signal import decide, report

    # 40 tasks: 20 where z* = z_T (one rollout), 20 where it differs; S wins 13, loses 1
    w, rows = {}, []
    for i in range(40):
        iid = f"t{i}"
        w[iid] = _w(iid, i < 20)
        rows.append(_r(iid, "T", ok=(i % 3 == 0) or i == 20))
        if i >= 20:
            rows.append(_r(iid, "S", ok=(i % 3 == 0) or i >= 21))
    rep = report(rows, w)
    m = rep["metrics"]["all_hunks"]
    assert rep["tasks"] == 40 and rep["z_star_differs"] == 20
    assert m["only_S"] == len([i for i in range(21, 40) if i % 3]) and m["only_T"] == 1
    assert rep["metrics_where_z_star_differs"]["all_hunks"]["tasks"] == 20
    assert rep["z_tokens"]["cut"] == round(1 - (20 * 3000 + 20 * 1000) / (40 * 3000), 4)
    assert rep["decision"] == "pass" and m["p_mcnemar"] < 0.1

    # tie with a much smaller z* -> pass_cheaper; tie without -> fail; infra failures left out
    tie = [_r(f"t{i}", c, ok=i % 2 == 0) for i in range(40) for c in ("T", "S") if c == "T" or i >= 20]
    tie.append({"instance_id": "t99", "condition": "T", "stop": "model_error"})
    w["t99"] = _w("t99", True)
    rep = report(tie, w)
    assert rep["decision"] == "pass_cheaper" and rep["excluded_broken"] == 1 and rep["tasks"] == 40
    w2 = {k: {**v, "z_star_tokens": v["teacher_tokens"]} for k, v in w.items()}
    assert report(tie, w2)["decision"] == "fail"
    # the fixer rarely edits -> uninformative
    lazy = [_r(f"t{i}", c, ok=False, edited=False, stop="command") for i in range(40) for c in ("T", "S")]
    assert report(lazy, w)["decision"] == "uninformative"
    # a gain without significance -> more tasks
    few = {"metrics": {"all_hunks": {"rate_T": 0.2, "diff_S_minus_T": 0.05, "p_mcnemar": 0.4},
                       "edited": {"rate_T": 0.9}}, "z_tokens": {"cut": 0.0}}
    assert decide(few)["decision"] == "more_tasks"


@needs_tok
def test_signal_cli_end_to_end_with_stand_in_fixer(stage3_data, cfg, monkeypatch):
    """prepare -> (stand-in) scores -> weights -> signal test, with an oracle fixer that writes the
    verified patch: both conditions reproduce it on every task with full code."""
    from locfix.cli import main
    from locfix.em import rollout as ro
    from locfix.em.tasks import read_jsonl
    from locfix.score.sequences import canonical_edits

    d = stage3_data
    prep = d / "prepare"
    if not (prep / "items.jsonl").exists():
        assert main(["score.prepare", "--loc-records", str(d / "records/locfix/s3_loc_r0.jsonl"),
                     "--folds", str(d / "records/folds"), "--samples", str(d / "samples.jsonl"),
                     "--traces", str(ROOT / "tests/fixtures/traces"), "--solved", str(d / "solved.jsonl"),
                     "--code", str(CODE), "--tokenizer", str(TOK_DIR), "--out", str(prep), "--workers", "1"]) == 0
    items = read_jsonl(prep / "items.jsonl")
    scores = [{k: it[k] for k in ("instance_id", "fold", "cand_id", "source", "mode", "n_requests", "n_z_tokens")}
              | {"logp_plus_mean": -0.5 - 0.0001 * it["n_z_tokens"] + (0.3 if it["source"] == "sample" else 0.0)}
              for it in items]
    (d / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
    assert main(["score.weights", "--scores", str(d / "scores.jsonl"), "--candidates", str(prep / "candidates.jsonl"),
                 "--out", str(d / "weights")]) == 0
    ws = read_jsonl(d / "weights/weights.jsonl")
    assert all(w["trace"] for w in ws) and any(not w["z_star_is_teacher"] for w in ws)

    loader = {}

    def oracle_factory(url, model, seed=0, timeout=0):
        from locfix.harness.scripted_server import chat_response

        def fn(req):
            head = "".join(str(m.get("content")) for m in req["messages"][:3])
            iid = next(i for i, t in loader.items() if t.problem_statement.strip()[:120] in head)
            task = loader[iid]
            if not any('"edit_file"' in json.dumps(m.get("tool_calls") or "") for m in req["messages"]):
                return chat_response(model, None, "fix", _calls(*[(t, a) for t, a, _ in canonical_edits(task)]))
            return chat_response(model, None, "test", _calls(("run_command", {"command": "pytest -q"})))
        return fn

    from locfix.em.tasks import TaskLoader
    tl = TaskLoader(ROOT / "tests/fixtures/traces", d / "solved.jsonl", [str(CODE)], cfg)
    for w in ws:
        loader[w["instance_id"]] = tl.load(w["trace"])[0]
    monkeypatch.setattr(ro, "http_model_fn", oracle_factory)
    out = d / "signal"
    args = ["em.signal-test", "--weights", str(d / "weights/weights.jsonl"), "--traces", str(ROOT / "tests/fixtures/traces"),
            "--solved", str(d / "solved.jsonl"), "--code", str(CODE), "--out", str(out),
            "--server-url", "http://stand-in/v1", "--workers", "2", "--limit", "0"]
    assert main(args) == 0
    rep = json.loads((out / "signal_report.json").read_text())
    rows = read_jsonl(out / "rollouts.jsonl")
    assert {r["fixer"] for r in rows} <= {"fix_A", "fix_B"}
    assert all(r["fixer"] != f"fix_{r['fold']}" for r in rows)                  # cross-fitted
    assert rep["tasks"] >= 2 and rep["metrics"]["all_hunks"]["rate_T"] == rep["metrics"]["all_hunks"]["rate_S"] == 1.0
    assert rep["decision"] in ("pass_cheaper", "fail")
    # re-running continues (nothing left to do) and rebuilds the same report
    assert main(args) == 0
    assert len(read_jsonl(out / "rollouts.jsonl")) == len(rows)


@needs_tok
def test_prepare_refuses_records_built_with_other_agent_files(gunicorn, lc):
    from locfix.em.candidates import Candidate
    from locfix.render.chat import load_tokenizer
    from locfix.score.sequences import RecordsMismatch, sequences_for_task

    task, tl, recs = gunicorn
    loc = next(r for r in recs if r["stage"] == "s3_loc_r0")
    zt = [Location(e["path"], e["symbol"], *e["lines"], e["role"]) for e in loc["z"]]
    tok = load_tokenizer(TOK_DIR)
    good = loc["messages"][:-1]
    items, _ = sequences_for_task(task, tl, [Candidate(0, "teacher", zt)], lc, tok, ("none",), expect_loc_messages=good)
    assert len(items) == 1
    old = json.loads(json.dumps(good).replace("at most 15 tool calls", "at most 12 tool calls"))
    assert old != good
    with pytest.raises(RecordsMismatch):
        sequences_for_task(task, tl, [Candidate(0, "teacher", zt)], lc, tok, ("none",), expect_loc_messages=old)
