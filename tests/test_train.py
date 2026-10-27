"""Trainer tests on tiny random Gemma 4 models (CPU)."""
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
import torch

from tiny_gemma4 import tiny_model

from locfix.train.data import (
    Example,
    RowDataset,
    Source,
    epoch_plan,
    load_sources,
    make_plan,
    pack_rows,
    plan_summary,
)
from locfix.train.export import ExportError, export_adapter, k_eq_v_layers
from locfix.train.loss import reference_weighted_ce_sum, weighted_ce_sum
from locfix.train.model import LossModel, add_lora, lora_module_names

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "tests" / "fixtures" / "records"
TOK_DIR = ROOT / "work" / "model_meta" / "e4b"
needs_tok = pytest.mark.skipif(not (TOK_DIR / "tokenizer.json").exists(), reason="tokenizer not available")


def lora_model(kind="31b", vocab=512, r=4):
    pm = add_lora(tiny_model(kind, vocab=vocab), r=r, alpha=2 * r)
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.05, generator=torch.Generator().manual_seed(zlib.crc32(n.encode())))
    return pm


# --------------------------------------------------------------------------- loss
@pytest.mark.parametrize("kind", ["31b", "e4b"])
def test_chunked_loss_matches_full_logits(kind):
    pm = lora_model(kind)
    lm = LossModel(pm, ce_chunk=5)
    torch.manual_seed(1)
    x = torch.randint(3, 500, (1, 40))
    labels = x.clone()
    labels[:, :12] = -100
    labels[:, 20:25] = -100
    w = (labels != -100).float() * torch.rand(1, 40)
    out = lm(x, torch.arange(40)[None], labels, w)
    ref = reference_weighted_ce_sum(pm(input_ids=x).logits, labels, w)      # HF logits include softcap
    assert torch.allclose(out["loss"], ref, atol=1e-4)
    out["loss"].backward()                                                  # checkpointed chunks backprop
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in pm.named_parameters() if "lora_A" in n)


def test_weighted_ce_empty_and_softcap():
    h = torch.randn(0, 8, requires_grad=True)
    loss, ce = weighted_ce_sum(h, torch.randn(16, 8), torch.zeros(0, dtype=torch.long), torch.zeros(0), 30.0)
    assert float(loss) == 0 and ce.numel() == 0
    h = torch.randn(3, 8) * 100
    W = torch.randn(16, 8)
    t = torch.tensor([1, 2, 3])
    capped, _ = weighted_ce_sum(h, W, t, torch.ones(3), 30.0)
    logits = torch.tanh((h @ W.t()) / 30) * 30
    assert torch.allclose(capped, torch.nn.functional.cross_entropy(logits, t, reduction="sum"), atol=1e-4)


@pytest.mark.parametrize("kind", ["31b", "e4b"])
@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_packed_row_equals_separate_examples(kind, attn):
    """Packing restarts position_ids per example; Gemma 4's full and sliding-window masks must keep
    examples apart (tiny sliding window of 8 so both paths matter)."""
    pm = lora_model(kind)
    pm.get_base_model().config._attn_implementation = attn
    lm = LossModel(pm)
    torch.manual_seed(2)
    xs = [torch.randint(3, 500, (1, n)) for n in (30, 22, 9)]
    sep, parts = 0.0, []
    for x in xs:
        lab = x.clone()
        lab[:, :4] = -100
        w = (lab != -100).float()
        sep += lm(x, torch.arange(x.shape[1])[None], lab, w)["loss"]
        parts.append((x, torch.arange(x.shape[1]), lab, w))
    packed = lm(torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts])[None],
                torch.cat([p[2] for p in parts], 1), torch.cat([p[3] for p in parts], 1))["loss"]
    assert torch.allclose(sep, packed, rtol=1e-5, atol=1e-4)


# --------------------------------------------------------------------------- data plan
def _ex(n, iid, src=0):
    import numpy as np

    lab = np.arange(n, dtype=np.int32)
    lab[0] = -100
    return Example(np.arange(n, dtype=np.int32), lab, np.ones(n, dtype=np.float32), f"r{iid}", iid, "s", src)


def test_epoch_plan_repeat_and_fraction():
    exs = [_ex(10, f"a{i}", 0) for i in range(10)] + [_ex(10, f"b{i}", 1) for i in range(4)]
    srcs = [Source("a.jsonl", repeat=0.5), Source("b.jsonl", repeat=2)]
    e0, e1 = epoch_plan(exs, srcs, 0, 1), epoch_plan(exs, srcs, 1, 1)
    assert len(e0) == len(e1) == 5 + 8
    assert sorted(i for i in e0 if i >= 10) == sorted(list(range(10, 14)) * 2)
    assert {i for i in e0 if i < 10} != {i for i in e1 if i < 10}       # a new half each epoch
    assert epoch_plan(exs, srcs, 0, 1) == e0                              # deterministic


def test_pack_rows_and_trim():
    lengths = [30, 20, 15, 10, 5, 50]
    rows = pack_rows(list(range(6)), lengths, 50, packing=True)
    assert sorted(i for r in rows for i in r) == list(range(6))
    assert all(sum(lengths[i] for i in r) <= 50 for r in rows)
    assert pack_rows([0, 1], lengths, 50, packing=False) == [[0], [1]]
    with pytest.raises(ValueError):
        pack_rows([5], lengths, 40, packing=True)
    exs = [_ex(10, f"t{i % 3}") for i in range(7)]
    plan = make_plan(exs, [Source("x")], epochs=2, rows_per_step=2, max_tokens=100, packing=False, seed=0)
    assert plan.steps == 6 and plan.dropped_rows == 2 and plan.epoch_end_steps() == [3, 6]
    summ = plan_summary(plan, exs, [Source("x")])
    assert summ["tasks"] == 3 and summ["optimizer_steps"] == 6


def test_row_dataset_positions_and_weights():
    exs = [_ex(4, "a", 0), _ex(3, "b", 1)]
    ds = RowDataset([[0, 1]], exs, [Source("a", weight=1.0), Source("b", weight=0.5)])
    item = ds[0]
    assert item["position_ids"].tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert item["loss_weights"].tolist() == [1, 1, 1, 1, 0.5, 0.5, 0.5]
    assert item["source_ids"].tolist() == [0, 0, 0, 0, 1, 1, 1]


@needs_tok
def test_tokenize_sources_and_cache(tmp_path):
    from locfix.render.chat import load_tokenizer

    tok = load_tokenizer(TOK_DIR)
    src = [Source(str(RECORDS / "locfix_fixtures.jsonl"))]
    exs, reps = load_sources(src, tok, 32768, cache_dir=tmp_path)
    assert reps[0].records == 12 and reps[0].examples == 12 and reps[0].render_errors == 0
    again, reps2 = load_sources(src, tok, 32768, cache_dir=tmp_path)            # from cache
    assert len(list(tmp_path.iterdir())) == 1
    assert all((a.input_ids == b.input_ids).all() and (a.weights == b.weights).all() for a, b in zip(exs, again, strict=True))
    short, rep3 = load_sources(src, tok, 3000, cache_dir=tmp_path)
    assert rep3[0].too_long > 0 and len(short) == 12 - rep3[0].too_long


# --------------------------------------------------------------------------- export
def test_export_k_eq_v_copy_and_checks(tmp_path):
    pm = lora_model("31b")
    pm.save_pretrained(tmp_path / "a")
    cfg = pm.get_base_model().config
    assert k_eq_v_layers(cfg) == [2, 5]
    assert not any(".2.self_attn.v_proj" in n for n in lora_module_names(pm))   # HF: no v_proj there
    rep = export_adapter(tmp_path / "a", tmp_path / "out", cfg, serve_id="gemma-4-31b-it-qat-w4a16-ct")
    assert rep["k_eq_v_layers_copied"] == [2, 5] and rep["modules"] == 6 * 7 and rep["dtype"] == "bfloat16"
    from safetensors.torch import load_file

    t = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
    for ab in "AB":
        k = f"base_model.model.model.layers.2.self_attn.%s.lora_{ab}.weight"
        assert torch.equal(t[k % "v_proj"], t[k % "k_proj"])
    acfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert acfg["target_modules"] == sorted(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    assert acfg["base_model_name_or_path"] == "gemma-4-31b-it-qat-w4a16-ct"
    # e4b-like config: no copies
    pe = lora_model("e4b")
    pe.save_pretrained(tmp_path / "e")
    assert export_adapter(tmp_path / "e", tmp_path / "eo", pe.get_base_model().config)["k_eq_v_layers_copied"] == []
    # unknown key is refused
    from safetensors.torch import save_file

    bad = load_file(str(tmp_path / "a" / "adapter_model.safetensors"))
    bad["base_model.model.model.embed_tokens.lora_A.weight"] = torch.zeros(4, 4)
    save_file(bad, str(tmp_path / "a" / "adapter_model.safetensors"))
    with pytest.raises(ExportError, match="unexpected adapter key"):
        export_adapter(tmp_path / "a", tmp_path / "bad", cfg)


# --------------------------------------------------------------------------- end to end
@needs_tok
def test_training_run_cpu(tmp_path):
    from locfix.render.chat import load_tokenizer
    from locfix.train.sft import TrainSettings, run_training

    recs = tmp_path / "loc.jsonl"
    lines = [x for x in (RECORDS / "locfix_fixtures.jsonl").read_text().splitlines()
             if '"stage": "s2_loc_edit"' in x or '"stage": "s3_loc_r0"' in x]
    recs.write_text("\n".join(lines) + "\n")
    s = TrainSettings(output_dir=str(tmp_path / "run"), base_model="unused", sources=[Source(str(recs))],
                      val_sources=[Source(str(recs))], epochs=2, learning_rate=1e-3, lora_r=4, lora_alpha=8,
                      global_batch_rows=2, warmup_steps=0, cache_dir=str(tmp_path / "cache"), bf16=False,
                      serve_id="gemma-4-31b-it-qat-w4a16-ct")
    out = run_training(s, use_cpu=True, model=tiny_model("31b", vocab=262144),
                       tokenizer=load_tokenizer(TOK_DIR))
    run = tmp_path / "run"
    assert out["plan"]["optimizer_steps"] == 4 and out["plan"]["epoch_end_steps"] == [2, 4]
    for d in ("epoch-1", "epoch-2", "final", "export/final", "export/epoch-1"):
        assert (run / d / "adapter_model.safetensors").exists(), d
    vals = [json.loads(x) for x in (run / "val.jsonl").read_text().splitlines()]
    assert len(vals) == 2 and vals[1]["val_nll/all"] < vals[0]["val_nll/all"]
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["extra"]["plan"]["tasks"] == 2 if "extra" in meta else True
    assert out["exports"]["final"]["k_eq_v_layers_copied"] == [2, 5]


@needs_tok
@pytest.mark.multiproc
def test_loss_normalization_devices_vs_accumulation(tmp_path):
    """Spec 05 acceptance: one SGD step on the same 2 rows (different lengths and source weights)
    gives the same update with 1 process × 2 accumulation steps and 2 processes × 1 step, and
    both equal a hand-computed step on Σ w·CE / Σ w."""
    lines = (RECORDS / "locfix_fixtures.jsonl").read_text().splitlines()
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(next(x for x in lines if '"stage": "s2_loc_edit"' in x) + "\n")
    b.write_text([x for x in lines if '"stage": "s3_loc_r0"' in x][-1] + "\n")
    helper = ROOT / "tests" / "helpers" / "norm_run.py"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    args = [str(a), str(b), str(TOK_DIR)]
    for cmd, out in (([sys.executable, str(helper)], "accum"),
                     ([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", str(helper)], "ddp")):
        r = subprocess.run([*cmd, str(tmp_path / out), *args], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
    from safetensors.torch import load_file

    acc = load_file(str(tmp_path / "accum" / "final" / "adapter_model.safetensors"))
    ddp = load_file(str(tmp_path / "ddp" / "final" / "adapter_model.safetensors"))
    assert max(float((acc[k].float() - ddp[k].float()).abs().max()) for k in acc) < 1e-5

    # hand-computed step
    sys.path.insert(0, str(helper.parent))
    from norm_run import init_model, settings

    from locfix.render.chat import load_tokenizer

    s = settings(str(tmp_path / "x"), str(a), str(b), str(tmp_path / "cache2"))
    base = init_model()
    torch.manual_seed(1)                                   # same order as the helper
    pm = add_lora(base, 4, 8)
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.02, generator=torch.Generator().manual_seed(zlib.crc32(n.encode())))
    lm = LossModel(pm)
    exs, _ = load_sources(s.sources, load_tokenizer(TOK_DIR), s.max_seq_len, s.cache_dir)
    plan = make_plan(exs, s.sources, 1, 2, s.max_seq_len, False, s.seed)
    ds = RowDataset(plan.rows, exs, s.sources)
    W = sum(float(ds[k]["loss_weights"].sum()) for k in range(2))
    for k in range(2):
        (lm(**{kk: v.unsqueeze(0) for kk, v in ds[k].items()})["loss"] / W).backward()
    for n, p in pm.named_parameters():
        if p.requires_grad:
            want = (p.data - s.learning_rate * p.grad).float()
            assert torch.allclose(acc[n.replace(".default", "")].float(), want, atol=1e-5), n


def test_buffer_order_is_independent_of_hash_seed():
    """Gemma 4 registers RoPE buffers in set order (hash-seed dependent); DDP broadcasts by
    position. After `stabilize_buffer_order` every process must list buffers identically."""
    code = ("import sys; sys.path.insert(0, 'tests'); from tiny_gemma4 import tiny_model; "
            "from locfix.train.model import stabilize_buffer_order as s; m = tiny_model('31b'); "
            "raw = [n for n, _ in m.named_buffers()]; s(m); print(raw, [n for n, _ in m.named_buffers()])")
    outs = set()
    raws = set()
    for seed in ("1", "2", "3", "4", "5", "6"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(ROOT / "src"))
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        raw, fixed = r.stdout.strip().split("] [")
        raws.add(raw)
        outs.add(fixed)
    assert len(raws) > 1          # the problem exists in this transformers version
    assert len(outs) == 1         # and the fix removes it


def test_parity_hf_side_and_random_adapter(tmp_path):
    """CPU part of the parity check: random adapter → export (k_eq_v copies) → HF log-probs with and
    without it. The vLLM side runs on GPU (notebooks/train_smoke.py)."""
    from locfix.quant.parity import compare, hf_side, random_adapter

    m = tiny_model("31b")
    m.config.architectures = ["Gemma4ForCausalLM"]
    m.save_pretrained(tmp_path / "bf16")
    rep = random_adapter(str(tmp_path / "bf16"), str(tmp_path / "ra"), r=4, alpha=8, scale=0.2)
    assert rep["k_eq_v_layers_copied"] == [2, 5] and rep["modules"] == 42
    torch.manual_seed(0)
    items = [{"id": f"r{i}", "input_ids": torch.randint(3, 500, (n,)).tolist(), "positions": list(range(5, n))}
             for i, n in enumerate((20, 33))]
    (tmp_path / "items.json").write_text(json.dumps(items))
    hf_side(str(tmp_path / "bf16"), None, str(tmp_path / "items.json"), str(tmp_path / "base.json"))
    hf_side(str(tmp_path / "bf16"), str(tmp_path / "ra" / "export"), str(tmp_path / "items.json"),
            str(tmp_path / "ad.json"))
    base = json.loads((tmp_path / "base.json").read_text())
    ad = json.loads((tmp_path / "ad.json").read_text())
    assert len(base["r1"]) == 28
    assert compare(base, base)["max"] == 0
    assert compare(ad, base)["mean"] > 1e-3                    # the adapter changes the outputs
