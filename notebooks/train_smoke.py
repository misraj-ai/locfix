"""End-to-end GPU check of the training path on E4B (Colab A100/H100 or a training node).

    dequantize -ct -> parity (base, random adapter) -> overfit LoRA on fixture records -> export
    -> parity (trained adapter) -> vLLM generation with the adapter reproduces a trained turn

Spec 01 and 05 acceptance tests. Needs: vllm==0.19.1, transformers==5.13.1, peft, accelerate,
compressed-tensors, and the -ct model (a local folder, or downloaded with kagglehub).

    python notebooks/train_smoke.py --ct /path/to/gemma-4-e4b-it-qat-w4a16-ct  [--work /content/smoke]
    python notebooks/train_smoke.py --kaggle google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2

Writes <work>/smoke_report.json; send it back (and the parity folders if a check fails).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PYTHONPATH", str(REPO / "src"))


def step(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def overfit_records(out: Path, n: int = 8, max_tokens: int = 12000) -> Path:
    """n short fixture records (single agent + localizer + fixer) as one training file."""
    from locfix.render.chat import load_tokenizer  # noqa: F401  (import check)

    recs = []
    for f in ("s1_fixtures.jsonl", "locfix_fixtures.jsonl"):
        for line in open(REPO / "tests" / "fixtures" / "records" / f):
            recs.append(line)
    recs.sort(key=len)
    out.write_text("".join(recs[:n]))
    return out


def generation_check(ct: str, adapter: str, record_file: Path, tok_dir: str, text_only: bool = False,
                     max_new: int = 48) -> dict:
    """Greedy generation from the prompt of each record's first trained turn; share of the trained
    target tokens reproduced (an overfit adapter should reproduce most of them)."""
    code = f"""
import json
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
import sys; sys.path.insert(0, {str(REPO / 'src')!r})
from locfix.render.chat import load_tokenizer, render_record
tok = load_tokenizer({tok_dir!r})
items = []
for line in open({str(record_file)!r}):
    r = render_record(json.loads(line), tok)
    s = next(i for i, l in enumerate(r.labels) if l != -100)
    e = s
    while e < len(r.labels) and r.labels[e] != -100:
        e += 1
    items.append((r.input_ids[:s], r.input_ids[s:min(e, s + {max_new})]))
r = json.load(open({str(Path(adapter) / 'adapter_config.json')!r}))["r"]
extra = {{"hf_overrides": {{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}}}} if {text_only!r} else {{}}
llm = LLM(model={ct!r}, max_model_len=32768, enable_lora=True, max_lora_rank=r, max_loras=1,
          gpu_memory_utilization=0.5, max_num_seqs=1, enforce_eager=True, **extra)
outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p, _ in items],
                    SamplingParams(max_tokens={max_new}, temperature=0.0, skip_special_tokens=False),
                    lora_request=LoRARequest("a", 1, {adapter!r}))
res = []
for (p, t), o in zip(items, outs):
    g = list(o.outputs[0].token_ids)[:len(t)]
    same = sum(1 for a, b in zip(g, t) if a == b)
    res.append({{"target": len(t), "matched": same, "prefix_ok": g[:8] == t[:8],
                 "target_text": tok.decode(t), "generated": tok.decode(g)}})
print("GEN_RESULT " + json.dumps(res))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    line = next((x for x in r.stdout.splitlines() if x.startswith("GEN_RESULT ")), None)
    if line is None:
        return {"error": r.stderr[-3000:]}
    res = json.loads(line[len("GEN_RESULT "):])
    tot = sum(x["target"] for x in res)
    return {"records": len(res), "token_match": round(sum(x["matched"] for x in res) / max(tot, 1), 3),
            "first8_ok": sum(x["prefix_ok"] for x in res), "examples": res[:3]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--ct", help="local -w4a16-ct model folder")
    src.add_argument("--kaggle", help="kagglehub model handle to download")
    ap.add_argument("--work", default="/content/smoke" if Path("/content").exists() else "work/smoke")
    ap.add_argument("--skip-parity", action="store_true", help="no vLLM (train/export only)")
    ap.add_argument("--vllm-text-only", action="store_true",
                    help="vLLM through Gemma4ForCausalLM (approximate; when the competition vLLM wheel is unavailable)")
    ap.add_argument("--steps-epochs", type=int, default=12, help="epochs of the overfit run")
    a, _ = ap.parse_known_args(argv)
    if not (a.ct or a.kaggle):
        print("give --ct <folder> or --kaggle <handle>")
        return
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    ct = a.ct
    if a.kaggle:
        import kagglehub

        step("download")
        ct = kagglehub.model_download(a.kaggle)
    report["ct"] = ct
    if not a.skip_parity:
        from locfix.quant.parity import vllm_lora_support

        sup = vllm_lora_support()
        report["vllm_lora_support"] = sup
        print("vLLM LoRA support:", sup, flush=True)
        if not sup["Gemma4ForConditionalGeneration"] and not a.vllm_text_only:
            raise SystemExit(
                "This vLLM build cannot serve LoRA on Gemma4ForConditionalGeneration (the class the harness loads):\n"
                "it is the public PyPI vllm 0.19.1. Install the competition's wheel on top, then rerun:\n"
                "  pip install --no-deps --force-reinstall /path/to/gemma-4-developer-agent-wheelhouse/vllm-*.whl\n"
                "or rerun with --vllm-text-only for an approximate check (or --skip-parity).")

    from locfix.quant.dequantize import dequantize
    from locfix.quant.parity import run_parity
    from locfix.train.data import Source
    from locfix.train.sft import TrainSettings, run_training

    bf16 = work / "bf16"
    step("dequantize")
    if not (bf16 / "dequantize_report.json").exists():
        report["dequantize"] = {k: v for k, v in dequantize(ct, bf16).items() if k != "grid"}
    else:
        report["dequantize"] = "reused"
    recs = [str(REPO / "tests" / "fixtures" / "records" / f) for f in ("s1_fixtures.jsonl", "locfix_fixtures.jsonl")]
    if not a.skip_parity:
        step("parity: base only (mode 1)")
        report["parity_base"] = run_parity(str(bf16), ct, recs, str(work / "parity_base"), n=8,
                                           vllm_text_only=a.vllm_text_only)["base"]
        step("parity: random adapter (mode 2)")
        rp = run_parity(str(bf16), ct, recs, str(work / "parity_random"), random_rank=64, n=8,
                        vllm_text_only=a.vllm_text_only)
        report["parity_random"] = {k: rp[k] for k in ("base", "adapter", "effect_hf", "effect_vllm")}
        report["parity_random"]["base"].pop("per_record", None)

    step("overfit training")
    train_file = overfit_records(work / "overfit.jsonl")
    s = TrainSettings(output_dir=str(work / "run"), base_model=str(bf16), sources=[Source(str(train_file))],
                      val_sources=[Source(str(train_file))], epochs=a.steps_epochs, learning_rate=5e-4,
                      lora_r=16, lora_alpha=32, global_batch_rows=4, warmup_steps=2,
                      cache_dir=str(work / "token_cache"), serve_id="gemma-4-e4b-it-qat-w4a16-ct")
    out = run_training(s)
    report["train"] = {"metrics": out["train"], "plan": out["plan"], "export": out["exports"]["final"]}
    vals = [json.loads(x) for x in (work / "run" / "val.jsonl").read_text().splitlines()]
    report["val_nll_first_last"] = [vals[0]["val_nll/all"], vals[-1]["val_nll/all"]]
    adapter = str(work / "run" / "export" / "final")

    if not a.skip_parity:
        step("parity: trained adapter (mode 3)")
        rp = run_parity(str(bf16), ct, recs, str(work / "parity_trained"), adapter=adapter, n=8,
                        vllm_text_only=a.vllm_text_only)
        report["parity_trained"] = {k: rp[k] for k in ("adapter", "effect_hf", "effect_vllm")}
        step("generation with the adapter in vLLM")
        report["generation"] = generation_check(ct, adapter, train_file, str(bf16), a.vllm_text_only)
    (work / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str)[:8000])
    print(f"\nreport: {work / 'smoke_report.json'}")


if __name__ == "__main__":
    main()
