"""`locfix quant.parity`: does training see what vLLM serves? (spec 01 §3)

Per-token log-probs of the *trained* (assistant) tokens of a few records are computed twice:

- **train side (HF):** the dequantized bf16 text-only base (+ adapter with PEFT), the same forward and
  the same softcapped logits the trainer uses;
- **serve side (vLLM 0.19.1):** the real `-w4a16-ct` checkpoint (+ adapter via `LoRARequest`), flags
  as in the harness, `prompt_logprobs` on the same token ids.

Modes: no adapter (proves the dequantization), `--random-adapter R` (a random non-zero LoRA run
through `export_adapter`, proves vLLM applies every module and the k_eq_v fix), `--adapter DIR`
(a trained, exported adapter: run before every evaluation). Each side runs in its own subprocess so
GPU memory is released between them. Output: `parity.json` with mean/max |Δ| per record, the
share of tokens with |Δ| > 0.1, and the adapter's effect size on each side.

    locfix quant.parity --bf16 work/models/e4b-dequant --ct /kaggle/input/.../2 \
        --records tests/fixtures/records/s1_fixtures.jsonl --random-adapter 64 --out work/parity/e4b-random
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- inputs
def prepare_items(record_files: list[str], tokenizer_dir: str, n: int, max_tokens: int) -> list[dict[str, Any]]:
    """Up to n records (shortest first above 1K tokens, then longer): token ids and trained positions."""
    from ..render.chat import load_tokenizer, render_record

    tok = load_tokenizer(tokenizer_dir)
    items = []
    for f in record_files:
        for line in open(f):
            rec = json.loads(line)
            r = render_record(rec, tok)
            if len(r.input_ids) > max_tokens:
                continue
            pos = [t for t, lab in enumerate(r.labels) if lab != -100]
            items.append({"id": rec["id"], "input_ids": list(map(int, r.input_ids)), "positions": pos})
    items.sort(key=lambda x: len(x["input_ids"]))
    return items[:n]


def random_adapter(bf16_dir: str, out_dir: str, r: int, alpha: int, scale: float = 0.02, seed: int = 0) -> dict:
    """PEFT LoRA with random non-zero A and B on every target module, saved like a trained
    adapter and then passed through `export_adapter` (adds the k_eq_v v_proj copies)."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    from ..train.export import export_adapter
    from ..train.model import LORA_TARGETS

    cfg = AutoConfig.from_pretrained(bf16_dir)
    with torch.device("meta"):
        model = Gemma4ForCausalLM(cfg)
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in LORA_TARGETS or not isinstance(mod, torch.nn.Linear) or ".layers." not in name:
            continue
        tensors[f"base_model.model.{name}.lora_A.weight"] = torch.randn(r, mod.in_features, generator=g) * scale
        tensors[f"base_model.model.{name}.lora_B.weight"] = torch.randn(mod.out_features, r, generator=g) * scale
    raw = Path(out_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(raw / "adapter_model.safetensors"))
    (raw / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": alpha, "lora_dropout": 0.0,
        "bias": "none", "use_rslora": False, "use_dora": False, "fan_in_fan_out": False,
        "target_modules": list(LORA_TARGETS), "base_model_name_or_path": bf16_dir}))
    return export_adapter(raw, Path(out_dir) / "export", cfg)


# --------------------------------------------------------------------------- sides
def hf_side(bf16_dir: str, adapter_dir: str | None, items_path: str, out_path: str, chunk: int = 1024) -> None:
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Gemma4ForCausalLM.from_pretrained(bf16_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
                                              device_map=dev)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir).get_base_model()
    model.eval()
    softcap = model.config.final_logit_softcapping
    items = json.loads(Path(items_path).read_text())
    out = {}
    with torch.no_grad():
        for it in items:
            ids = torch.tensor([it["input_ids"]], device=dev)
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            pos = torch.tensor(it["positions"], device=dev)
            lps = []
            for s in range(0, len(pos), chunk):
                p = pos[s:s + chunk]
                logits = (hidden[p - 1] @ model.lm_head.weight.t()).float()
                if softcap is not None:
                    logits = torch.tanh(logits / softcap) * softcap
                lp = torch.log_softmax(logits, -1).gather(-1, ids[0, p].unsqueeze(-1)).squeeze(-1)
                lps.append(lp.cpu())
            out[it["id"]] = torch.cat(lps).tolist()
    Path(out_path).write_text(json.dumps(out))


# prompt_logprobs takes a float32 log-softmax over the 262K vocabulary for every scheduled prompt
# token: 16,384 tokens (vLLM's default chunk) need ~17 GB on top of the KV cache. Parity needs little
# KV cache, so: half the GPU, one prompt at a time, 4,096-token prefill chunks (~4 GB of logits).
VLLM_PARITY_KW = {"gpu_memory_utilization": 0.5, "max_num_seqs": 1, "max_num_batched_tokens": 4096,
                  "enable_chunked_prefill": True, "enable_prefix_caching": False}


def vllm_side(ct_dir: str, adapter_dir: str | None, items_path: str, out_path: str, max_lora_rank: int) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw: dict[str, Any] = {"model": ct_dir, "max_model_len": 32768, "enforce_eager": True, **VLLM_PARITY_KW}
    if adapter_dir:
        kw.update(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1)
    llm = LLM(**kw)
    lora = None
    if adapter_dir:
        from vllm.lora.request import LoRARequest

        lora = LoRARequest("adapter", 1, adapter_dir)
    items = json.loads(Path(items_path).read_text())
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=it["input_ids"]) for it in items], sp, lora_request=lora)
    res = {}
    for it, o in zip(items, outs, strict=True):
        plp = o.prompt_logprobs
        res[it["id"]] = [float(plp[t][it["input_ids"][t]].logprob) for t in it["positions"]]
    Path(out_path).write_text(json.dumps(res))


# --------------------------------------------------------------------------- compare
def compare(a: dict[str, list[float]], b: dict[str, list[float]]) -> dict[str, Any]:
    per, all_d = {}, []
    for k in a:
        d = np.abs(np.asarray(a[k]) - np.asarray(b[k]))
        all_d.append(d)
        per[k] = {"tokens": int(d.size), "mean": round(float(d.mean()), 5), "max": round(float(d.max()), 4),
                  "frac_gt_0.1": round(float((d > 0.1).mean()), 5)}
    d = np.concatenate(all_d) if all_d else np.zeros(0)
    return {"tokens": int(d.size), "mean": round(float(d.mean()), 5) if d.size else None,
            "max": round(float(d.max()), 4) if d.size else None,
            "frac_gt_0.1": round(float((d > 0.1).mean()), 5) if d.size else None, "per_record": per}


def _run_side(side: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", "locfix.quant.parity", "--side", side, *args]
    env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn", PYTORCH_ALLOC_CONF="expandable_segments:True")
    subprocess.run(cmd, check=True, env=env)


def run_parity(bf16: str, ct: str, records: list[str], out: str, adapter: str | None = None,
               random_rank: int | None = None, n: int = 12, max_tokens: int = 32000,
               threshold: float | None = None) -> dict[str, Any]:
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    items = prepare_items(records, bf16, n, max_tokens)
    (o / "items.json").write_text(json.dumps(items))
    report: dict[str, Any] = {"bf16": bf16, "ct": ct, "records": records, "n_items": len(items)}
    rank = None
    if random_rank:
        report["random_adapter"] = random_adapter(bf16, str(o / "random_adapter"), random_rank, 2 * random_rank)
        adapter = str(o / "random_adapter" / "export")
    if adapter:
        rank = int(json.loads((Path(adapter) / "adapter_config.json").read_text())["r"])
    for side in ("hf", "vllm"):
        _run_side(side, ["--bf16", bf16, "--ct", ct, "--items", str(o / "items.json"),
                         "--side-out", str(o / f"{side}_base.json")])
        if adapter:
            _run_side(side, ["--bf16", bf16, "--ct", ct, "--items", str(o / "items.json"), "--adapter", adapter,
                             "--max-lora-rank", str(rank), "--side-out", str(o / f"{side}_adapter.json")])
    load = lambda name: json.loads((o / name).read_text())  # noqa: E731
    report["base"] = compare(load("hf_base.json"), load("vllm_base.json"))
    if adapter:
        report["adapter"] = compare(load("hf_adapter.json"), load("vllm_adapter.json"))
        report["effect_hf"] = compare(load("hf_adapter.json"), load("hf_base.json"))
        report["effect_vllm"] = compare(load("vllm_adapter.json"), load("vllm_base.json"))
        for k in ("adapter", "effect_hf", "effect_vllm"):
            report[k].pop("per_record")
    if threshold is not None:
        ok = report["base"]["mean"] <= threshold
        if adapter:
            ok = ok and report["adapter"]["mean"] <= threshold and \
                report["effect_vllm"]["mean"] >= 10 * report["base"]["mean"]
        report["threshold"] = threshold
        report["ok"] = bool(ok)
    (o / "parity.json").write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- CLI
def add_args_cmd_parity(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bf16", required=True, help="dequantized text-only checkpoint (training side)")
    p.add_argument("--ct", required=True, help="the -w4a16-ct checkpoint (serving side)")
    p.add_argument("--records", nargs="+", required=True, help="record JSONL files")
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--adapter", help="exported adapter folder (mode 3)")
    g.add_argument("--random-adapter", type=int, metavar="RANK", help="random adapter of this rank (mode 2)")
    p.add_argument("--n", type=int, default=12, help="number of records")
    p.add_argument("--threshold", type=float, help="max mean |Δ logp| accepted (set from the first run)")


def cmd_parity(args: argparse.Namespace, cfg: dict) -> int:
    rep = run_parity(args.bf16, args.ct, args.records, args.out, args.adapter, args.random_adapter, args.n,
                     threshold=args.threshold)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("records",)}, indent=2)[:6000])
    return 0 if rep.get("ok", True) else 1


def _side_main(argv: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=["hf", "vllm"], required=True)
    p.add_argument("--bf16")
    p.add_argument("--ct")
    p.add_argument("--items", required=True)
    p.add_argument("--adapter")
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--side-out", required=True)
    a = p.parse_args(argv)
    if a.side == "hf":
        hf_side(a.bf16, a.adapter, a.items, a.side_out)
    else:
        vllm_side(a.ct, a.adapter, a.items, a.side_out, a.max_lora_rank)


if __name__ == "__main__":
    _side_main(sys.argv[1:])
