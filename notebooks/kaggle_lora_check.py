"""Does vLLM (as the competition harness starts it) load a LoRA adapter on a Gemma 4 -w4a16-ct model?

Run on a Kaggle GPU notebook (L4x4, internet off) with the competition model attached and
vLLM 0.19.1 installed from the competition wheelhouse. No model download needed: the dummy adapter
is built from config.json alone (meta-device model, random LoRA tensors).

    python kaggle_lora_check.py --model-dir /kaggle/input/gemma-4/transformers/gemma-4-e4b-it-qat-w4a16-ct/1
    python kaggle_lora_check.py --model-dir <31b dir> --mode text-only     # also try the 12B-style override

Modes:
  harness    flags exactly as the harness (12B gets the text-only override, others none)
  text-only  force --hf-overrides to Gemma4ForCausalLM (what the harness does for 12B only)

Writes /kaggle/working/lora_check_<model>_<mode>.json with: architectures, k_eq_v layers,
whether vLLM started, whether the adapter request worked, and whether the adapter changed outputs.

Self-test on CPU (no vLLM, tiny config): python kaggle_lora_check.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TEXT_ONLY_OVERRIDE = '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'
PORT = 8011


# --------------------------------------------------------------------------- adapter
def text_config_of(cfg):
    return getattr(cfg, "text_config", None) or cfg


def build_dummy_adapter(model_dir: str, out_dir: str, r: int = 8, alpha: int = 16,
                        scale: float = 0.02, kv_copy: bool = True, seed: int = 0) -> dict:
    """PEFT-format LoRA with random non-zero A and B on all language-model target modules."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    cfg = AutoConfig.from_pretrained(model_dir)
    tcfg = text_config_of(cfg)
    with torch.device("meta"):
        model = Gemma4ForCausalLM(tcfg)

    g = torch.Generator().manual_seed(seed)
    tensors, k_eq_v_layers, n_modules = {}, [], 0
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in TARGETS or not isinstance(mod, torch.nn.Linear):
            continue
        a = (torch.randn(r, mod.in_features, generator=g) * scale).to(torch.bfloat16)
        b = (torch.randn(mod.out_features, r, generator=g) * scale).to(torch.bfloat16)
        tensors[f"base_model.model.{name}.lora_A.weight"] = a
        tensors[f"base_model.model.{name}.lora_B.weight"] = b
        n_modules += 1
        # k_eq_v: HF computes V from k_proj (no v_proj module); vLLM copies the k_proj base weight
        # into the V slot only, so the adapter needs an explicit v_proj LoRA equal to k_proj's.
        if leaf == "k_proj":
            attn = model.get_submodule(name.rsplit(".", 1)[0])
            if getattr(attn, "v_proj", "missing") is None:
                k_eq_v_layers.append(name.split(".layers.")[1].split(".")[0])
                if kv_copy:
                    vname = name[: -len("k_proj")] + "v_proj"
                    tensors[f"base_model.model.{vname}.lora_A.weight"] = a.clone()
                    tensors[f"base_model.model.{vname}.lora_B.weight"] = b.clone()

    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    adapter_cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": alpha,
        "lora_dropout": 0.0, "bias": "none", "use_rslora": False, "fan_in_fan_out": False,
        "target_modules": list(TARGETS), "base_model_name_or_path": model_dir,
    }
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_cfg, f, indent=2)
    return {
        "architectures": getattr(cfg, "architectures", None),
        "attention_k_eq_v": getattr(tcfg, "attention_k_eq_v", None),
        "k_eq_v_layers": [int(x) for x in k_eq_v_layers],
        "num_kv_shared_layers": getattr(tcfg, "num_kv_shared_layers", None),
        "num_hidden_layers": getattr(tcfg, "num_hidden_layers", None),
        "lora_modules": n_modules,
        "lora_tensors": len(tensors),
    }


# --------------------------------------------------------------------------- vLLM
def http_json(url: str, payload: dict | None = None, timeout: float = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if body else {}


def wait_ready(proc: subprocess.Popen, timeout_s: int) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def vllm_cmd(model_dir: str, adapter_dir: str, r: int, tp: int, mode: str, is_12b: bool) -> list[str]:
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_dir, "--served-model-name", "base", "--port", str(PORT),
           "--tensor-parallel-size", str(tp), "--gpu-memory-utilization", "0.90",
           "--max-model-len", "32768",
           "--enable-auto-tool-choice", "--tool-call-parser", "gemma4", "--reasoning-parser", "gemma4",
           "--enable-lora", "--max-loras", "1", "--max-lora-rank", str(r),
           "--lora-modules", f"dummy={adapter_dir}"]
    if mode == "text-only" or (mode == "harness" and is_12b):
        cmd += ["--hf-overrides", TEXT_ONLY_OVERRIDE]
    return cmd


def first_token_logprobs(model: str) -> dict:
    out = http_json(f"http://127.0.0.1:{PORT}/v1/completions", {
        "model": model, "prompt": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
        "max_tokens": 4, "temperature": 0, "logprobs": 5})
    return out["choices"][0]["logprobs"]["top_logprobs"][0]


def run_check(args) -> dict:
    import torch
    name = Path(args.model_dir.rstrip("/")).parts[-2] if args.model_dir.rstrip("/").endswith("/1") else Path(args.model_dir.rstrip("/")).name
    is_12b = "12b" in args.model_dir.lower()
    report = {"model_dir": args.model_dir, "mode": args.mode}
    try:
        import vllm
        report["vllm_version"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        report["vllm_version"] = f"not importable: {e}"
    import transformers
    report["transformers_version"] = transformers.__version__

    adapter_dir = tempfile.mkdtemp(prefix="dummy_lora_")
    report["adapter"] = build_dummy_adapter(args.model_dir, adapter_dir, r=args.rank, kv_copy=not args.no_kv_copy)
    print(json.dumps(report, indent=2))

    tp = args.tp or max(1, torch.cuda.device_count())
    cmd = vllm_cmd(args.model_dir, adapter_dir, args.rank, tp, args.mode, is_12b)
    report["command"] = " ".join(cmd)
    log_path = f"/kaggle/working/vllm_{name}_{args.mode}.log" if os.path.isdir("/kaggle/working") else f"vllm_{name}_{args.mode}.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ok = wait_ready(proc, args.timeout)
        report["server_started"] = ok
        if ok:
            try:
                base = first_token_logprobs("base")
                lora = first_token_logprobs("dummy")
                report["base_top"] = base
                report["adapter_top"] = lora
                common = set(base) & set(lora)
                diff = max((abs(base[t] - lora[t]) for t in common), default=None)
                report["adapter_request_ok"] = True
                report["max_abs_logprob_diff"] = diff
                report["adapter_changes_output"] = bool(diff is not None and diff > 1e-3) or set(base) != set(lora)
            except Exception as e:  # noqa: BLE001
                report["adapter_request_ok"] = False
                report["adapter_error"] = repr(e)
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    tail = Path(log_path).read_text(errors="replace").splitlines()[-40:]
    report["log_tail"] = tail
    report["log_path"] = log_path
    report["does_not_support_lora"] = any("does not support LoRA" in line for line in tail)

    out = (Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".")) / f"lora_check_{name}_{args.mode}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "log_tail"}, indent=2))
    print("\n--- last log lines ---\n" + "\n".join(tail[-15:]))
    print(f"\nreport: {out}")
    return report


# --------------------------------------------------------------------------- self-test
def selftest() -> None:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from safetensors.torch import load_file

    tmp = tempfile.mkdtemp()
    cfg = Gemma4TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
        num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=16,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        hidden_size_per_layer_input=0, num_kv_shared_layers=0,
    )
    cfg.architectures = ["Gemma4ForCausalLM"]
    cfg.save_pretrained(tmp)
    out = os.path.join(tmp, "adapter")
    info = build_dummy_adapter(tmp, out)
    keys = load_file(os.path.join(out, "adapter_model.safetensors"))
    assert info["k_eq_v_layers"] == [5], info
    k = keys["base_model.model.model.layers.5.self_attn.k_proj.lora_B.weight"]
    v = keys["base_model.model.model.layers.5.self_attn.v_proj.lora_B.weight"]
    assert (k == v).all()
    assert "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight" in keys
    assert info["lora_modules"] == 6 * 7 - 1, info  # full-attention layer has no v_proj module
    print("selftest ok:", json.dumps(info))


def check(model_dir: str, mode: str = "harness", rank: int = 8, tp: int = 0, timeout: int = 1500,
          no_kv_copy: bool = False) -> dict:
    """Notebook entry point: `from kaggle_lora_check import check; check("<model dir or HF id>")`."""
    return run_check(argparse.Namespace(model_dir=model_dir, mode=mode, rank=rank, tp=tp,
                                        timeout=timeout, no_kv_copy=no_kv_copy))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", help="local checkpoint dir or Hugging Face id")
    p.add_argument("--mode", choices=["harness", "text-only"], default="harness")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--tp", type=int, default=0)
    p.add_argument("--timeout", type=int, default=1500)
    p.add_argument("--no-kv-copy", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a, unknown = p.parse_known_args(argv)        # tolerate Jupyter's "-f kernel.json"
    if unknown and not all(u.endswith(".json") or u == "-f" for u in unknown):
        p.error(f"unrecognized arguments: {' '.join(unknown)}")
    if a.selftest:
        selftest()
    elif not a.model_dir:
        print("Nothing to do. From a notebook cell call check(\"<model dir or HF id>\");"
              " from a shell run: python kaggle_lora_check.py --model-dir <dir> [--mode text-only]")
    else:
        run_check(a)


if __name__ == "__main__":
    main()
