"""Adapter export for vLLM: checks, the k_eq_v fix, bf16, PEFT layout.

Input: a PEFT adapter saved by the trainer (`adapter_config.json`, `adapter_model.safetensors`).
Output: the same adapter as vLLM 0.19.1 must load it:

- keys exactly `base_model.model.model.layers.<i>.(self_attn|mlp).<proj>.lora_(A|B).weight`, with
  `<proj>` in q/k/v/o/gate/up/down (the layout verified on Kaggle, `harness_facts.md` §13);
  anything else raises (a module vLLM does not load is a silent train/serve mismatch);
- **k_eq_v layers** (31B full-attention layers): HF has no `v_proj` there and computes V with
  `k_proj`, so the trained `k_proj` LoRA changes K and V; vLLM loads the `k_proj` weight into the K
  and V slots of its packed `qkv_proj`, and a `k_proj` LoRA only reaches K. A `v_proj` LoRA equal to
  the `k_proj` one is added for those layers;
- weights in bf16; `target_modules` as an explicit list; rank ≤ 128 (harness `max_lora_rank`).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

KEY = re.compile(r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\."
                 r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight$")
PARENT = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
          "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
MAX_RANK = 128


class ExportError(ValueError):
    pass


def k_eq_v_layers(text_config) -> list[int]:
    get = (lambda k, d=None: text_config.get(k, d)) if isinstance(text_config, dict) \
        else (lambda k, d=None: getattr(text_config, k, d))
    if not get("attention_k_eq_v", False):
        return []
    return [i for i, t in enumerate(get("layer_types") or []) if t == "full_attention"]


def export_adapter(adapter_dir: str | Path, out_dir: str | Path, text_config, serve_id: str | None = None,
                   dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    acfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    for flag in ("use_rslora", "use_dora"):
        if acfg.get(flag):
            raise ExportError(f"{flag} is set; only plain LoRA is served as trained")
    r = int(acfg["r"])
    if r > MAX_RANK:
        raise ExportError(f"rank {r} > max_lora_rank {MAX_RANK}")
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    found: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for k, v in tensors.items():
        m = KEY.match(k)
        if not m:
            raise ExportError(f"unexpected adapter key {k!r}")
        layer, parent, proj, ab = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        if PARENT[proj] != parent:
            raise ExportError(f"{proj} under {parent}: {k}")
        found.setdefault((layer, proj), {})[ab] = v
    for (layer, proj), d in found.items():
        if set(d) != {"A", "B"}:
            raise ExportError(f"layer {layer} {proj}: missing lora_{'A' if 'A' not in d else 'B'}")
        if d["A"].shape[0] != r or d["B"].shape[1] != r:
            raise ExportError(f"layer {layer} {proj}: rank mismatch {tuple(d['A'].shape)} / {tuple(d['B'].shape)}")

    kv = k_eq_v_layers(text_config)
    copied = []
    for layer in kv:
        if (layer, "v_proj") in found:
            raise ExportError(f"layer {layer} is k_eq_v but the adapter has a v_proj LoRA")
        if (layer, "k_proj") in found:
            found[(layer, "v_proj")] = {ab: t.clone() for ab, t in found[(layer, "k_proj")].items()}
            copied.append(layer)

    out: dict[str, torch.Tensor] = {}
    for (layer, proj), d in sorted(found.items()):
        for ab in ("A", "B"):
            out[f"base_model.model.model.layers.{layer}.{PARENT[proj]}.{proj}.lora_{ab}.weight"] = \
                d[ab].to(dtype).contiguous()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "adapter_model.safetensors"))
    acfg = dict(acfg)
    acfg["target_modules"] = sorted({proj for _, proj in found})
    if serve_id:
        acfg["base_model_name_or_path"] = serve_id
    (out_dir / "adapter_config.json").write_text(json.dumps(acfg, indent=2))
    for extra in ("README.md",):
        if (adapter_dir / extra).exists():
            shutil.copy(adapter_dir / extra, out_dir / extra)
    size = sum(p.stat().st_size for p in out_dir.iterdir())
    report = {"r": r, "lora_alpha": acfg.get("lora_alpha"), "modules": len(found), "tensors": len(out),
              "k_eq_v_layers_copied": copied, "target_modules": acfg["target_modules"], "bytes": size,
              "dtype": str(dtype).replace("torch.", "")}
    (out_dir / "export.json").write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- CLI
def finish_run(run_dir: str | Path, base_model: str | Path, serve_id: str | None = None) -> dict[str, Any]:
    """Export a training run's saved adapters (vLLM layout) after the run stopped before its own
    export: every `epoch-*` folder, and `final` (copied from the last epoch when it is missing:
    the last epoch's adapter IS the final adapter). Writes `result.json` if the run has none
    (`recovered: true`; no training metrics or val NLL), else `recovered.json`."""
    run = Path(run_dir)
    epochs = sorted(run.glob("epoch-*"), key=lambda p: int(p.name.split("-")[1]))
    if not epochs:
        raise ExportError(f"{run}: no epoch-* adapter folders")
    for d in epochs:
        if not (d / "adapter_model.safetensors").is_file():
            raise ExportError(f"{d}: no adapter_model.safetensors (the save did not finish)")
    final = run / "final"
    final_from = None
    if not (final / "adapter_model.safetensors").is_file():
        shutil.rmtree(final, ignore_errors=True)
        shutil.copytree(epochs[-1], final)
        final_from = epochs[-1].name
    cfg = json.loads((Path(base_model) / "config.json").read_text())
    text_config = cfg.get("text_config", cfg)
    out: dict[str, Any] = {"recovered": True, "final_from": final_from, "final_adapter": str(final), "exports": {}}
    for d in [*epochs, final]:
        out["exports"][d.name] = export_adapter(d, run / "export" / d.name, text_config, serve_id=serve_id)
    dest = run / ("recovered.json" if (run / "result.json").exists() else "result.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    out["written"] = str(dest)
    return out


def add_args_cmd_finish(p) -> None:
    p.add_argument("--run-dir", required=True, help="the run's output_dir (holds epoch-1, epoch-2, ...)")
    p.add_argument("--base-model", required=True, help="the dequantized bf16 base the run trained on (its config.json)")
    p.add_argument("--model", default="e4b", help="model key in the config (sets the served model id)")


def cmd_finish(args, cfg: dict) -> int:
    rep = finish_run(args.run_dir, args.base_model, serve_id=cfg["models"][args.model]["serve_id"])
    print(json.dumps({"final_from": rep["final_from"], "exports": sorted(rep["exports"]), "written": rep["written"]},
                     indent=2))
    return 0
