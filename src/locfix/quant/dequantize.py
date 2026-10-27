"""`locfix quant dequantize`: the `-w4a16-ct` checkpoint → a bf16 text-only checkpoint for training.

Every int4 linear layer is decompressed with compressed-tensors' own
`PackedQuantizationCompressor.decompress` (the reference implementation of the format), so the bf16
weights are exactly int4 × group scale: the values vLLM's kernels multiply with. Unquantized tensors
(embeddings, norms, `lm_head` if stored) are copied bit for bit. The output is the text-only model
(`Gemma4ForCausalLM`, `model_type: gemma4_text`): language-model keys are renamed
`model.language_model.X → model.X` and the vision/audio towers are left out (training never feeds
images or audio, and adapter keys then have the layout verified on Kaggle).

Grid check: for sampled layers, weight ÷ group scale must be an integer in [-8, 7] up to the bf16
rounding of the product (bf16 keeps 8 significant bits: |q|·2⁻⁸ ≤ 0.031); a wrong unpacking or
scale layout gives deviations near 0.5. The report gives the maximum deviation.

    locfix quant.dequantize /kaggle/input/.../gemma-4-e4b-it-qat-w4a16-ct/2 --out work/models/gemma4-e4b-dequant
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

QUANT_SUFFIXES = ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point", "weight_g_idx")
DROP_PREFIXES = ("model.vision_tower.", "model.audio_tower.", "model.embed_vision.", "model.embed_audio.")
GRID_TOL = 8 * 2 ** -8                    # |q| ≤ 8, bf16 rounding of q·s
COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
              "special_tokens_map.json", "processor_config.json")


class DequantError(ValueError):
    pass


def _scheme(qcfg: dict[str, Any]):
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    if qcfg.get("format") != "pack-quantized":
        raise DequantError(f"unsupported format {qcfg.get('format')!r}")
    groups = qcfg["config_groups"]
    if len(groups) != 1:
        raise DequantError(f"expected one config group, found {list(groups)}")
    g = next(iter(groups.values()))
    w = g["weights"]
    if not (w["num_bits"] == 4 and w["type"] == "int" and w["symmetric"] and w["strategy"] == "group"):
        raise DequantError(f"expected symmetric int4 group quantization, got {w}")
    args = QuantizationArgs(**{k: v for k, v in w.items() if v is not None})
    return QuantizationScheme(targets=g["targets"], weights=args), int(w["group_size"])


def _shards(src: Path) -> list[Path]:
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        files = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        return [src / f for f in files]
    single = src / "model.safetensors"
    if single.exists():
        return [single]
    raise DequantError(f"no safetensors weights in {src}")


def _out_name(key: str, text_only: bool) -> str | None:
    if not text_only:
        return key
    if key.startswith(DROP_PREFIXES):
        return None
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    if key == "lm_head.weight":
        return key
    raise DequantError(f"unexpected key {key!r} (not language model, not a known tower)")


class _ShardWriter:
    def __init__(self, dst: Path, max_bytes: int):
        self.dst, self.max_bytes = dst, max_bytes
        self.buf: dict[str, torch.Tensor] = {}
        self.size = 0
        self.files: list[tuple[str, list[str]]] = []
        self.total = 0

    def add(self, name: str, t: torch.Tensor) -> None:
        n = t.numel() * t.element_size()
        if self.buf and self.size + n > self.max_bytes:
            self.flush()
        self.buf[name] = t.contiguous()
        self.size += n
        self.total += n

    def flush(self) -> None:
        if not self.buf:
            return
        fname = f"model-{len(self.files) + 1:05d}.safetensors"
        save_file(self.buf, str(self.dst / fname), metadata={"format": "pt"})
        self.files.append((fname, list(self.buf)))
        self.buf, self.size = {}, 0

    def finish(self) -> None:
        self.flush()
        n = len(self.files)
        weight_map = {}
        for i, (fname, keys) in enumerate(self.files):
            final = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            (self.dst / fname).rename(self.dst / final)
            weight_map.update(dict.fromkeys(keys, final))
        (self.dst / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": self.total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))


def grid_deviation(weight: torch.Tensor, scale: torch.Tensor, group: int) -> tuple[float, float, float]:
    """(max |w/s − round(w/s)|, min round, max round) for one layer."""
    s = scale.float().repeat_interleave(group, dim=1)[:, : weight.shape[1]]
    q = weight.float() / s
    r = torch.round(q)
    return float((q - r).abs().max()), float(r.min()), float(r.max())


def dequantize(src: str | Path, dst: str | Path, text_only: bool = True, max_shard_bytes: int = 4 << 30,
               grid_samples: int = 16, seed: int = 0) -> dict[str, Any]:
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor

    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "config.json").read_text())
    qcfg = cfg.get("quantization_config")
    if not qcfg:
        raise DequantError(f"{src} has no quantization_config")
    scheme, group = _scheme(qcfg)
    dst.mkdir(parents=True, exist_ok=True)
    writer = _ShardWriter(dst, max_shard_bytes)
    rng = random.Random(seed)
    report: dict[str, Any] = {"source": str(src), "quantized_layers": 0, "copied_tensors": 0, "dropped_tensors": 0,
                              "grid": []}
    tied = None
    for shard in _shards(src):
        with safe_open(str(shard), framework="pt") as f:
            keys = list(f.keys())
            modules: dict[str, dict[str, torch.Tensor]] = {}
            plain: list[str] = []
            for k in keys:
                head, _, tail = k.rpartition(".")
                if tail in QUANT_SUFFIXES:
                    modules.setdefault(head, {})[tail] = f.get_tensor(k)
                else:
                    plain.append(k)
            for k in plain:
                name = _out_name(k, text_only)
                if name is None:
                    report["dropped_tensors"] += 1
                    continue
                t = f.get_tensor(k)
                if name == "lm_head.weight" and text_only:
                    tied = t
                    continue
                writer.add(name, t)
                report["copied_tensors"] += 1
            for prefix, parts in modules.items():
                if "weight_packed" not in parts or "weight_scale" not in parts:
                    raise DequantError(f"{prefix}: incomplete quantized tensors {sorted(parts)}")
                name = _out_name(prefix + ".weight", text_only)
                if name is None:
                    report["dropped_tensors"] += len(parts)
                    continue
                dec = PackedQuantizationCompressor.decompress(dict(parts), scheme)
                w = dec["weight"].to(torch.bfloat16)
                if rng.random() < grid_samples / 400 or len(report["grid"]) < 2:
                    dev, lo, hi = grid_deviation(w, parts["weight_scale"], group)
                    report["grid"].append({"layer": name, "max_dev": round(dev, 5), "min_q": lo, "max_q": hi})
                writer.add(name, w)
                report["quantized_layers"] += 1
    writer.finish()

    tc = cfg.get("text_config", cfg) if text_only else cfg
    tie = cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", True))
    if tied is not None and not tie:
        raise DequantError("lm_head stored and embeddings not tied: text-only export would drop it")
    if text_only:
        out_cfg = dict(tc)
        out_cfg["architectures"] = ["Gemma4ForCausalLM"]
        out_cfg["model_type"] = "gemma4_text"
        out_cfg["tie_word_embeddings"] = tie
    else:
        out_cfg = dict(cfg)
    out_cfg.pop("quantization_config", None)
    out_cfg["dtype"] = "bfloat16"
    out_cfg["locfix_dequantized_from"] = {"path": str(src), "quantization_config": qcfg}
    (dst / "config.json").write_text(json.dumps(out_cfg, indent=2))
    for fname in COPY_FILES:
        if (src / fname).exists():
            shutil.copy(src / fname, dst / fname)
    devs = [g["max_dev"] for g in report["grid"]]
    report["grid_max_dev"] = max(devs) if devs else None
    report["grid_ok"] = bool(devs) and max(devs) <= GRID_TOL and all(-8 <= g["min_q"] and g["max_q"] <= 7
                                                              for g in report["grid"])
    report["output"] = str(dst)
    (dst / "dequantize_report.json").write_text(json.dumps(report, indent=2))
    if not report["grid_ok"]:
        raise DequantError(f"grid check failed: {report['grid'][:5]}")
    return report


# --------------------------------------------------------------------------- CLI
def add_args_cmd_dequantize(p: argparse.ArgumentParser) -> None:
    p.add_argument("src", help="folder of the -w4a16-ct checkpoint (config.json + safetensors)")
    p.add_argument("--out", required=True, help="output folder (bf16, text-only)")
    p.add_argument("--keep-towers", action="store_true", help="keep vision/audio and the multimodal layout")
    p.add_argument("--grid-samples", type=int, default=16)


def cmd_dequantize(args: argparse.Namespace, cfg: dict) -> int:
    rep = dequantize(args.src, args.out, text_only=not args.keep_towers, grid_samples=args.grid_samples)
    print(json.dumps({k: v for k, v in rep.items() if k != "grid"}, indent=2))
    return 0
