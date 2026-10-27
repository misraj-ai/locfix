import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from tiny_gemma4 import tiny_model

QCFG = {"config_groups": {"group_0": {"format": "pack-quantized", "input_activations": None, "output_activations": None,
                                      "targets": ["Linear"],
                                      "weights": {"actorder": None, "block_structure": None, "dynamic": False,
                                                  "group_size": 32, "num_bits": 4, "observer": "memoryless_minmax",
                                                  "observer_kwargs": {}, "scale_dtype": None, "strategy": "group",
                                                  "symmetric": True, "type": "int", "zp_dtype": None}}},
        "format": "pack-quantized", "ignore": ["lm_head"], "quant_method": "compressed-tensors",
        "quantization_status": "compressed"}


def make_ct_checkpoint(path, kind="e4b"):
    """A tiny checkpoint in the -w4a16-ct layout: multimodal key names, int4 language linears
    compressed by compressed-tensors itself, unquantized embeddings/norms and a vision tensor."""
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor

    from locfix.quant.dequantize import _scheme

    scheme, group = _scheme(QCFG)
    model = tiny_model(kind).to(torch.bfloat16)
    linear = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.startswith("model.layers.")}
    out, expected = {}, {}
    for k, v in model.state_dict().items():
        if k == "lm_head.weight":
            continue                                           # tied to the embeddings
        mk = "model.language_model." + k[len("model."):]
        mod = k.rsplit(".", 1)[0]
        if mod in linear and k.endswith(".weight") and v.shape[1] % group == 0:
            g = v.float().reshape(v.shape[0], -1, group)
            scale = (g.abs().amax(-1) / 7).clamp(min=1e-6).to(torch.bfloat16)
            comp = PackedQuantizationCompressor.compress({"weight": v, "weight_scale": scale}, scheme)
            for part, t in comp.items():
                out[mk.rsplit(".", 1)[0] + "." + part] = t
            dec = PackedQuantizationCompressor.decompress(dict(comp), scheme)["weight"].to(torch.bfloat16)
            expected[k] = dec
        else:
            out[mk] = v.contiguous()
            expected[k] = v
    out["model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"] = torch.randn(8, 8)
    out["model.embed_vision.embedding_projection.weight"] = torch.randn(8, 8)
    path.mkdir(parents=True, exist_ok=True)
    save_file(out, str(path / "model.safetensors"))
    cfg = {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
           "text_config": model.config.to_dict(), "tie_word_embeddings": True, "quantization_config": QCFG}
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "tokenizer.json").write_text("{}")
    return model, expected


@pytest.mark.parametrize("kind", ["e4b", "31b"])
def test_dequantize_tiny_checkpoint(tmp_path, kind):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    from locfix.quant.dequantize import dequantize

    model, expected = make_ct_checkpoint(tmp_path / "ct", kind)
    rep = dequantize(tmp_path / "ct", tmp_path / "bf16")
    assert rep["grid_ok"] and rep["quantized_layers"] > 0 and rep["dropped_tensors"] == 2
    cfg = json.loads((tmp_path / "bf16" / "config.json").read_text())
    assert cfg["model_type"] == "gemma4_text" and "quantization_config" not in cfg
    assert cfg["architectures"] == ["Gemma4ForCausalLM"] and (tmp_path / "bf16" / "tokenizer.json").exists()
    got = {}
    for f in (tmp_path / "bf16").glob("model-*.safetensors"):
        got.update(load_file(str(f)))
    assert set(got) == set(expected)
    for k, v in expected.items():
        assert torch.equal(got[k], v), k                       # exact: library decompression / bit copy
    loaded = Gemma4ForCausalLM.from_pretrained(tmp_path / "bf16", dtype=torch.bfloat16)
    ref = tiny_model(kind)                                     # same loading path for the reference
    ref.load_state_dict({**expected, "lm_head.weight": expected["model.embed_tokens.weight"]})
    ref.save_pretrained(tmp_path / "ref")
    ref = Gemma4ForCausalLM.from_pretrained(tmp_path / "ref", dtype=torch.bfloat16)
    x = torch.randint(3, 500, (1, 20))
    with torch.no_grad():
        assert torch.equal(loaded(input_ids=x).logits, ref(input_ids=x).logits)


def test_grid_deviation_detects_off_grid_weights():
    from locfix.quant.dequantize import grid_deviation

    scale = torch.full((4, 2), 0.1, dtype=torch.bfloat16)
    q = torch.randint(-8, 8, (4, 64)).float()
    w = (q * 0.1).to(torch.bfloat16)
    dev, lo, hi = grid_deviation(w, scale, 32)
    assert dev <= 8 * 2 ** -8 and lo >= -8 and hi <= 7
    dev2, _, _ = grid_deviation((q * 0.1 + 0.04).to(torch.bfloat16), scale, 32)
    assert dev2 > 0.3


def test_refuses_unknown_keys(tmp_path):
    from locfix.quant.dequantize import DequantError, dequantize

    make_ct_checkpoint(tmp_path / "ct")
    extra = load_file(str(tmp_path / "ct" / "model.safetensors"))
    extra["model.mystery_head.weight"] = torch.zeros(2, 2)
    save_file(extra, str(tmp_path / "ct" / "model.safetensors"))
    with pytest.raises(DequantError, match="unexpected key"):
        dequantize(tmp_path / "ct", tmp_path / "out")
