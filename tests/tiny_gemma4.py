"""Tiny random Gemma 4 text models for CPU tests (same code paths as the real 31B / E4B)."""
from __future__ import annotations

import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

ROPE = {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}


def tiny_config(kind: str = "31b", vocab: int = 512) -> Gemma4TextConfig:
    """`31b`: k_eq_v on full-attention layers. `e4b`: per-layer inputs and KV-shared layers."""
    layer_types = ["sliding_attention", "sliding_attention", "full_attention"] * 2
    common = dict(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
                  num_global_key_value_heads=1, sliding_window=8, layer_types=layer_types,
                  final_logit_softcapping=30.0, rope_parameters=ROPE, tie_word_embeddings=True,
                  max_position_embeddings=4096, hidden_activation="gelu_pytorch_tanh",
                  pad_token_id=0, bos_token_id=2, eos_token_id=1)
    if kind == "31b":
        return Gemma4TextConfig(**common, attention_k_eq_v=True, num_kv_shared_layers=0,
                                hidden_size_per_layer_input=0, vocab_size_per_layer_input=vocab)
    if kind == "e4b":
        return Gemma4TextConfig(**common, attention_k_eq_v=False, num_kv_shared_layers=2,
                                hidden_size_per_layer_input=8, vocab_size_per_layer_input=vocab)
    raise ValueError(kind)


def tiny_model(kind: str = "31b", seed: int = 0, dtype=torch.float32, vocab: int = 512) -> Gemma4ForCausalLM:
    torch.manual_seed(seed)
    m = Gemma4ForCausalLM(tiny_config(kind, vocab))
    return m.to(dtype).eval()
