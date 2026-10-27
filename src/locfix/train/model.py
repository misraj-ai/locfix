"""Base model + LoRA, and the module the trainer optimizes.

The base is the text-only Gemma 4 model (`Gemma4ForCausalLM`) loaded from the dequantized
`-w4a16-ct` checkpoint (`locfix quant dequantize`): exactly int4 × scale in bf16, the weights vLLM
serves. LoRA (bf16-servable, standard scaling) goes on the language model's attention and MLP
projections. Module names are those of the adapter layout verified on Kaggle
(`base_model.model.model.layers.<i>.{self_attn,mlp}.<proj>`).

`LossModel` wraps the PEFT model so that the whole loss is computed inside `forward` (the
distributed wrapper's gradient hooks see every parameter use) without materializing full logits.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .loss import trained_positions, weighted_ce_sum

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_base(model_dir: str, dtype: torch.dtype = torch.bfloat16, attn_implementation: str = "sdpa",
              gradient_checkpointing: bool = True, device_map: Any = None):
    """Text-only Gemma 4 from a dequantized checkpoint. Refuses quantized checkpoints."""
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    cfg = AutoConfig.from_pretrained(model_dir)
    if getattr(cfg, "quantization_config", None):
        raise ValueError(f"{model_dir} is quantized; train on the output of `locfix quant dequantize`")
    if getattr(cfg, "model_type", "") != "gemma4_text":
        raise ValueError(f"{model_dir}: expected a text-only gemma4_text checkpoint, got {cfg.model_type!r}")
    model = Gemma4ForCausalLM.from_pretrained(model_dir, dtype=dtype, attn_implementation=attn_implementation,
                                              device_map=device_map)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def add_lora(model, r: int, alpha: int, dropout: float = 0.0, init_adapter: str | None = None,
             targets: tuple[str, ...] = LORA_TARGETS):
    """Plain LoRA (no rsLoRA/DoRA) on the language-model projections; or load an adapter to
    continue training (warm start)."""
    from peft import LoraConfig, PeftModel, get_peft_model

    if init_adapter:
        return PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    regex = r"model\.layers\.\d+\.(self_attn|mlp)\.(" + "|".join(targets) + ")"
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=regex, bias="none",
                     use_rslora=False, use_dora=False, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def stabilize_buffer_order(module: nn.Module) -> int:
    """Sort every submodule's buffers by name; returns the number of modules reordered.

    DDP (and FSDP) broadcast module state from rank 0 *by position*. Gemma 4's rotary embedding
    registers its per-layer-type buffers while iterating over `set(config.layer_types)`, whose
    order depends on Python's per-process hash seed, so ranks can hold the buffers in different
    orders and the startup broadcast writes the sliding-window RoPE frequencies into the
    full-attention ones (shapes 8 vs 16 in the tiny test model). Found by
    `tests/test_train.py::test_loss_normalization_devices_vs_accumulation`."""
    n = 0
    for m in module.modules():
        if len(m._buffers) > 1:
            ordered = dict(sorted(m._buffers.items()))
            if list(ordered) != list(m._buffers):
                n += 1
            m._buffers = ordered
    return n


def lora_module_names(peft_model) -> list[str]:
    return sorted(n for n, m in peft_model.named_modules() if hasattr(m, "lora_A") and len(m.lora_A))


class LossModel(nn.Module):
    """forward(batch) → {"loss": Σ w·CE, "ce": per trained token CE, "source_ids": per trained token}."""

    def __init__(self, peft_model, ce_chunk: int = 1024):
        super().__init__()
        stabilize_buffer_order(peft_model)          # before any DDP/FSDP wrapping
        self.peft = peft_model
        self.ce_chunk = ce_chunk
        causal_lm = peft_model.get_base_model()
        self.softcap = causal_lm.config.final_logit_softcapping
        self._no_split_modules = getattr(causal_lm, "_no_split_modules", None)

    @property
    def causal_lm(self):
        return self.peft.get_base_model()

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, labels: torch.Tensor,
                loss_weights: torch.Tensor, source_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        lm = self.causal_lm
        hidden = lm.model(input_ids=input_ids, position_ids=position_ids, use_cache=False).last_hidden_state
        rows, cols = trained_positions(labels)
        h = hidden[rows, cols - 1]
        loss, ce = weighted_ce_sum(h, lm.lm_head.weight, labels[rows, cols], loss_weights[rows, cols],
                                   self.softcap, self.ce_chunk)
        out = {"loss": loss, "ce": ce, "weights": loss_weights[rows, cols].detach()}
        if source_ids is not None:
            out["source_ids"] = source_ids[rows, cols]
        return out
