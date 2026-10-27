# Spec 01 — Dequantized base and train/serve parity

## Purpose

Make training see exactly the weights vLLM serves, and prove it with a test that every adapter
must pass before its evaluation counts.

## Why (short)

vLLM computes `y = Q(W)·x + B·A·x`: `Q(W)` is Google's int4 base (group 32, symmetric), fixed;
`B·A` is our bf16 LoRA, never quantized or merged. Training must use the same `Q(W)` and keep LoRA
in bf16. Dequantizing the `-w4a16-ct` checkpoint once gives a normal bf16 model whose weights are
exactly `int4 × scale`. (torchao QAT + LoRA, as in Unsloth/Axolotl, simulates merge-then-quantize,
which is not our deployment. NF4 QLoRA uses a different 4-bit grid.)

## 1. `locfix quant dequantize`

**Input:** a `-w4a16-ct` checkpoint (`google/gemma-4-{31b,12b,e4b}-it-qat-w4a16-ct`).
**Output:** a bf16 safetensors checkpoint with no `quantization_config` in `config.json`.

Steps:
1. Load with `CompressedTensorsConfig(run_compressed=False)` (decompresses int4 × scale to bf16),
   `torch_dtype=bfloat16`, on CPU.
2. Clear `model.config.quantization_config`; `save_pretrained(..., safe_serialization=True)`.
3. Copy tokenizer, chat template and generation config unchanged.
4. **12B:** vLLM loads it as text-only `Gemma4ForCausalLM` (harness passes
   `--hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'`). Load it
   for training with the same text-only class so adapter key names match what vLLM expects.
5. Unquantized layers in the `-ct` checkpoint (check the `ignore` list in its quantization config:
   typically `lm_head`, embeddings, vision/audio towers, norms) must be copied bit-for-bit.

**Grid check (part of the command):** for a sample of quantized linear layers, read
`weight_packed`/`weight_scale` from the original safetensors, divide the dequantized weight by its
per-group scale (groups of 32 along the input dimension) and assert every value is an integer in
[-8, 7] (exact for symmetric int4). Report max deviation; it must be 0 up to bf16 rounding of the
product.

## 2. LoRA target modules

- Target only the **language-model** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
  down_proj`. Exclude vision/audio towers (Gemma 4 wraps them in `Gemma4ClippableLinear`, which PEFT
  refuses), embeddings, `lm_head`, and E4B's per-layer-embedding projections.
- Derive the final list from **what vLLM's Gemma 4 model class supports for LoRA** in the pinned
  vLLM version (its supported/packed module mapping, e.g. `qkv_proj`, `gate_up_proj`). A module
  PEFT trains but vLLM does not load is a silent train/serve mismatch. Write the derived list to
  `configs/model/<model>.yaml` and test it.
- **k_eq_v layers (confirmed in vLLM 0.19.1, see `harness_facts.md` §13):** in full-attention
  layers with `attention_k_eq_v`, vLLM loads the `k_proj` weight into both K and V slots of the
  packed `qkv_proj`, so a LoRA on `k_proj` only reaches K. In HF, V is computed with the key
  projection, so the trained `k_proj` LoRA affects K and V. `export.py` must add, for exactly those
  layers, a `v_proj` LoRA identical to the trained `k_proj` LoRA. Read the layer list from the
  config (full-attention layers when `attention_k_eq_v` is true; for 31B a published adapter lists
  layers 5, 11, …, 59). The parity test (mode 2) must fail without this fix and pass with it.
- **LoRA serving:** confirmed on Kaggle for E4B and 31B with vLLM 0.19.1 (`harness_facts.md` §13).

## 3. `locfix quant parity`

Compares per-token log-probs of **assistant tokens** between:
- **train side:** dequantized bf16 base (+ adapter via PEFT), HF forward, the same renderer used
  for training records (spec 03);
- **serve side:** vLLM on the real `-w4a16-ct` base (+ adapter via `LoRARequest`), harness flags
  (`max_model_len=32768`, `max_lora_rank` = adapter rank, `tool_call_parser=gemma4`,
  `reasoning_parser=gemma4`), `prompt_logprobs` on the same token ids.

Inputs: ~50 held-out rendered records (mix of short and ~30K-token ones), optional adapter path.

Three modes:
1. **Base only:** proves the dequantization. Expect mean |Δ logp| at bf16 noise level (order
   0.01 nats; set the threshold from the first run and keep it fixed).
2. **Random adapter:** PEFT LoRA with `B` initialized to non-zero random values (so it actually
   changes outputs) and saved like a real adapter. Must (a) match between sides within the same
   threshold and (b) differ clearly from base-only on both sides. If vLLM's numbers equal its
   base-only numbers, vLLM ignored the adapter (wrong module names, unsupported modules, rank cap).
3. **Trained adapter:** run on every adapter before it is evaluated or submitted; store the report
   next to the adapter.

Report: per-record mean/max |Δ|, fraction of tokens with |Δ| > 0.1, and the adapter's effect size
(mean |logp_adapter − logp_base|) on each side.

## Acceptance tests

- Grid check passes on E4B and 31B (all sampled layers integer-valued in [-8, 7]).
- Parity mode 1 and mode 2 pass on E4B and 31B; mode 2 shows an effect size at least 10× the
  base-only |Δ|.
- Exported adapters load in vLLM with `max_lora_rank` equal to their rank and contain only the
  derived target modules (test reads the safetensors keys).
- Adapter size check: total `adapters/` size reported; the submission packer refuses > 3 GiB in
  total with everything else.

## Implementation status (2026-10-26)

`locfix quant.dequantize` (compressed-tensors' `PackedQuantizationCompressor.decompress`, streamed
per shard; text-only output; grid check with tolerance |q|·2⁻⁸ ≤ 0.031, the bf16 rounding bound)
and `locfix quant.parity` (HF and vLLM sides in separate processes; modes base / random adapter /
trained adapter) are implemented and CPU-tested on tiny checkpoints built with compressed-tensors
itself. Pending: the first GPU run on E4B and 31B (`notebooks/train_smoke.py`), which also sets the
parity threshold.
