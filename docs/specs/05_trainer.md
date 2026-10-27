# Spec 05 — Trainer

## Purpose

One trainer for every stage (baselines and our method), so comparisons in the paper differ only
in data and weights, never in training code.

## Stack

transformers `Trainer` subclass + PEFT LoRA + Accelerate, on the dequantized bf16 base (spec 01).
(Implemented without TRL and Liger: the data pipeline and the loss are our own, and the loss is
computed in chunks on trained positions only, so neither adds anything.) No Unsloth.

## Inputs

- Records (spec 03/04 format), one or more files with mixing ratios.
- Model config (`configs/model/<model>.yaml`): base path (dequantized), LoRA target modules
  (derived in spec 01), chat-template renderer settings, text-only class for 12B.
- Run config (`configs/runs/<run>.yaml`), defaults in `configs/defaults.yaml`.

## Data pipeline

1. Render each record with `src/locfix/render/chat.py` → `input_ids`, `labels` (−100 outside
   trained assistant tokens), `token_weights` (per-token float: record `weight` × turn `weight`;
   0 where not trained).
2. Drop examples longer than `train.max_seq_len` (32,768) and report them (they should not exist
   after spec 03 filters).
3. **Padding-free packing** with sequence boundaries (`position_ids` reset per example, varlen
   flash-attention); no attention across examples. Check Gemma 4's sliding-window/global attention
   works with the packed varlen path in the pinned `transformers`; if not, pack by padding to
   buckets instead.

## Loss

```
loss = Σ_tokens token_weight · CE(token) / Σ_tokens token_weight
```

summed over the **whole global batch across devices and gradient-accumulation steps** (not a mean
of per-example or per-device means). Use Liger's fused linear cross-entropy (never materialize
full 262K-vocab logits). Implement in `compute_loss` of the trainer subclass; verify the
normalization with a test that compares 1 device × 2 accumulation steps against 2 devices × 1 step
on the same data.

Soft-EM "weighted" variant uses the same loss with candidate weights in `weight`; the
"resampled" variant sets all weights to 1 and repeats records instead.

## LoRA and optimizer (defaults; search ranges in `configs/defaults.yaml`)

`r` 64 (E4B and 31B), `lora_alpha` 32 for every rank, dropout 0, standard scaling (no rsLoRA),
targets from the model config. AdamW (0.9, 0.999), weight decay 0, grad clip 1.0, bf16, 10 warmup
steps, cosine decay to 10% of peak. Peak LR: 1e-4 (31B), 2e-4 (E4B) until the sweep decides.
Global batch 16 sequences (8 GPUs × 1 × 2 accumulation). Gradient checkpointing on.

## Parallelism

- **H200:** 31B bf16 (~62 GB) + checkpointed activations at 32K fit on one GPU → DDP, or several
  independent runs side by side (sweeps, cross-fit fixers A/B, localizer).
- **H100:** FSDP2 across ≥ 2 GPUs for 31B.
- Launch through `accelerate launch` with configs in `configs/accelerate/`.

## Stage rules (from `docs/reference/hyperparameters.md`)

- Stage 1 sweeps LR and rank; later stages reuse them.
- Each verified patch y\* is seen ≈ 3 times in total per training run (epochs × records per issue).
- EM rounds warm-start from the previous round's adapter with LR × 0.5, 1 epoch.
- Stage 5 warm-starts from the best adapter with LR × 0.5, 1–2 epochs.

## Checkpoints and export

- Save adapters at the end of each epoch and at the end; keep optimizer state only for the last.
- `export.py`: write `adapter_config.json` + `adapter_model.safetensors` (PEFT format, names that
  vLLM loads), apply the k/v-sharing fix if spec 01 requires it, run the parity test (spec 01,
  mode 3), write `parity.json` next to the adapter. An adapter without a passing parity report
  cannot be evaluated or packaged.

## Logging

Per step: loss, token-weighted loss per stage/source in the mix, grad norm, LR, tokens/s per GPU
(feed the measured value into `tools/gpu_hours_estimate.py`). Per epoch: held-out NLL on
`val.jsonl` per record type. Save `run_meta.json` (config, git SHA, data manifests, versions).

## Acceptance tests

- Loss normalization test (devices vs accumulation) passes.
- Masking test: tokens of system/user/tool messages and zero-weight turns have label −100 or
  weight 0; the first and last trained token of each assistant turn are correct (decode and check
  on the four fixture traces).
- Overfit test: E4B on 8 records reaches near-zero loss; generation from the trained adapter in
  vLLM reproduces a trained turn (sanity check of the whole train → export → serve path).
- Stage 1 E4B run completes; tokens/s per GPU logged; parity report passes.

## Implementation status (2026-10-26)

Implemented: `src/locfix/train/` (`data.py`, `loss.py`, `model.py`, `sft.py`, `export.py`),
`src/locfix/quant/dequantize.py`, `src/locfix/quant/parity.py`, CLI `train`, `train.plan`,
`quant.dequantize`, `quant.parity`; guide `docs/TRAINING.md`; GPU check `notebooks/train_smoke.py`.

Acceptance status:
- Loss normalization (devices vs accumulation, and vs a hand-computed SGD step): **passes**
  (`pytest -m multiproc`). This test found a real DDP bug: Gemma 4 registers RoPE buffers in
  `set()` order, so ranks hold them in different orders and DDP's positional broadcast corrupts
  them on every rank but 0. Fixed by sorting buffers before wrapping (`stabilize_buffer_order`).
- Masking: trained tokens are exactly the renderer's labels (`render_record`, golden-checked);
  packed rows equal separate records (sdpa and eager, both model shapes).
- Overfit + vLLM generation, Stage 1 E4B run, tokens/s: pending GPU (`notebooks/train_smoke.py`).

Differences from the text above: packing is optional (`train.packing`, default off; on for
localizer records); the model is the text-only `Gemma4ForCausalLM` (the dequantizer writes a
text-only checkpoint; adapter keys then match the layout verified on Kaggle); global batch is
counted in rows (one record, or several packed records, per row).
