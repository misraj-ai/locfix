# Training: from records to vLLM-ready adapters

One trainer serves every experiment; runs differ only in their record files and settings
(`configs/runs/*.yaml`). The data side (traces → records) is in [`DATA.md`](DATA.md).

```
 -w4a16-ct checkpoint ──► locfix quant.dequantize ──► bf16 text-only base (exactly int4 × scale)
                                                            │
 records (*.jsonl) ──► locfix train.plan (dry run)          │
        │                                                   ▼
        └────────────► locfix train  (accelerate launch) ──► adapters per epoch + export/ (vLLM-ready)
                                                            │
                         locfix quant.parity ◄──────────────┘  HF (bf16 + adapter) vs vLLM (-ct + LoRA)
```

## 1. Why it is built this way

- **Train on what vLLM serves.** Kaggle serves the int4 `-w4a16-ct` model with a bf16 LoRA on top,
  unmerged. We dequantize the int4 weights once (int4 × group scale → bf16, with compressed-tensors'
  own decompression) and train a plain bf16 LoRA on that. No 4-bit training kernels, no fake
  quantization, no merging. `quant.parity` measures the remaining difference.
- **Exact training text.** Records are rendered with `render_record`, the same renderer that
  matched vLLM token-for-token (141/141).
- **Token-weighted loss over the global batch.** loss = Σ w·CE / Σ w over all trained tokens of the
  optimizer step, across GPUs and accumulation steps, so long and short records count per token and
  the result does not depend on how the batch is split (tested: 2 processes × 1 step = 1 process ×
  2 accumulation steps = a hand-computed step).
- **Never materialize full logits.** The vocabulary has 262,144 entries; logits are computed only at
  trained positions, in chunks, under activation checkpointing, with Gemma 4's softcap (30).

## 2. Setup

Training node / Colab (CUDA):
```bash
pip install "transformers==5.13.1" peft accelerate safetensors compressed-tensors   # + torch with CUDA
pip install -e .
# for quant.parity and the smoke test also vLLM, the COMPETITION build (harness_facts §16):
#   pip install "vllm==0.19.1"                                         # dependencies
#   pip install --no-deps --force-reinstall <wheelhouse>/vllm-*.whl    # the build Kaggle scores with
# (the public PyPI 0.19.1 cannot serve LoRA on Gemma4ForConditionalGeneration)
```
Tested versions: transformers 5.13.1, peft 0.21.2, accelerate 1.15.0, compressed-tensors 0.15.0.1,
torch 2.14 (CPU tests). The harness wheels are not needed for training.

## 3. Step by step

### Step 1: dequantize the base (once per model)

```bash
locfix quant.dequantize /kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2 \
    --out work/models/gemma4-e4b-w4a16-dequant-bf16
```

| | |
| --- | --- |
| Input | the `-w4a16-ct` folder (config.json with `quantization_config`, safetensors shards) |
| Output | bf16 text-only checkpoint (`Gemma4ForCausalLM`, `model_type: gemma4_text`), tokenizer and chat template copied, `dequantize_report.json` |
| Checks | every sampled layer: weight ÷ group scale is an integer in [-8, 7] up to bf16 rounding (≤ 0.031); unknown tensor names raise |
| Size / time | E4B ≈ 16 GB, 31B ≈ 62 GB on disk; CPU only, a few minutes (streams shard by shard) |

The vision/audio towers are left out (training is text only), which also gives adapter key names
in the layout verified on Kaggle. Set `models.<m>.dequantized` (or `train.base_model`) to the output.

### Step 2 (GPU, once per environment): smoke test

```bash
python notebooks/train_smoke.py --ct /path/to/gemma-4-e4b-it-qat-w4a16-ct     # or --kaggle <handle>
```
Runs on one Colab A100/H100: dequantize → parity of the base and of a random adapter → an overfit
run on 8 fixture records → export → parity of the trained adapter → vLLM greedy generation with
the adapter reproduces trained turns. Writes `smoke_report.json`. This is the first time the whole
path runs on a GPU and on the real model; send the report back before the 8×H200 runs.

### Step 3: check the plan (dry run, CPU)

```bash
locfix train.plan --config configs/runs/S2b-e4b-fixer.yaml --world-size 8 --tokenizer <tokenizer dir>
```
Tokenizes the sources (cached for the real run) and prints optimizer steps, tokens, trained tokens,
dropped records (too long / render errors) and **visits per task** (how often each teacher fix is
seen: the rule is ≈ 3 per run).

### Step 4: train

```bash
accelerate launch --config_file configs/accelerate/ddp_8gpu.yaml -m locfix.cli train \
    --config configs/runs/S1-e4b.yaml
```

| | |
| --- | --- |
| Input | `train.sources` (record files with `weight` and `repeat`), `train.val_sources`, the dequantized base |
| Output (`train.output_dir`) | `epoch-<k>/` and `final/` (PEFT adapters), `export/<name>/` (vLLM-ready, see below), `val.jsonl` (held-out NLL per stage at each epoch end), `run_meta.json` (config, plan, data reports, versions), `result.json` |
| Logs (every step) | loss, loss per source file, grad norm, LR, tokens/s per GPU |
| Launch files | `configs/accelerate/ddp_8gpu.yaml` (8×H200, DDP), `single_gpu.yaml`, `fsdp2_8gpu.yaml` (H100 31B; not yet run) |

**Export** (automatic at the end, also `locfix.train.export.export_adapter`): checks every key is
`base_model.model.model.layers.<i>.(self_attn|mlp).<proj>.lora_(A|B).weight` (anything else
raises), rank ≤ 128, plain LoRA; on 31B adds the `v_proj` copy of the `k_proj` LoRA in the
k_eq_v layers (5, 11, …, 59); writes bf16 weights and `export.json`.

### Step 5: parity before evaluating an adapter

```bash
locfix quant.parity --bf16 <dequantized dir> --ct <-ct dir> --adapter <run>/export/final \
    --records <val records> --out <run>/parity --threshold <from the first base-only run>
```
Reports mean / max |Δ log p| on trained tokens between HF and vLLM, with and without the adapter,
and the adapter's effect size on both sides. An adapter whose parity fails is not evaluated.

## 4. Experiments as run configs

| Run | Adapter | `train.sources` |
| --- | --- | --- |
| `S1-{e4b,31b}` | single agent | `s1_agent.jsonl` |
| `S2a-e4b-localizer` | localizer | `s2_loc_edit.jsonl` (packing on: short records) |
| `S2a-e4b-fixer` | fixer | `s2_fix_gold.jsonl` |
| `S2b-e4b-fixer` | fixer | `s2_fix_gold.jsonl` + `s2_fix_noisy.jsonl` (1 epoch ≈ 3 visits per fix) |
| `S3` rounds | localizer / fixer | `s3_loc_r<t>` with EM weights in `weight`; `init_adapter` + `lr_scale: 0.5` (spec 06) |

Mixing: `{path, weight, repeat}` per file. `weight` multiplies the loss weights of the file's
tokens; `repeat` 2 uses every record twice per epoch, 0.5 a different random half each epoch.

## 5. Settings (`configs/defaults.yaml`, section `train`)

| Key | Default | Meaning |
| --- | --- | --- |
| `lr.{e4b,31b}`, `lr_scale` | 2e-4, 1e-4; 1.0 | peak LR (sweep in Stage 1); 0.5 for warm starts |
| `lora` | r 64, alpha 32, dropout 0 | plain LoRA on q,k,v,o,gate,up,down of the language model |
| `epochs` | 3 | adapters saved at every epoch end |
| `global_batch_sequences`, `per_device_batch` | 16, 1 | rows per optimizer step; accumulation = 16 / (GPUs × 1) |
| `schedule` | 10 warmup steps, cosine to 10% | |
| `optimizer` | AdamW (0.9, 0.999), wd 0, clip 1.0 | |
| `max_seq_len` | 32768 | longer records are dropped and counted |
| `packing` | false | true: several records per row (first-fit), `position_ids` restart per record |
| `attn_implementation` | sdpa | packing with sdpa is tested (packed row = separate records, exactly) |
| `ce_chunk` | 1024 | trained tokens per vocabulary-projection chunk |
| `cache_dir` | `${paths.records}/token_cache` | tokenized records, reused across runs |

## 6. Things to know

- **DDP and Gemma 4's RoPE buffers.** transformers 5.13.1 registers the rotary-embedding buffers in
  `set()` order, which differs between processes (Python hash randomization). DDP broadcasts module
  state by position, so without a fix every GPU except the first silently trains with wrong
  positional frequencies. `LossModel` sorts all buffers by name before wrapping
  (`stabilize_buffer_order`); `tests/test_train.py` checks both the problem and the fix.
- **Packing memory.** With sdpa, a packed or sliding-window row uses a dense T×T mask (≈ 1 GB per
  mask at 32K). Fine on H200; packing is off by default except for the short localizer records.
- **Not yet done:** resuming an interrupted run (adapters are saved per epoch; restart from one
  with `init_adapter`), FSDP2 on real hardware, Liger kernels (not needed: logits are chunked).
- **Tests:** `pytest` (CPU, tiny random Gemma 4 models of both shapes: 31B-like k_eq_v and E4B-like
  per-layer inputs / shared KV) and `pytest -m multiproc` (2-process DDP normalization test).
