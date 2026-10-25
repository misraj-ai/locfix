# Hyperparameters: fixed values, search ranges, rationale

Values live in `configs/defaults.yaml`; this file explains them.

## Data scale

1,577 trajectories × ~17K tokens × 1.2 (compaction windows) ≈ **32M tokens per epoch** ≈ 840
sequences packed to 32K. Global batch 8–16 sequences → 50–100 optimizer steps per epoch. LoRA is
less tolerant of large batches (keep the effective batch < 32 sequences).

## Fixed

| Setting | Value | Why |
| --- | --- | --- |
| Targets | language-model q,k,v,o,gate,up,down (as vLLM supports) | attention-only LoRA underperforms; unsupported modules are silently dropped by vLLM |
| `lora_alpha` | 32 for every rank, standard scaling | with 1/r scaling the best LR is nearly rank-independent, so one LR sweep serves all ranks |
| Dropout, weight decay | 0, 0 | small data, few steps |
| Optimizer | AdamW (0.9, 0.999), clip 1.0, bf16 | |
| Schedule | 10 warmup steps, cosine to 10% of peak | |
| Loss | token-weighted mean over all trained tokens of the global batch | long trajectories are not under-weighted; accumulation-invariant |
| Max length | 32,768, padding-free packing | matches serving |

## Searched

| Setting | 31B | E4B | Expected |
| --- | --- | --- | --- |
| Peak LR | {5e-5, 1e-4, 2e-4} | {1e-4, 2e-4, 4e-4} | 1e-4 / 2e-4 |
| Rank | {32, 64}; 128 for the single agent only | {32, 64} | 64 |
| Epochs | checkpoints at 2 and 3 of one 3-epoch run | same | 2–3 |
| Global batch | {8, 16} | {8, 16} | 16 |

Adapter size: rank 64 on 31B is roughly 0.45–0.9 GB; localizer + fixer at 64 fit under 3 GiB.

## Per stage

- Stage 1: the sweep happens here; later stages reuse LR and rank.
- Stage 2 localizer: rank {16, 32}, short sequences, 2–3 epochs. Fixer: like Stage 1; 2b has ~3
  records per issue → ~1 epoch. Rule: **each y\* seen ≈ 3 times per run** (avoids a memorizing
  fixer, which flattens EM weights).
- Stage 3: warm start, LR × 0.5, 1 epoch per round, ≤ 3 rounds, stop when held-out resolve stops
  improving.
- Stage 4: unsolved mixing ratio {0, 0.5, 1.0}.
- Stage 5: warm start, LR × 0.5, 1–2 epochs, ≤ 2 trajectories per task.

## EM settings

| Knob | Rule / values | Default |
| --- | --- | --- |
| Score | mean log p per y\*₊ token | |
| K | 8 (16 if candidates are near-identical) | 8 |
| τ | per issue, set by target ESS ∈ {1, 2, 4} | 2 |
| λ_ctx | median-sized z costs {0, 0.25σ, 0.5σ} (σ = median within-issue score std, round 0) | 0.25σ |
| λ_req | one request costs {0.1σ, 0.25σ, 0.5σ} | 0.25σ |
| β (anchor) | {0.25, 0.5, 1} | 0.5 |
| M (resample) | 2–4 | 3 |
| No-request fraction | 40% | |
| z budget | ≤ 8 locations, ≤ 6K tokens | |
| Localizer calls | ≤ 12 counted calls | |

Search order: ESS target → λ_ctx → others at defaults.

## Protocol

1. E4B first (pipeline debugging + its LR sweep); 31B gets its own 3-point LR sweep (1 epoch).
2. Prune by held-out assistant-token NLL; decide by harness resolve rate.
3. Fix LR and rank per model, then keep them for every stage.
4. Final comparisons: heldout (~300) + dev (129), paired McNemar, 2 sampling seeds.

## Sources

- LoRA Without Regret — https://thinkingmachines.ai/blog/lora/
- TRL guide — https://huggingface.co/docs/trl/main/en/lora_without_regret
- Beware of the Batch Size (LoRA) — https://arxiv.org/html/2602.09492v1
- SWE-smith (≤ 3 trajectories per task; repeated easy ones hurt) — https://arxiv.org/pdf/2504.21798
