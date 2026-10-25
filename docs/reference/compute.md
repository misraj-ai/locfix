# Compute estimate

Estimates from FLOP counts and assumed efficiency, not measurements. Re-run
`tools/gpu_hours_estimate.py` with measured tokens/s after the first E4B run.

## Totals per model (25% allowance for failed runs and restarts included)

| Model | H100 GPU-h | H200 GPU-h | H200 range | Compute days on 8×H200 |
| --- | --- | --- | --- | --- |
| 31B | 640 | 530 | 420–700 | 2.8 |
| 12B (optional) | 270 | 240 | 190–310 | 1.2 |
| E4B | 145 | 130 | 105–170 | 0.7 |

## 31B breakdown (H100, before the allowance)

| Part | GPU-h | Basis |
| --- | --- | --- |
| Training | 404 | 72 epoch-equivalents × 32M tokens at ~1,600 tokens/s/GPU |
| Scoring | 40 | 6.5 passes × K=8 × 10K tokens × 1,577 issues + localizer sampling |
| Evaluation + rollouts | 68 | ~45 harness runs over 430 tasks |

One epoch of all trajectories ≈ 5.6 GPU-h ≈ 40 min on 8 GPUs.

Epoch-equivalents by stage: Stage 1: 8; Stage 2: 6.6; Stage 3: 32.4 (round 0 + 4 variants in
round 1 + 2 in round 2); Stage 4: 18; Stage 5: 3; one-model ablation: 4.

## Assumptions

- Training FLOPs/token = 6N (forward + recomputed forward + activation gradients; LoRA skips
  weight gradients) + 8–15% attention.
- Efficiency: 35% of BF16 peak (31B, 12B), 30% (E4B); scoring 40–45%.
- H200 vs H100: same compute peak; ~1.2× for 31B training (no weight sharding needed), ~1.35× for
  evaluation (memory bandwidth).
- Evaluation GPU time = wall-clock × GPUs held; ~32 concurrent tasks hold one GPU ~1.5 h per
  430-task run.
- If the prefix-consistency test (spec 02) forces per-turn training, training tokens grow
  substantially; update the estimate.

## Cut order if short on time

12B runs → round-2 variants (keep the best only) → never_found ablation.
