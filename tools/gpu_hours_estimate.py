"""GPU-hour estimate for the joint localizer-fixer experiments (Gemma 4, bf16 LoRA).

Re-run with measured numbers once the first real run finishes:
  - replace TRAIN_TOK_PER_S / SCORE_TOK_PER_S with measured tokens/s per GPU, or
  - adjust MFU, EPOCH_TOKENS, run counts.
All counts are per model. "E" = one epoch-equivalent of the full-trajectory data.
"""

# ---------------- data ----------------
N_TRAJ = 1577
MEAN_TOKENS = 17_000          # Kimi tokens, median 14K; Gemma tokenizer within ~10%
COMPACTION_OVERHEAD = 1.20    # extra windows for trajectories > 14K tokens
EPOCH_TOKENS = N_TRAJ * MEAN_TOKENS * COMPACTION_OVERHEAD   # ~32M

# ---------------- hardware ----------------
PEAK_BF16 = 989e12            # dense BF16 FLOP/s, same for H100 SXM and H200
H200_SPEEDUP = {"31B": 1.20, "12B": 1.10, "E4B": 1.10}  # memory headroom: DDP, less recompute
H200_EVAL_SPEEDUP = 1.35      # decode is memory-bandwidth bound (4.8 vs 3.35 TB/s)

# ---------------- models ----------------
# N = compute parameters per token (incl. lm_head); attn = attention FLOPs as a fraction
# of the 2N forward cost at ~17K-token documents (sliding-window + 1-in-6 global layers).
MODELS = {
    #        N        attn  train_MFU score_MFU  eval GPU-h per 430-task run (H100)
    "31B": (30.7e9,  0.08,  0.35,     0.45,      1.5),
    "12B": (12.0e9,  0.10,  0.35,     0.45,      0.8),
    "E4B": ( 5.0e9,  0.15,  0.30,     0.40,      0.5),
}

# ---------------- plan (epoch-equivalents) ----------------
TRAIN_E = {
    "Stage 1  single agent (3 LR x 1E, final 3E, rank variant 2E)": 8.0,
    "Stage 2  localizer x2 labels, fixer 2a, fixer 2b":             6.6,
    "Stage 3  round 0 (loc + fixer A + fixer B)":                   3.6,
    "Stage 3  EM rounds (4 variants r1 + 2 variants r2, 4.8E each)": 28.8,
    "Stage 4  +unsolved (2 mix ratios + never_found ablation)":     18.0,
    "Stage 5  own passing runs (2 runs)":                            3.0,
    "Paper    one model, two roles":                                 4.0,
}
SCORE_TOKENS_PER_PASS = N_TRAJ * 8 * 10_000   # K=8 candidates x ~10K-token fixer context
SCORE_PASSES = 6.5        # round 1, round 2 (x2 variants), stage 4 (~1.5), leave-one-out (~1)
LOC_SAMPLING_GPU_H_31B = 1.5   # per pass, K=8 short localizer generations per issue
LOC_SAMPLING_PASSES = 4
EVAL_RUNS = 45            # stage 0 baselines, sweeps, stage finals x2 seeds, stage-5 rollouts
SAFETY = 1.25             # failed runs, restarts, debugging


def estimate(name):
    N, attn, mfu_t, mfu_s, eval_h = MODELS[name]
    train_flop_tok = 6 * N * (1 + attn * 2.25)     # fwd + recompute fwd + bwd(activations), LoRA
    score_flop_tok = 2 * N * (1 + attn)            # forward only
    train_tps = PEAK_BF16 * mfu_t / train_flop_tok
    score_tps = PEAK_BF16 * mfu_s / score_flop_tok

    train_tokens = sum(TRAIN_E.values()) * EPOCH_TOKENS
    train_h = train_tokens / train_tps / 3600
    score_h = SCORE_PASSES * SCORE_TOKENS_PER_PASS / score_tps / 3600
    score_h += LOC_SAMPLING_PASSES * LOC_SAMPLING_GPU_H_31B * N / MODELS["31B"][0]
    eval_h_total = EVAL_RUNS * eval_h

    h100 = (train_h + score_h + eval_h_total) * SAFETY
    h200 = (train_h / H200_SPEEDUP[name] + score_h / 1.1
            + eval_h_total / H200_EVAL_SPEEDUP) * SAFETY
    return dict(train_tps=train_tps, epoch_h=EPOCH_TOKENS / train_tps / 3600,
                train=train_h, score=score_h, eval=eval_h_total, h100=h100, h200=h200,
                train_tokens=train_tokens)


if __name__ == "__main__":
    print(f"epoch = {EPOCH_TOKENS/1e6:.1f}M tokens; plan = {sum(TRAIN_E.values()):.1f} epoch-equivalents\n")
    print(f"{'model':5} {'tok/s/GPU':>9} {'GPU-h/epoch':>11} {'train':>7} {'score':>6} {'eval':>6} "
          f"{'H100 total':>10} {'H200 total':>10} {'8xH100 days':>11} {'8xH200 days':>11}")
    for m in MODELS:
        r = estimate(m)
        print(f"{m:5} {r['train_tps']:9.0f} {r['epoch_h']:11.1f} {r['train']:7.0f} {r['score']:6.0f} "
              f"{r['eval']:6.0f} {r['h100']:10.0f} {r['h200']:10.0f} "
              f"{r['h100']/8/24:11.1f} {r['h200']/8/24:11.1f}")
    print("\nrange: x0.75 (MFU 45%) to x1.4 (MFU 25%) on the training share")
