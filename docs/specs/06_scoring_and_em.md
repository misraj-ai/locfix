# Spec 06 — Scoring and EM rounds (Stages 3–4)

## Purpose

Compute, for each candidate localization, how likely a frozen cross-fitted fixer is to write the
verified patch given that localization; turn those scores into weights; train the localizer and
fixer on weighted candidates; repeat for a few rounds. See
`context/DESIGN_joint_localizer_fixer.md` §3–4, 9–11 for the math.

## 1. Folds and cross-fitting

Issues are split into folds A and B (`fold` field in `solved.jsonl`; unsolved tasks get a fold by
hash of `instance_id`). Fixer A is trained on fold A and scores fold B's candidates, and vice
versa. A fixer never scores issues it was trained on.

## 2. Candidates (`src/locfix/em/candidates.py`)

Per issue, K = `em.K` (8) candidates z_k:
- **Round 1:** sample from the round-0 localizer (temperature `em.sample_temperature`, 0.8),
  conditioned on the teacher's exploration prefix (cheap: one prefill, K short generations, vLLM
  `n=K`), plus perturbations (spec 04 §6) to fill up to K. Deduplicate identical z.
- **Later rounds:** sample from the current localizer.
- Optional later variant (on-policy): full localizer rollouts in the harness.

## 3. Scoring sequence

For each (issue, z_k, request sequence a_k), build the fixer sequence of spec 04 §5 steps 1–3
(localizer call, reads of z_k, requests a_k), followed by the **canonical edit calls** for the
patch y\* (diff → `edit_file` converter, spec 04 §8) instead of the teacher's iterative edit
phase. Render with the training renderer.

**y\*₊ spans** (`src/locfix/render/spans.py`): the tokens of lines in each `new_string` that are not
in its `old_string` (for `write_file`, all lines). Locate them in the rendered text (tool-call
arguments are serialized by the chat template; map character spans to tokens with the tokenizer's
offset mapping) and test by decoding the selected tokens.

## 4. Scorer (`src/locfix/score/vllm_scorer.py`)

- vLLM on the real `-w4a16-ct` base, fixer adapter as `LoRARequest`, `prompt_logprobs` returning
  the actual token's log-prob, prefix caching on (the issue prompt is shared).
- s_k = **mean** log p over y\*₊ tokens (per-token mean, so scores compare across issues).
- Also store the sum, the y\*₊ token count, tokens(z_k), |a_k|.
- Fallback/validation: HF forward on the dequantized base with the same adapter (no grad) for a
  sample of 50 sequences; must match within the parity threshold (spec 01).

Scoring record (`schemas/scoring_record.schema.json`):

```json
{"instance_id": "...", "fold": "B", "round": 1, "fixer_adapter": "fix_A_r0",
 "cand_id": 5, "z_source": "sample:t0.8", "z": {"locations": [...]},
 "requests": [{"tool": "read_file", "args": {...}}],
 "n_ctx_tokens": 3120, "n_requests": 1, "n_plus_tokens": 96,
 "logp_plus_sum": -41.7, "logp_plus_mean": -0.434,
 "score": -0.51, "w": 0.31, "w_z": 0.44}
```

## 5. Weights (`src/locfix/score/weights.py`)

```
score_k = logp_plus_mean_k − λ_ctx · tokens(z_k)/1000 − λ_req · |a_k|
w_k     = softmax_k(score_k / τ_i)
```

- **τ per issue, set by a target effective sample size:** choose τ_i so that 1/Σ w_k² equals
  `em.ess_target` (1 = hard EM / best-of-K; 2; 4). Solve by bisection on log τ.
- **λ calibration from round-0 scores:** σ = median over issues of the std of `logp_plus_mean`
  across that issue's candidates. Then λ_ctx is set so that adding a median-sized z costs
  `em.ctx_cost_sigma` × σ (0, 0.25, 0.5), and λ_req so that one request costs
  `em.req_cost_sigma` × σ (0.1, 0.25, 0.5).
- **Marginal weight of a localization:** w(z) = Σ_a w(z, a).
- **Anchor:** the z_teacher (L_edit ∪ L_view) localizer record is always added with weight
  `em.beta` (0.5) relative to the candidates' total weight of 1.

## 6. Leave-one-out gains (variant)

For the top-weighted candidate per issue, Δ_i = s(z) − s(z \ {i}) for each location i. Locations
with Δ_i ≤ 0 are pruned from the localizer target ("pruned target" variant). Items the fixer kept
requesting and whose removal lowers the score are added to that issue's localizer target with
their weight (design doc §9).

## 7. Round orchestration (`src/locfix/em/rounds.py`)

```
round 0:  train loc_r0 (z = L_edit ∪ L_view), fix_A_r0, fix_B_r0 (s2_fix_noisy data per fold),
          fix_full_r0 (both folds, for evaluation)
signal test (once, after round 0)
round t ≥ 1:
  1. candidates from loc_{t-1}
  2. score with fix_{other fold}_{t-1}
  3. weights → localizer records (+ anchor) and fixer records
  4. train loc_t (warm start from loc_{t-1}); freeze it
  5. train fix_A_t, fix_B_t, fix_full_t (warm start)
  6. evaluate (loc_t, fix_full_t) in the harness; stop when held-out resolve stops improving
```

Variants (each a run config): `hard` (ESS 1), `soft_resample` (ESS 2–4, M = `em.resample_m`
records per issue drawn ∝ w, weight 1), `soft_weighted` (all K records, weight = w), `no_requests`,
`loo_pruned`. Round 1 runs all variants from one shared scoring pass; round 2 continues the best
two.

## 8. Signal test (decides whether the loop is worth running)

On ~300 held-out-fold issues that have tests: the round-0 fixer generates a patch from (a) the
highest-w candidate and (b) the teacher-label candidate (L_edit ∪ L_view), via a harness-like run
with the localizer answer fixed to that z. Compare resolve rates (paired McNemar). If (a) does not
beat (b), stop and report; the paper still has the baselines and the analysis.

## 9. Stage 4

Same machinery with the unsolved issues added: their candidates (gold location + detours) are
scored by the cross-fitted fixer; their fixer records use the gold patch. Mixing ratio
`em.unsolved_ratio` ∈ {0, 0.5, 1.0} × number of solved issues.

## Acceptance tests

- y\*₊ span test: decoded span tokens equal the added lines for 100% of fixture edits.
- τ bisection hits the ESS target within 1% for all issues; ESS 1 reproduces best-of-K.
- Scorer vs HF forward agree within the parity threshold on 50 sequences.
- A full round on E4B with 100 issues runs end to end and writes scoring records, weights, both
  record sets and adapters.
- Diagnostics logged per round: weight entropy, fraction of issues where the top candidate is the
  teacher label, average |z| tokens, requests per issue, recall/precision of z vs L_edit.
