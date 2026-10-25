# Build plan

Status: agreed design as of 2026-10-24. Deadlines (UTC): **paper track Nov 12**, competition entry
and team merger Nov 25, **final submissions Dec 2**. Small-model bonus ($10K) goes to the best
E2B/E4B submission.

## 1. Decisions already made

| Topic | Decision | Where |
| --- | --- | --- |
| Models | **E4B** (debug + bonus), **31B** (main track); 12B optional (LoRA serving on both confirmed on Kaggle) | |
| Quantization | "QAT" = bf16 LoRA trained on the **dequantized `-w4a16-ct`** base, adapters served unmerged on the int4 base; a parity test proves training and serving match | spec 01 |
| Training stack | TRL `SFTTrainer` subclass + PEFT + Accelerate (DDP/FSDP2) + Liger kernels | spec 05 |
| Versions | serve: vLLM 0.19.1 + transformers 5.13.1 (Kaggle wheelhouse); train: same transformers if possible | |
| Scoring / rollouts / eval | vLLM on the real `-w4a16-ct` base with multi-LoRA | specs 06, 07 |
| Agent structure | Single agent (baseline). Ours: **fixer = root agent, localizer = `agent_tool` sub-agent** (`skip_summarization: true`); the fixer can call it again for more context | spec 04 |
| Tools | harness tools only: `run_command`, `read_file`, `edit_file`, `write_file`, `get_status`, `submit_patch` (graph tools dropped from trained agents; keeping them is an ablation) | |
| Thoughts | `include_thoughts: true` by default; Stage 0 checks on/off | |
| Budget | 5 min per task, ~30 tool calls (the warning fires at ≤10 remaining), turns as now; final values from Kaggle timing runs | spec 07 |
| Data | `cleaned/` solved set (1,577) after the **second recall-cleaning pass** (run separately, see `context/RECALL_CLEANING_V2.md`); `experiment/` unsolved set (1,338) | |
| Hyperparameters | `configs/defaults.yaml`, rationale in `docs/reference/hyperparameters.md` | |
| Compute | about 530 H200 GPU-hours for 31B, 130 for E4B (`docs/reference/compute.md`) | |

## 1b. Where each part runs

| Place | Resources | Used for |
| --- | --- | --- |
| **Build workspace** (this Claude session) | CPU, can install the harness wheels, google-adk, transformers 5.13.1, PEFT; no GPU, no Hugging Face download | writing all code; CPU unit tests (trace loading, budget/prompt via the real harness functions, scripted replay with a fake model, diff → edit converter, observation renderer, locations, weights math, stats, packaging) |
| **Kaggle notebook** (L4×4, 96 GB, internet off) | the competition's own models attached, vLLM 0.19.1 wheelhouse, dev data, harness | anything that must match the competition: **LoRA load check**, golden captures (M0), timing runs, serve-side parity for 31B, final submission checks |
| **Colab** | single GPU (A100/H100 if available), internet on | E4B: dequantize, tokenizer/chat-template golden tests, training smoke tests, parity, short evaluations |
| **8×H200 node** (if available) | real training | Stage 1–5 runs, 31B training, scoring passes, large evaluations |

Every GPU step ships as a script in `notebooks/` that clones the public repo, installs pinned
versions and writes its outputs (reports, adapters) to Drive or Kaggle output. Tests that need a
GPU, a tokenizer download or vLLM are marked `@pytest.mark.gpu` / `@pytest.mark.hf` and run there.

## 2. Stages (what gets trained)

| Stage | What | Baseline or method |
| --- | --- | --- |
| 0 | Base model, no training: single agent, loc+fix structure, thoughts on/off | reference numbers |
| 1 | One agent, SFT on full teacher trajectories | baseline 1 |
| 2 | Localizer (labels: L_edit or L_view) + fixer (2a: gold context, 2b: noisy context) trained separately | baselines 2, 3 (**2b is the one to beat**) |
| 3 | Joint EM: round 0 → signal test → rounds 1–2; variants hard EM / soft-resample / soft-weighted / ±requests / leave-one-out | **our method** |
| 4 | Stage 3 + unsolved tasks for the localizer (and fixer on gold patches); never_found as ablation | our method + data |
| 5 | Rejection sampling on our own model's passing harness runs | after the paper, for the submission |
| — | Our method as one model with two roles | paper ablation |

## 3. Milestones

Each milestone ends when the acceptance tests in its spec pass. Dates are targets.

| ID | Milestone | Spec | Depends on | Target |
| --- | --- | --- | --- | --- |
| **M0** | Repo, environments, vendor wheels; **capture real harness requests** (golden set); resolve VERIFY items | 02 | — | Oct 24–26 |
| **M1** | Dequantize E4B/31B(/12B); grid check; **parity test** (base only, random adapter, trained adapter later) | 01 | M0 env | Oct 25–27 |
| **M2** | Trace → agent records (Stage 1): prompt rebuild, budget rewrite, filters, loss masks, compaction replay; golden render tests | 03 | M0 | Oct 26–29 |
| **M3** | Trainer; Stage 1 on E4B (LR sweep); adapter export + parity on a trained adapter | 05 | M1, M2 | Oct 28–31 |
| **M4** | Eval runner (dev 129 + held-out ~300), stats; **Stage 0** on E4B and 31B; Kaggle timing runs → `eval_config.yaml` | 07 | M0 | Oct 27–31 (parallel) |
| **M5** | Locations, z format, edit converter, observation renderer, localizer/fixer records, agent YAMLs; **Stage 2** on E4B | 04 | M2, M3 | Oct 30–Nov 3 |
| **M6** | Scorer, weights, candidates; EM round 0 + **signal test**; round 1 | 06 | M5 | Nov 2–5 |
| **M7** | EM round 2; **Stage 4** (unsolved) on E4B | 06, 04 | M6 | Nov 5–8 |
| **M8** | 31B: Stage 1 sweep, Stage 2b, best Stage 3/4 configuration | 05–07 | M3 (then M6) | Nov 3–10 |
| **M9** | Paper tables/figures, code release tag, reproduction section of README | 08 | M7, M8 | Nov 8–12 |
| M10 | Stage 5, submission tuning, E4B bonus submission | 07 | M9 | Nov 13–Dec 2 |

**Critical path:** M0 → M2 → M3 → M5 → M6 → M7 → M9. M1 and M4 run in parallel with M2. 31B
work (M8) starts as soon as Stage 1 works on E4B.

**If behind on Nov 5:** the paper reports Stages 0–3 on E4B plus Stages 1, 2b and the best
Stage 3 configuration on 31B; Stage 4 moves to an appendix or future work.

## 4. Data flow

```
codeagent_final/cleaned (after recall v2) ──► M2 agent records ──► Stage 1
                                         └──► M5 locations / z / y* ──► Stage 2 records
experiment/ (unsolved, after recall v2) ──► M5 unsolved records ──► Stage 4
Stage 2 adapters ──► M6 candidates ──► vLLM scoring ──► weights ──► Stage 3 records ──► rounds
every adapter ──► M1 parity check ──► M4 harness eval ──► results table
```

Until the second cleaning pass finishes, develop on the first-pass output. Every builder must be
re-runnable on the new data with one command, and record counts must be reported per source.

## 5. Risks and the check that catches each

| Risk | Check |
| --- | --- |
| ~~vLLM 0.19.1 refuses LoRA on 31B/E4B~~ | **resolved 2026-10-25:** adapters load and apply on both (Kaggle check, `harness_facts.md` §13) |
| vLLM silently ignores some LoRA modules, or k/v-sharing layers apply the adapter differently | parity test with the adapter (spec 01): adapter must change vLLM log-probs exactly as in training |
| Training renders differently from what the harness sends (thought placement, tool-call syntax) | golden render tests against captured requests and vLLM `/tokenize` (spec 02, 03) |
| Chat template not prefix-consistent across turns (e.g. drops earlier thoughts) | prefix-consistency test on captured requests decides one-sequence vs per-turn training (spec 02) |
| Compaction in training data differs from the real one | replay uses ADK's own summarizer code path; compare with a captured long session (spec 03) |
| Teacher recall in reasoning | second cleaning pass (separate), plus loss mask on reference-seeking calls |
| Dev set too small to separate methods | held-out ~300 tasks, paired McNemar tests, 2 seeds (spec 07) |
| 12-hour Kaggle limit | timing runs on Kaggle L4×4 before every submission (spec 07) |
| Fixer memorizes y\*, EM weights go flat | cross-fit folds A/B; ~3 visits per y\* (spec 05, 06) |

## 6. Outputs per run (for the paper)

Every training run writes: resolved config, git SHA, data manifest (record counts per source and
stage, token totals), training curves, adapter + parity report. Every eval run writes:
per-task results (JSONL), resolve rate with 95% CI, time and tool-call statistics, and the
agent YAML used.
