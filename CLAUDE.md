# CLAUDE.md — working instructions for this repository

Read this file first, then `docs/PLAN.md`, then the spec for the milestone you are working on
(`docs/specs/`). The data pipeline (traces → records, commands, formats) is described in `docs/DATA.md`.
Facts about the competition harness are in `docs/reference/harness_facts.md`;
treat them as ground truth and do not re-derive them from memory.

## What this project is

Training code for **joint localizer–fixer training via SFT** on Gemma 4 for the Kaggle
*Gemma 4 Developer Agent* competition and its paper track. Two LoRA adapters share one frozen
base model: a **localizer** finds where an issue lives, a **fixer** edits the code. The
localizer is trained with weights that come from how likely the fixer is to produce the
verified patch given the localizer's output (EM over a latent localization; see
`context/DESIGN_joint_localizer_fixer.md`). Baselines: a single agent trained on full teacher
trajectories, and localizer/fixer trained separately.

The code will be published with the paper. Write it so an outside reader can reproduce every
number: configs in files, seeds fixed, every run saves its resolved config and git SHA.

## Non-negotiable rules

1. **Train on the exact numerics vLLM serves.** The deployed base is
   `gemma-4-*-it-qat-w4a16-ct` (int4 weights, group 32, symmetric) with bf16 LoRA adapters served
   on top, never merged. Therefore:
   - train bf16 LoRA on the **dequantized `-w4a16-ct` checkpoint** (int4 × scale → bf16, done once
     by `locfix quant dequantize`);
   - never load the base in 4-bit/NF4 (no bitsandbytes, no QLoRA), never use torchao QAT
     fake-quantization, never merge adapters for submission;
   - plain LoRA only (no DoRA, no PiSSA, no rsLoRA unless the parity test passes with it).
2. **Every training example must render to the same tokens vLLM sees in the harness.** Records
   store OpenAI-format chat messages and tool schemas exactly as LiteLLM sends them; rendering uses
   the model's own chat template. Golden tests compare our token ids with vLLM's `/tokenize` on
   requests captured from real harness runs (spec 02). If a golden test fails, fix the renderer;
   never loosen the test.
3. **Never change teacher actions or observations.** Tool-call arguments, tool outputs and patches
   from teacher traces are immutable. Only reasoning text may be rewritten (by the cleaning
   pipeline, not here), and only loss masks/weights may be changed here.
4. **The harness is the source of truth.** Import functions from the `swegemma` /
   `adk_submission` wheels (e.g. `build_agent_prompt`) instead of re-implementing them. When
   re-implementation is unavoidable (e.g. the budget-warning text), add a test that compares our
   output with the harness function on the same inputs.
5. **Only harness tools.** Agents may use the built-in tools, sub-agents via `agent_tool`, and
   skills. No custom tools.
6. **No data, weights or secrets in git.** Paths come from `configs/paths.yaml` (gitignored; a
   `paths.example.yaml` is committed). The `context/` and `vendor/` folders are private and
   gitignored.
7. **Fail loudly.** Converters raise on unexpected input (unknown entry type, edit that does not
   apply, adapter key that vLLM will not load). Silent fallbacks have caused real bugs in this
   kind of pipeline (e.g. vLLM ignoring LoRA modules it does not support).

## Environment

- Python **3.12** (the `swegemma` wheel requires >=3.12).
- Harness wheels (private, in `vendor/`): `swegemma-0.2.11`, `adk_submission-0.2.13`,
  `adk_eval_core-0.1.0`; `google-adk>=1.34,<2` (1.39.1 was inspected).
- Training: `transformers==5.13.1` (preferred, see below), `peft`, `trl`, `accelerate`
  (FSDP2 on H100, DDP or FSDP2 on H200), `liger-kernel` (fused/chunked cross-entropy is required:
  vocab is 262K).
- Serving/scoring: **`vllm==0.19.1`** and **`transformers==5.13.1`** — the versions in the Kaggle
  scoring wheelhouse. LoRA behaviour and chat-template rendering differ across versions, so the
  serve environment must use exactly these. Use `transformers==5.13.1` in the training environment
  too if `trl`/`peft` allow it (the chat template and tokenizer then match by construction);
  otherwise the golden render tests (spec 02) must pass across the two versions.
- Hardware: 8×H200 (primary), H100 possible. Evaluation needs Docker on the same node.
- Use `uv` for environments; two lockfiles are fine if vLLM and the trainer need different torch
  versions (`envs/train`, `envs/serve`).

## Where code runs

Code is written and CPU-tested in the build workspace (no GPU, no Hugging Face downloads), pushed
to the public repo, and GPU steps run from `notebooks/` on Kaggle (competition models and vLLM
0.19.1), Colab (E4B) and the training node. Keep every GPU step runnable as one script with
`--help`, and keep CPU tests independent of model downloads (use tiny configs and the fixtures).

## Repository layout (target)

```
locfix/
├── CLAUDE.md, README.md, LICENSE (Apache-2.0), pyproject.toml
├── configs/        defaults.yaml, paths.example.yaml, model/*.yaml, runs/*.yaml
├── schemas/        record.schema.json, scoring_record.schema.json
├── src/locfix/
│   ├── quant/      dequantize.py, parity.py
│   ├── harness/    capture_proxy.py, golden.py, prompts.py (wraps swegemma), budget.py
│   ├── data/       traces.py (load/validate), agent_records.py, compaction.py,
│   │               locations.py (AST index, L_edit, L_view), edits.py (diff → edit_file),
│   │               render_obs.py (offline read_file / grep rendering), loc_fix_records.py,
│   │               unsolved.py, filters.py
│   ├── render/     chat.py (messages → tokens + loss mask), spans.py (y*₊ token spans)
│   ├── train/      sft.py (TRL subclass), collate.py, export.py (adapter checks)
│   ├── score/      vllm_scorer.py, weights.py (ESS-targeted τ, λ calibration), loo.py
│   ├── em/         candidates.py, rounds.py (orchestration)
│   ├── eval/       run_harness.py, stats.py (McNemar, CIs), timing.py
│   └── submit/     package.py (agent.yaml + adapters + size check)
├── agents/         agent YAML templates: single/, locfix/ (fixer root + localizer agent_tool)
├── notebooks/      one script per GPU step (Kaggle / Colab / node)
├── tests/          unit tests with tiny fixtures; GPU tests marked @pytest.mark.gpu
├── tools/          gpu_hours_estimate.py and small utilities
├── docs/           PLAN.md, specs/, reference/
└── context/        private background docs (gitignored)
```

One CLI entry point, `locfix`, with subcommands per module (`locfix quant dequantize`,
`locfix data agent-records`, `locfix train`, `locfix score`, `locfix eval`, …). Every subcommand
takes `--config` and writes outputs plus `run_meta.json` (config, git SHA, seed, versions).

## Working style

- Build milestone by milestone in the order of `docs/PLAN.md`; each spec ends with acceptance
  tests. A milestone is done when its acceptance tests pass, not when the code exists.
- Develop on **E4B** (cheap), then run on 31B.
- Small fixtures in `tests/fixtures/` (the four sample traces in `context/` are good ones).
- Items marked **VERIFY** in the specs are facts we could not confirm without running the
  harness. Resolve them in milestone M0 and record the answer in
  `docs/reference/harness_facts.md`.
- Keep the README's "Reproducing the paper" section up to date as commands become final.
