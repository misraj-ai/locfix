# Starting the build in Claude Code

## Before the first session

1. Create the repository from this folder (`git init`, first commit). `context/`, `vendor/` and
   `configs/paths.yaml` are gitignored on purpose.
2. Put the three harness wheels in `vendor/` (`swegemma-0.2.11`, `adk_submission-0.2.13`,
   `adk_eval_core-0.1.0`).
3. Copy `configs/paths.example.yaml` to `configs/paths.yaml` and fill in the paths on the GPU node
   (data root `codeagent_final/`, competition dev data, model directory, work directories).
4. Make Hugging Face access available on the node (Gemma 4 `-w4a16-ct` checkpoints) through the
   environment, not in files.
5. Have a Kaggle L4×4 notebook ready for the timing runs (spec 07); it is the only place that
   gives real competition timings.
6. Versions are known: vLLM 0.19.1, transformers 5.13.1 (Kaggle scoring wheelhouse).
7. If you can, copy the competition's `scripts/inference.py` into `context/` (it has
   `MODEL_VLLM_EXTRA_ARGS` and the compaction settings).

## First prompt to paste

> Read `CLAUDE.md`, then `docs/PLAN.md`, then `docs/specs/01_quant_and_parity.md`,
> `docs/specs/02_harness_capture.md` and `docs/reference/harness_facts.md` §13. Set up the
> Python 3.12 project (pyproject, `uv`, `locfix` CLI skeleton, tests folder, Apache-2.0 LICENSE)
> with vllm==0.19.1 and transformers==5.13.1 (one environment if trl/peft allow it, otherwise a
> separate train environment). **First**, run VERIFY item 0 of spec 02: start vLLM exactly as the
> harness does with `--enable-lora` and a dummy adapter on E4B and 31B, and report whether it
> loads. Then implement milestone M0 (capture proxy and the capture sessions) and M1 (dequantize +
> parity) for E4B. Stop after each milestone's acceptance
> tests and report the results, including answers to the VERIFY items in
> `docs/reference/harness_facts.md` §9.

## Later prompts (one milestone each)

- M2: "Implement spec 03 (agent records via scripted replay). Use `tests/fixtures/traces/` for
  tests and the first-pass cleaned data until cleaning v2 lands."
- M3: "Implement spec 05 and run the Stage 1 LR sweep on E4B."
- M4: "Implement spec 07 (runner, stats, packaging, timing) and run Stage 0."
- M5–M7: specs 04 and 06, in order.
- Keep `docs/reference/harness_facts.md`, `configs/defaults.yaml` and the README's reproduction
  section updated as facts and commands become final.
