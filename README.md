# locfix: joint localizer–fixer training via SFT

Code for training a two-role software-engineering agent on Gemma 4 with supervised fine-tuning
only. A **localizer** finds the code relevant to an issue; a **fixer** edits it. Both are LoRA
adapters on one frozen, 4-bit (W4A16) base model. The localizer is trained with weights derived
from how likely the fixer is to write the verified patch given the localizer's output, an EM
procedure over a latent localization, instead of being trained only to find the files of the
reference patch.

Built for the Kaggle *Gemma 4 Developer Agent* competition and its paper track.

> Status: under construction. Results and exact reproduction commands will be added before the
> paper is released.

## What is in here

- **Train/serve parity for quantized deployment:** LoRA is trained in bf16 on the dequantized
  W4A16 checkpoint, so training sees exactly the int4 weights vLLM serves; a parity test checks
  every adapter against vLLM before it is evaluated.
- **Exact training contexts:** teacher trajectories are replayed through the real agent stack
  (Google ADK + LiteLLM + the competition harness tools) with a scripted model, so every training
  example is token-for-token what the deployed agent would see, including budget warnings,
  nudges, sub-agent calls and context compaction.
- **Baselines and method:** single agent on full trajectories; separately trained localizer and
  fixer (gold and noisy context); joint EM with fixer-likelihood weights (hard / soft variants,
  context requests, leave-one-out targets); unsolved tasks for the localizer.
- **Evaluation:** competition harness runs, paired statistics, timing under the competition's
  per-task budget, submission packaging with all competition checks.

## Layout

See `CLAUDE.md` (repository layout and rules) and `docs/PLAN.md` (milestones). Component specs
are in `docs/specs/`, harness facts in `docs/reference/harness_facts.md`.

## Development setup (CPU)

The competition harness wheels are not on PyPI; put them in `vendor/` first
(`swegemma-0.2.11`, `adk_submission-0.2.13`, `adk_eval_core-0.1.0`).

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python vendor/*.whl "google-adk>=1.34,<2" "mcp>=2" \
    "transformers==5.13.1" peft safetensors
uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q          # CPU tests; GPU/HF/vLLM tests are marked and skipped
```

GPU steps run from `notebooks/` on Kaggle, Colab or the training node (see `docs/PLAN.md` §1b).

## Reproducing the paper

To be completed: one command per row of the main results table (see `docs/specs/08_experiments.md`).

## License

Apache-2.0.
