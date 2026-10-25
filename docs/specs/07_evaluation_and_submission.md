# Spec 07 — Evaluation, timing and submission packaging

## 1. Evaluation sets

| Set | Size | Source | Use |
| --- | --- | --- | --- |
| `dev` | 129 | competition `tasks.jsonl` + snapshots (FastAPI, Requests, Rich, HTTPX) | closest to the hidden test; reported in the paper |
| `heldout` | ~300 | validated tasks never attempted by the teacher (~935 available), stratified by source; never used in training | main comparison set (more statistical power) |
| `train_rollouts` | training tasks | for Stage 5 and the signal test | not a score |

The `heldout` tasks run with the task's own Docker image through the existing generation
infrastructure in `khalil-Hennara/codeagent-validate` (same swegemma agent loop), pointed at our
local vLLM endpoint. Dev tasks run with `swegemma eval`.

## 2. Runner (`locfix eval run`)

- Starts vLLM with the **harness flags** (`tensor_parallel_size` as fits, `max_model_len=32768`,
  `tool_call_parser=gemma4`, `reasoning_parser=gemma4`, reasoning config, `enable_lora`,
  `max_loras` = number of adapters, `max_lora_rank` = max adapter rank, 12B hf-overrides) and the
  submission's adapters.
- Builds a submission directory with `src/locfix/submit/package.py` (§5) and evaluates **that
  directory**, so what we evaluate is exactly what we would submit.
- Uses the submission's `eval_config.yaml` budgets.
- Concurrency limited by CPU/RAM (each sandbox: 2 vCPU, 4 GB).
- Writes per-task JSONL (resolved, error, time, counted tool calls, tokens, nudges, whether
  `submit_patch` was called, whether a timeout fallback patch was used) and `summary.json`.

## 3. Statistics (`src/locfix/eval/stats.py`)

- Resolve rate with Wilson 95% CI.
- Paired comparison of two systems on the same tasks: exact McNemar test; report the discordant
  counts.
- Two sampling seeds for final comparisons; report mean and both values.
- Note for the paper: at ~40% resolve, 129 tasks give SE ≈ 4.3 points (95% CI ≈ ±8.5), so dev-only
  differences under ~8 points are not significant on their own.

## 4. Timing and the Kaggle budget

The hidden set (~120 tasks) runs **sequentially** within 12 hours, including sandbox setup, on
4×L4. The session timeout is a hard kill (`asyncio.timeout`); edits already in the working tree are
still graded. The only "wrap up" signal is `budget_warning`, added when ≤ 10 counted calls remain
(and the limit is ≥ 20).

Defaults (`configs/defaults.yaml`): `max_time_minutes: 5`, `max_tool_calls: 30`, turns unchanged.

`locfix eval timing` on a Kaggle L4×4 notebook (internet off, competition data attached):
- measure vLLM startup time, per-task sandbox setup time, prefill and decode tokens/s for each
  model, per-tool-call wall time;
- outputs `timing.json`, used by spec 03 (time projection and filters) and to check that
  `N_tasks × (setup + cap) + startup` stays under 12 h with margin.

## 5. Submission packaging (`locfix submit package`)

Produces `submission/` and `submission.zip`:
- `agent.yaml` (+ `sub_agents/`, `prompts/`, `configs/sampling.yaml`, `eval_config.yaml`) from
  `agents/<structure>/` templates with the chosen model alias and adapter names;
- `adapters/<name>/adapter_config.json` + `adapter_model.safetensors`, each with a passing
  `parity.json` (refuse otherwise; `parity.json` itself is not packaged);
- checks: one root config; one model for all agents, from the allowed list; allowed file
  extensions only (`.yaml .yml .md .txt .py .json .safetensors`); no symlinks; total unpacked
  size < 3 GiB; adapter names match `[A-Za-z0-9][A-Za-z0-9._-]*`; `max_lora_rank` ≤ 128;
- runs the official validator from the vendor wheels if available.

## 6. Stage 0 (base model, no training)

On E4B and 31B, dev + heldout:
1. your current single-agent config without graph tools;
2. the same with graph tools (prompt includes the Code Intelligence section);
3. fixer + localizer `agent_tool` structure with base weights;
4. `include_thoughts` true vs false for the single agent.

These give the reference numbers, decide `include_thoughts`, and validate the runner.

## Acceptance tests

- `locfix eval run` on 10 dev tasks with the base model reproduces your earlier Kaggle-style
  results within noise.
- Packaged submission passes the vendor validator and the size/extension checks; a deliberately
  broken one (wrong model, symlink, 3.1 GiB) is refused.
- `timing.json` exists for E4B and 31B and the 12-hour check passes for the chosen budget.
