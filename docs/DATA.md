# Data: from teacher traces to training records

This part of the repository turns the teacher trajectories (Kimi-K3 runs in the competition
harness) into training records for Gemma 4. It is a two-step design:

```
 codeagent_final/cleaned/                 STEP 2: build records (once, CPU, needs the harness wheels)
   solved.jsonl  ─┐                        locfix data.agent-records   ──►  s1_agent.jsonl
   traces/*.json ─┼──────────────────────►                                   (single agent)
 code/<id>.json  ─┘                        locfix data.locfix-records  ──►  s2_*.jsonl, s3_loc_r0.jsonl
   (optional)                                                                 (localizer + fixer)
                                                       │
                                                       ▼
                                           STEP 3: check (CPU, tokenizer only)
                                           locfix data.check-records  ──►  report + HTML previews
                                                       │
                                                       ▼
                                           STEP 4: trainer dataloader (every training run, GPU node)
                                           record ─► Gemma 4 chat template ─► input_ids / labels / loss_weights
```

- **Step 2 runs once.** It replays every trace through the real harness code (ADK, LiteLLM, the
  harness's own tools and prompt builder, our agent YAML) with a scripted model that repeats the
  teacher's turns and recorded tool outputs. The output records hold the chat messages exactly as
  the model will receive them on Kaggle. This is the only step that needs the harness wheels.
- **Steps 3 and 4 only need `transformers==5.13.1` and the Gemma 4 tokenizer files.** Step 4 is the
  PyTorch `Dataset`/collator of the trainer (spec 05); it uses the same `render_record` as step 3.

Why a replay instead of formatting the traces directly: the trace is a log of what happened, not
the request the model was sent, and our setting differs from the one the traces were made in.
The replay adds what the log does not contain (tool schemas, ADK's system-prompt line, message
structure), rewrites the budget (prompt, `budget_warning`s, `get_status`) from the teacher's
8 min / 100 calls to ours, and builds the two-agent conversations (fixer calling the localizer
as a tool) that never happened in the harness.

---

## 1. Inputs

From the data pipeline's `codeagent_final/` folder (see `context/DATA_PIPELINE.md`):

| Input | Used fields | Needed by |
| --- | --- | --- |
| `cleaned/traces/<instance_id>__a<N>.json` | `entries`: `system_instruction`, `task_prompt`, `thinking`, `text`, `tool_call` (`tool`, `args`), `tool_response` (`result`), `final`, `continuation_nudge`; optional `train_weight` marks from cleaning | all steps |
| `cleaned/solved.jsonl` | `instance_id`, `problem_statement`, `verifier` (→ `source.dataset`), `fold`, `teacher.patch` | step 2 |
| `code/<instance_id>.json` | `{path: file text}` at the base commit (bug applied) | two-agent records only |
| harness wheels | `swegemma-0.2.11`, `adk_submission-0.2.13`, `adk_eval_core-0.1.0` (+ `google-adk>=1.34,<2`, `litellm`, `mcp>=2`) | step 2 |
| tokenizer files | `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` of `gemma-4-*-it-qat-w4a16-ct` (31B and E4B: identical `tokenizer.json`; the chat templates differ only when thinking is off, and our records always have thinking on) | steps 3, 4 |

**Code files.** The two-agent records need the full files at the base commit, because the fixer
re-reads the locations the localizer returns. `code/<id>.json` exists for the unsolved set
(`experiment/code/`). For solved tasks, ask the data pipeline to write `cleaned/code/<id>.json`
with the same extraction. Without a code file, the code is rebuilt from the lines the teacher
printed before its first edit (`read_file` outputs, plain `sed -n 'a,bp' FILE` and `cat FILE`);
records whose reads would touch a line the teacher never printed are dropped (`unseen_code`).
Each record says which source it used (`meta.code_source`: `code_json` or `trace`).

## 2. Setup

```bash
# Python >= 3.12 (tested on 3.12 and 3.13). Only the three harness wheels are needed, not the
# whole Kaggle wheelhouse; --no-deps skips their torch/torchvision/accelerate requirements,
# which the data code does not use.
pip install --no-deps swegemma-0.2.11-py3-none-any.whl adk_submission-0.2.13-py3-none-any.whl \
    adk_eval_core-0.1.0-py3-none-any.whl
pip install "google-adk==1.39.1" "litellm==1.106.3" "transformers==5.13.1" docker cachetools \
    networkx python-dotenv rich pandas safetensors pyyaml jsonschema
pip install "mcp==2.3.0"     # litellm 1.106.3 needs mcp 2.x; google-adk declares mcp<2 but works
                             # with it. pip prints a dependency-conflict message here: expected.
pip install -e .             # this repository (or put src/ on PYTHONPATH)
```

For steps 3 and 4 alone: `pip install transformers==5.13.1 pyyaml jsonschema` and the code.
Machine-specific paths can go in `configs/paths.yaml` (copy `configs/paths.example.yaml`).

Every command is `locfix <subcommand>`, accepts `--config <run.yaml>` (repeatable) and
`--set key=value` (repeatable), and writes a `run_meta.json` (resolved config, git SHA, package
versions, counts) next to its output.

## 3. Step by step

### Step 1: statistics (choose the budget, the maximum length and the batching)

```bash
# traces only: fast, no harness wheels (token counts approximate, within ~1.5%)
locfix data.stats --traces codeagent_final/cleaned_v2/traces --solved codeagent_final/cleaned_v2/solved.jsonl \
    --tokenizer <tokenizer dir> --out work/stats/traces

# records: exact token counts, per stage, and the multi-GPU step-time simulation
locfix data.stats --records work/records/s1_agent.jsonl work/records/locfix --tokenizer <tokenizer dir> \
    --out work/stats/records --world-size 8 --global-batch 16 [--tokens-per-s <measured>]
```

| | |
| --- | --- |
| Input | traces (+ `solved.jsonl` for the dataset of each trace) and/or record files; the Gemma 4 tokenizer |
| Output | `stats.md` (the report), `stats.json`, one row per trace / record (`traces.jsonl`, `records.jsonl`), histograms (PNG, if matplotlib is installed) |
| Traces part | distributions (counted tool calls, turns, exploration before the first edit, nudges, parallel calls, teacher time, approximate tokens, largest tool output, thinking tokens), per dataset; **tool-call budget sweep** (traces kept at 16…100 calls), **exploration sweep** (`em.localizer_max_calls`), **length sweep**, budget × length grid |
| Records part | per stage: records, tasks, token percentiles, trained share, requests; **length sweep per stage** (records and trained tokens kept at 8K…32K); where the tokens go (system, task, tool outputs, thinking, calls); largest tool output / trained turn; **step-time simulation**: GPU efficiency with one record per row in random order, with `train.length_grouping`, and with `train.packing` |

Records carry the budget they were built with. To see every trajectory regardless of the budget,
build them once with a large budget (`--set budget.max_tool_calls=100 --set
budget_locfix.max_tool_calls=100`), read the statistics, then rebuild with the chosen budget.
(`locfix data.traces-stats` is the older, smaller summary.)

### Step 2a: single-agent records (Stage 1)

```bash
locfix data.agent-records codeagent_final/cleaned/traces \
    --solved codeagent_final/cleaned/solved.jsonl \
    --out work/records/s1_agent.jsonl
```

| | |
| --- | --- |
| Input | traces folder, `solved.jsonl`; agent `agents/single/`; budget `budget.*` |
| Output | `s1_agent.jsonl` (one record per segment of a trajectory, see §4) and `run_meta.json` with `counts`: `traces`, `records`, `skip_dev_repo`, `skip_over_budget`, `error:<type>` |
| Time | about 0.8 s per trace (≈ 20 min for 1,577 traces, one process) |
| Options | `--limit N` (first N traces) |

### Step 2b: localizer and fixer records (Stages 2 and 3, round 0)

```bash
locfix data.locfix-records codeagent_final/cleaned/traces \
    --solved codeagent_final/cleaned/solved.jsonl \
    --code codeagent_final/cleaned/code --code codeagent_final/experiment/code \
    --out work/records/locfix
```

| | |
| --- | --- |
| Input | traces, `solved.jsonl`, code folders (searched in order); agent `agents/locfix/` (fixer root + `localizer` as `agent_tool`); budget `budget_locfix.*`; `em.*` settings |
| Output | one file per stage (table below), `manifest.json` (counts) and `run_meta.json` |
| Time | about 3 s per task (one replay per variant of z; ≈ 80 min for 1,577 tasks) |
| Options | `--code DIR` (repeatable), `--no-code-fallback` (drop tasks without a code file instead of rebuilding code from the trace), `--limit N` |

Each trace is split at the teacher's **first edit**. The **localizer** conversation is the
teacher's exploration before it (only `run_command`/`read_file` calls), ending with an answer z, a
JSON list of locations. The **fixer** conversation is: call `localizer(request=...)` → read the
locations of z → (if z misses something) read it or reuse a teacher `grep` → the teacher's
trajectory from its first edit to `submit_patch` and the final text. Several versions of z are
built per task:

| Variant | z | Localizer records | Fixer records |
| --- | --- | --- | --- |
| `gold` | what the teacher edited ∪ viewed, pruned to `em.max_locations` (8) and `em.ctx_budget_tokens` (6K) | `s3_loc_r0.jsonl` | `s2_fix_gold.jsonl` (no requests) |
| `edit_only` | only the edited symbols | `s2_loc_edit.jsonl` | `s2_fix_noisy.jsonl` (with requests) |
| `view_only` | what the teacher viewed (edited ones marked `edit`) | `s2_loc_view.jsonl` | `s2_fix_noisy.jsonl`, only if z differs from `gold` |
| `gold_minus_one` | `gold` minus one context location | — | `s2_fix_noisy.jsonl` |

In `em.no_request_fraction` (40%) of the noisy variants whose z still covers every edited symbol,
requests are switched off, so the fixer also learns to work from z alone.

`manifest.json` keys: `traces`; `code:code_json` / `code:trace` (code source per task);
`records:<stage>`; `with_requests:<stage>`; `skip:<reason>` (whole task dropped);
`drop:<variant>:<reason>` (one variant dropped); `error:<type>`. Reasons are listed in §6.

### Step 3: check the records

```bash
locfix data.check-records work/records/s1_agent.jsonl work/records/locfix \
    --tokenizer /path/to/tokenizer_files --preview work/records/preview
```

| | |
| --- | --- |
| Input | record files or folders; tokenizer folder |
| Output | per file: `records`, `rendered`, `errors`, `tokens_median`, `tokens_max`, `trained_tokens_total`, `trained_share`, `over_32768`, stages, roles. With `--preview DIR`, one HTML page per file showing a random record as training text (grey = context, green = trained). Exit code 1 if any record fails the schema or rendering |
| What it checks | every record against `schemas/record.schema.json`; rendering with the Gemma 4 chat template exactly as vLLM does (prefix consistency, token boundaries of every trained span) |
| Options | `--sample N` (render N random records per file), `--max-tokens` |

### Step 4: what the trainer gets (spec 05)

For each record, `locfix.render.chat.render_record(record, tokenizer)` returns:

| Field | Meaning |
| --- | --- |
| `text` | the full training text (Gemma 4 chat template, as vLLM renders it) |
| `input_ids` | its token ids |
| `labels` | the token id where the model is trained, `-100` elsewhere (prompts, tool outputs, system text) |
| `weights` | loss weight per token (turn weight × record weight; 0 where not trained) |

Loss: Σₜ wₜ·CE(token t) / Σₜ wₜ over all trained tokens of a batch (across GPUs and accumulation
steps). The renderer was checked against vLLM 0.19.1 on Kaggle: 141/141 prompts token-identical
(`notebooks/golden_render_check_standalone.py`). Typically 12–22% of a record's tokens are trained.

## 4. Record format

One JSON object per line (`schemas/record.schema.json`):

| Field | Content |
| --- | --- |
| `id` | `<stage>__<instance_id>__a<attempt>__w<window>[__<variant>]` |
| `stage` | `s1_agent`, `s2_fix_gold`, `s2_fix_noisy`, `s2_loc_edit`, `s2_loc_view`, `s3_loc_r0`, … |
| `role` | `agent`, `fixer` or `localizer` (which adapter trains on it) |
| `instance_id`, `source` | task id; `{set, dataset, fold, attempt}` (`fold` A/B is used for cross-fitting in Stage 3) |
| `agent_config` | the agent folder the record was replayed with |
| `tools` | the tool schemas exactly as LiteLLM sends them (fixer: 6 harness tools + `localizer`; localizer: `run_command`, `read_file`) |
| `messages` | OpenAI chat messages exactly as sent to the model, ending with the last trained assistant message. Assistant messages carry `reasoning` (thinking), `tool_calls`, `content` |
| `loss` | `[{message_index, train, weight, reason?}]`: which assistant messages are trained; others are context |
| `weight` | record weight (1 now; EM weights in Stage 3) |
| `z` | localizer/fixer records: the locations, `[{path, symbol, lines: [start, end], role: edit/context}]` |
| `meta` | `request_params` (`chat_template_kwargs`, …), `budget`, `trace`, `turns`, counts of tool calls; locfix records also `variant`, `l_edit`, `l_view`, `requests`, `unrecovered_context`, `dangling_refs`, `code_source` |

Shortened fixer record (`s2_fix_noisy`):

```json
{"id": "s2_fix_noisy__benoitc__gunicorn...__a1__w0__edit_only", "stage": "s2_fix_noisy", "role": "fixer",
 "messages": [
  {"role": "system", "content": "You are an expert software engineer ...\n\nYou are an agent. Your internal name is \"fixer\"."},
  {"role": "user", "content": "You are evaluating a software engineering task ... Tool calls allowance: 40 calls ..."},
  {"role": "assistant", "reasoning": "I'll start by asking the localizer for the code relevant to this issue.",
   "tool_calls": [{"function": {"name": "localizer", "arguments": "{\"request\": \"Find the code locations relevant to this issue.\"}"}}]},
  {"role": "tool", "content": "{\"result\": \"{\\\"locations\\\": [{\\\"path\\\": \\\"gunicorn/workers/base.py\\\", ...}]}\"}"},
  {"role": "assistant", "reasoning": "The localizer returned 1 location(s). I'll read them.",
   "tool_calls": [{"function": {"name": "read_file", "arguments": "{\"filepath\": \"/workspace/gunicorn/workers/base.py\", \"start_line\": 86, \"end_line\": 114}"}}]},
  {"role": "tool", "content": "{\"status\": \"ok\", \"filepath\": \"/workspace/gunicorn/workers/base.py\", \"content\": \"    def init_process(self): ...\"}"},
  "... the teacher's edits, test runs, submit_patch, final text ..."],
 "loss": [{"message_index": 2, "train": true, "weight": 1.0}, {"message_index": 4, "train": true, "weight": 1.0}, "..."],
 "z": [{"path": "gunicorn/workers/base.py", "symbol": "Worker.init_process", "lines": [86, 114], "role": "edit"}]}
```

## 5. What the builder changes and what it never changes

Never changed: the teacher's tool-call arguments, tool outputs and patches (rule 3 of
`CLAUDE.md`). Changed or added, always with the harness's own functions where one exists:

| What | Why |
| --- | --- |
| Task prompt rebuilt with our budget (`build_agent_prompt`) | the model must learn under the budget it will have |
| `budget_warning` in tool outputs recomputed; the teacher's removed | warnings appear when ≤ 10 calls remain under *our* budget |
| `get_status` outputs recomputed (calls used, time) | same |
| Visible text of a tool-call turn moved into its thinking | Gemma 4's template puts that text after the tool outputs; keeping it there would train an order the model never produces |
| A turn that only repeats an identical earlier call (same output) gets weight 0; cleaning v2 `train_weight` marks are applied | do not teach wasted calls |
| A nudge (new user message) starts a new record | the template drops earlier thinking after a user message; one record must render as one prefix-consistent text |
| Two-agent split, synthetic turns with fixed short thinking (`TEMPLATES` in `loc_fix_records.py`) | the deployed structure; templated text avoids writing new reasoning with an LLM |
| Fixer reads of z rendered by the harness's `read_file` on the base-commit code | identical to what the harness would return |

Compaction is off by default when building records (`data.replay_compaction: false`); the
compaction path exists and is tested but needs a summarizer (spec 03).

## 6. Why records are dropped

| Reason | Step | Meaning |
| --- | --- | --- |
| `skip_dev_repo` / `skip:dev_repo` | 2a, 2b | repository of the dev set (`data.exclude_repos`) |
| `skip_over_budget` / `drop:<variant>:over_budget` | 2a, 2b | needs more counted tool calls than the budget (2b: localizer + fixer calls together) |
| `skip:not_in_solved` | 2b | trace without a row in `solved.jsonl` |
| `skip:no_edit` | 2b | trajectory never edits a file |
| `skip:mixed_first_edit_turn` | 2b | the first edit shares a turn with exploration calls |
| `skip:missing_code` | 2b | no code for an edited file (no code file, and the teacher never printed it in a usable form) |
| `skip:empty_l_edit` | 2b | the patch touches nothing that maps to a code location |
| `skip:no_code_file` | 2b | no code file and `--no-code-fallback` |
| `drop:<variant>:localizer_over_calls` | 2b | exploration longer than `em.localizer_max_calls` (15) |
| `drop:<variant>:impossible_input` | 2b | z misses an edited location and nothing recovers it |
| `drop:<variant>:unseen_code` | 2b | code rebuilt from the trace and a read needs lines the teacher never printed |

## 7. Settings that affect the data

In `configs/defaults.yaml`; change them with a run config (`--config configs/runs/<name>.yaml`)
or `--set`. They must match the submission's `agents/*/eval_config.yaml`.

| Key | Default | Effect |
| --- | --- | --- |
| `budget.max_time_minutes`, `budget.max_tool_calls`, `budget.max_turns` | 5, 40, 300 | single-agent budget (prompt, warnings, `get_status`, over-budget drops) |
| `budget_locfix.*` | 5, 48, 300 | the same for the two-agent records (+8 calls: the fixer re-reads the localizer's locations) |
| `data.teacher_time_minutes`, `data.time_scale` | 8, auto | clock shown by `get_status`: teacher's elapsed time × scale; `auto` = budget minutes / teacher minutes (0.625), so the teacher's used share of its time is kept |
| `data.exclude_repos` | fastapi, requests, rich, httpx | repositories of the dev set |
| `data.training_layout` | auto | `auto`/`one_sequence`: one record per segment; `per_turn`: one record per model call |
| `data.replay_compaction` | false | replay with the harness's context compaction |
| `em.max_locations`, `em.ctx_budget_tokens` | 8, 6000 | size of z |
| `em.localizer_max_calls` | 15 | longest localizer exploration kept |
| `em.no_request_fraction` | 0.4 | share of noisy fixer records without requests |
| `seed` | 1234 | variant choices (per task: `seed:instance_id`) |

Example, 8 minutes and 28 tool calls (`configs/runs/budget_8m_28.yaml`):

```yaml
budget:        {max_time_minutes: 8, max_tool_calls: 28, max_turns: 300}
budget_locfix: {max_time_minutes: 8, max_tool_calls: 28, max_turns: 300}
```

```bash
locfix data.agent-records ... --config configs/runs/budget_8m_28.yaml
locfix config --config configs/runs/budget_8m_28.yaml      # print the resolved settings
```

and set the same numbers in `agents/single/eval_config.yaml` / `agents/locfix/eval_config.yaml`.

## 8. Looking at the data

`notebooks/inspect_training_data.ipynb` (Colab, CPU): picks a random task and shows trace →
record → highlighted training text → tokens and labels, for the single agent and for the
localizer/fixer. Set `RECORDS_DIR` to the output folder of step 2 to read existing records
(no harness wheels needed), or leave it empty to build the records of a random task live.

## 9. Code map

| Module | Role |
| --- | --- |
| `src/locfix/data/traces.py` | load and validate traces; turns, tool calls, first edit; `data.traces-stats` |
| `src/locfix/harness/prompts.py` | parse the recorded task prompt and rebuild it with `build_agent_prompt` |
| `src/locfix/harness/budget.py` | budget warnings and `get_status` with the harness's functions (`BudgetWalker`) |
| `src/locfix/harness/scripted_server.py` | local OpenAI-compatible server that answers with scripted turns |
| `src/locfix/harness/replay.py` | the replay: real ADK runner + compiled agent YAML + harness tool signatures; one script per agent |
| `src/locfix/data/agent_records.py` | Stage 1: trace → script → replay → records; `data.agent-records` |
| `src/locfix/data/loc_fix_records.py` | Stages 2/3: split, variants of z, localizer and fixer scripts, requests; `data.locfix-records` |
| `src/locfix/data/dataset.py` | `SolvedData`: the `codeagent_final` layout, one `Task` per trace, code lookup and fallback |
| `src/locfix/data/locations.py` | symbol index (AST), L_edit from a patch, L_view from the teacher's reads |
| `src/locfix/data/zloc.py` | z: JSON format, union, pruning to the size limits |
| `src/locfix/data/render_obs.py` | synthetic `read_file` outputs with the harness's own `read_file` |
| `src/locfix/data/code_from_trace.py` | base-commit code rebuilt from what the teacher printed |
| `src/locfix/data/edits.py` | unified diffs; patch → `edit_file` calls (validated by the harness's `edit_file`) |
| `src/locfix/harness/local_sandbox.py` | runs the harness's `read_file`/`edit_file`/`write_file` on a local folder |
| `src/locfix/render/chat.py` | record → Gemma 4 text → tokens, labels, weights (`render_record`) |
| `src/locfix/render/inspect.py` | readable views of traces/records; `data.check-records` |
| `schemas/record.schema.json` | the record format |

Tests: `tests/test_replay_records.py`, `tests/test_loc_fix_records.py`,
`tests/test_dataset_fallback.py`, `tests/test_render.py`, `tests/test_inspect.py` run on four real
fixture tasks (`tests/fixtures/`). Specs with the design details: `docs/specs/03_agent_records.md`
and `docs/specs/04_localizer_fixer_records.md`.
