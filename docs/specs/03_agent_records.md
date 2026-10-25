# Spec 03 — Agent records (Stage 1) by replaying teacher traces through the real harness stack

## Purpose

Turn each cleaned teacher trajectory into training examples that are token-for-token what the
deployed agent would see and produce under **our** agent config and budget. This spec also
defines the record format used by every later stage.

## 1. Method: scripted replay (chosen design)

Do not hand-build chat messages. Instead, run the trajectory through the same software stack the
harness uses, with the model and the tools replaced by recordings:

```
ADK Runner + App(events_compaction_config = harness values)
  agent  = compile_submission(our agent YAML)            # real adk_submission compiler
  model  = LiteLlm(api_base = scripted server)            # real LiteLLM + reasoning patch installed
  tools  = harness tool declarations, outputs replayed    # same names/schemas as swegemma tools
scripted server (spec 02 proxy in "scripted" mode):
  - records every request (messages, tools) = the exact training context of that turn
  - answers with the teacher's next turn as an OpenAI response
      (reasoning_content = thinking, content = text, tool_calls = calls)
  - forwards summarizer requests (compaction) to a real vLLM endpoint
```

What this gives for free, and exactly as at inference: system message assembly by ADK, thought
history handling (`include_thoughts`), tool-result serialization, parallel tool calls, nudges,
compaction (trigger, summary, retained events), and sub-agent (`agent_tool`) request/response
streams for spec 04.

### Requirements for the replay

- **Tool declarations identical to the harness.** Compare the `tools` array of a replayed request
  with the captured golden request (spec 02); must be equal.
- **Tool outputs** come from the trace (`tool_response.result`), matched to calls by order within
  the trajectory; a call/argument mismatch raises.
- **Budget behaviour from real harness code where possible.** Prefer running the real
  `budget_gated` wrapper with a `SwegemmaContext`-compatible object whose clock is virtual (see §3),
  so `budget_warning` and `get_status` come from harness code. If that is not practical, generate
  them in `src/locfix/harness/budget.py` and test against the harness functions
  (`swegemma/tools/base.py::_attach_low_budget_warning`, `swegemma/tools/execution.py::get_status`).
- **Initial user message** from the harness's own `build_agent_prompt(task, config,
  workspace_tree, declared_tools=...)` with our `eval_config` (see §2).
- **Nudges:** when the teacher's trace contains a `continuation_nudge`, the teacher's previous turn
  ended without a tool call; reproduce it by returning a text-only response, and let the runner
  produce the nudge (the texts are in `harness_facts.md` §6).
- **Determinism:** summarizer calls use fixed sampling and seed; record the summarizer model and
  adapter used.

### Grouping trace entries into model turns

A model turn is a run of consecutive `thinking` / `text` / `tool_call` entries produced by one
model call (the first entry of a turn carries `usage`; entries of one turn share a timestamp).
Several `tool_call` entries in one turn are parallel calls. `final` is the last turn's text.
Raise if the grouping is ambiguous.

## 2. Prompt and budget rewrite

- `eval_config` values (time, tool calls, turns) come from `configs/defaults.yaml`
  (`budget.*`), finalized by the Kaggle timing runs (spec 07).
- `task` fields: `repo`, `problem_statement`, `hints_text` (empty for the teacher data; see VERIFY
  8 in spec 02), `base_commit`.
- `workspace_tree`: reuse the block from the teacher's `task_prompt` (same command, same snapshot).
- **Regression test:** calling `build_agent_prompt` with the teacher's own budget (7.0 or 8.0 min,
  100 calls, 300 turns), the teacher's tools and the extracted tree must reproduce the teacher's
  `task_prompt` byte for byte.
- System instruction: the agent YAML's instruction (for the single agent: the teacher's system
  prompt, saved as `agents/single/prompts/system.md`); ADK may add text around it — whatever the
  replay captures is what we train on.

## 3. Virtual clock and time projection

Gemma is slower than the teacher's API model, so time-related fields and filters use a projected
Gemma timeline, not the teacher's timestamps:

```
t_turn  = prefill_new_tokens / prefill_tps + generated_tokens / decode_tps
t_tool  = teacher tool time (tool_response.elapsed_s − tool_call.elapsed_s; parallel calls: max)
clock   = Σ t_turn + Σ t_tool
```

`prefill_tps` and `decode_tps` per model come from the Kaggle timing runs (spec 07); tokens are
counted with the Gemma tokenizer on the replayed requests. `get_status` time fields and the
time filter use this clock.

### Compaction in the replay (implemented)

ADK decides to compact from the `prompt_tokens` the server reports, so the scripted server reports
a prompt-token count per request: `estimate_prompt_tokens` (4 chars/token) on CPU for tests, and
the real Gemma tokenizer (chat template applied to messages + tools) for real record builds, so
compaction fires at the same point as under vLLM. The summary request goes to a summarizer:
`static_summarizer` in tests, `forwarding_summarizer(base_url, model)` (vLLM with the base model
or the current adapter, temperature 0, seed 0) for real builds.

## 4. Filters (applied before replay; counts reported per source dataset)

| Filter | Rule |
| --- | --- |
| Tool-call budget | counted calls (`run_command`, `read_file`, `edit_file`, `write_file`) > `budget.max_tool_calls` → drop |
| Time | projected clock > `budget.max_time_minutes` × 0.9 → drop |
| Dev-set overlap | drop tasks from the dev repositories (FastAPI, Requests, Rich, HTTPX — any fork/mirror name) so dev scores stay honest |
| Held-out eval set | drop tasks reserved for the held-out eval set (spec 07) |
| Cleaning v2 | drop trajectories marked dropped by the patch-evidence audit |
| Context | any replayed request > 32,768 tokens after compaction → drop and report |

## 5. Loss masks and weights

Per assistant turn: `train` (bool) and `weight` (float, default 1.0).

| Case | Treatment |
| --- | --- |
| Normal assistant turn | `train: true`, weight 1 |
| Repeated call with identical arguments and identical output to an earlier call, no thinking in the turn (e.g. the teacher's `read_file` with an ignored `end_line`) | in context, `weight: 0` |
| Reference-seeking call marked by cleaning v2 (looking for an installed/original copy of the code) | in context, `weight: 0` |
| Summarizer output (compaction summary) | context only, never trained |
| Turns that only exist because of harness nudges | trained (the model must learn to continue) |

## 6. Training layout (implemented, `src/locfix/render/chat.py`)

Rules that make every trained turn's tokens identical to what the model generated at that point:
1. a tool-call turn's visible text is folded into its thinking (`text_merged` in turn meta);
2. a record ends before any new user message (nudge, compaction summary), because the template
   drops thinking that precedes the last user message;
3. targets are the generated text: thought channel + tool calls + the `<|tool_response>` stop
   token, or thought channel + text + `<turn|>` for the record's final text turn.
`render_record` raises `RenderError` if any of this does not hold.

Original plan text:

Decided by the prefix-consistency test (spec 02 §3):
- **consistent:** one record per replayed session, containing the final request + final response,
  loss on all assistant turns; when compaction happens, one record per compaction window (the
  request right before each compaction, plus the turns of that window).
- **not consistent:** one record per assistant turn (its exact request + its response). Costlier;
  `tools/gpu_hours_estimate.py` must be updated if this applies.

## 7. Record format

`schemas/record.schema.json` (JSON Schema). Summary:

```json
{
  "id": "s1__<instance_id>__a<attempt>__w<window>",
  "stage": "s1_agent",
  "role": "agent",
  "instance_id": "...",
  "source": {"set": "solved", "dataset": "swe_smith", "fold": "A", "attempt": 1},
  "agent_config": "agents/single/agent.yaml",
  "tools": [ ... exact `tools` array from the replayed request ... ],
  "messages": [ ... exact `messages` array, OpenAI format, plus the final assistant response ... ],
  "loss": [{"message_index": 3, "train": true, "weight": 1.0}, ...],
  "weight": 1.0,
  "spans": null,
  "meta": {"n_counted_calls": 13, "projected_seconds": 211.4, "compaction_window": 0,
           "trace_entry_ids": {...}, "budget": {...}, "summarizer": "...",
           "cleaning": "v2", "builder_version": "..."}
}
```

Records are stored untokenized (one file serves E4B, 12B and 31B); tokenization and the loss mask
are produced by `src/locfix/render/chat.py`, which the golden tests cover.

## 8. Outputs and statistics

`locfix data agent-records --config ...` writes `records/s1_agent/{train,val}.jsonl` (val: ~3% of
tasks, by task, stratified by source) and `manifest.json`: counts per source before/after each
filter, token totals (Gemma tokenizer) per model, distribution of counted calls and projected
time, number of compaction windows.

## Acceptance tests

- Regression: teacher prompt reproduced byte for byte with the teacher's budget (§2).
- Replay of the four sample traces in `context/` (after copying to `tests/fixtures/`) completes;
  every recorded tool output equals the trace's; in the exceptiongroup trace the second
  `read_file` (entry 5, `end_line: 598` ignored, identical output to entry 3) gets weight 0, while
  arrow's second `read_file` (`start_line: 150`, new content) keeps weight 1.
- Budget warning test: with `max_tool_calls: 22`, calls 12–22 carry the warning text exactly as the
  harness produces it; `submit_patch`/`get_status` do not count.
- Golden test: a replayed session of a task also captured live (spec 02, session A) produces the
  same request structure (system message, tools, message roles and serialization) as the live one.
- Compaction: replaying a trajectory longer than 14,336 tokens produces a compaction whose request
  layout matches capture D.
