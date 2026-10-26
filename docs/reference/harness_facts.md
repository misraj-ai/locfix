# Harness facts (verified from source)

Inspected: `swegemma-0.2.11`, `adk_submission-0.2.13`, `adk_eval_core-0.1.0` (wheels from the
competition), `google-adk` 1.39.1 (latest in `>=1.34,<2` on 2026-10-24), and the official
`HARNESS_README.md` (in `context/`). File references are inside the wheels. When an answer comes
from a live capture (spec 02), add it here with the evidence file.

## 1. Allowed models and serving

- Allowed base models: `gemma-4-31b-it-qat-w4a16-ct`, `gemma-4-12b-it-qat-w4a16-ct`,
  `gemma-4-e4b-it-qat-w4a16-ct`, `gemma-4-e2b-it-qat-w4a16-ct`. One model for all agents.
- vLLM on 4×L4: `tensor_parallel_size=4`, `gpu_memory_utilization=0.90`, `max_model_len=32768`,
  `enable_auto_tool_choice=True`, `tool_call_parser="gemma4"`, `reasoning_parser="gemma4"`,
  `reasoning_config={"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}`,
  `enable_lora = bool(adapters)`, `max_loras = max(1, n_adapters)`,
  `max_lora_rank = max(adapter ranks) or 128`. 12B adds
  `--hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'`.
- Adapters: `adapters/<name>/{adapter_config.json, adapter_model.safetensors}`, referenced by
  `adapter: <name>` on any `LlmAgent`; each agent's requests go to model `openai/<adapter_name>`.

## 2. Submission constraints

Total unpacked size < 3 GiB including adapters; allowed extensions `.yaml .yml .md .txt .py .json
.safetensors`; no symlinks or path traversal; declarative YAML only; `max_output_tokens` and
`thinking_budget` ≤ 32,768 (defaults 16,384 and 4,096). Agent classes: `LlmAgent`,
`SequentialAgent`, `ParallelAgent`, `LoopAgent`; sub-agents via `sub_agents` or
`agent_tool: {config_path, skip_summarization}`; skills allowed. `!include` paths resolve
**relative to the file containing the tag** (`adk_submission/yaml_loader.py`) and `..` is
rejected, so a YAML in `sub_agents/` must include files from `sub_agents/…`; `config_path` is
tried relative to the current file's directory first, then the submission root
(`adk_submission/context.py:98`). The hidden test runs tasks
sequentially (confirmed by the team) within 12 h on 4×L4, setup included, verification excluded.

## 3. Sandbox

Per task: a fresh Docker container (2 vCPU, 4 GB RAM, no network), repo at `/workspace` with a
`baseline` commit that includes harness-written `pytest.ini` and `conftest.py`. Agent time starts
after container setup.

## 4. Initial user prompt

`swegemma/harness/agent_runner.py:317`
`build_agent_prompt(task, config, workspace_tree='', *, enable_sandbox_testing=None, declared_tools=None)`.
Sections in order: header + problem statement; `## Hints:` (only if `hints_text`); `## Task Budget`
(`- Time allowance: {time_minutes} minutes`, `- Tool calls allowance: {N} calls`,
`- Max loop iterations: {N} turns`, cost if set); `## Execution Environment Rules` (the
`read_file` lines only if `read_file` is declared anywhere in the agent tree); `## Instructions:`
(0–5; item 3 depends on `enable_sandbox_testing`); `## Code Intelligence Tools` (only if graph and
embedding files exist for the repo **and** graph tools are declared); `## Workspace Layout`.
Declared tools are collected across the whole agent tree, including `agent_tool` sub-agents
(`_extract_declared_tools`, line 192). The time value is printed as given (`7.0`, `8.0`, `5`, …).

## 5. Budget accounting and `budget_warning`

- Counted tools (`@budget_gated`, default): `run_command`, `read_file`, `edit_file`, `write_file`
  (and the graph tools). **Not counted:** `submit_patch`, `get_status`
  (`count_tool_call=False`).
- `check_budget` (`swegemma/context.py:468`) increments the counter for counted calls and returns a
  `BudgetExceeded` error JSON when the tool-call, turn, cost or token budget is exhausted.
- `_attach_low_budget_warning` (`swegemma/tools/base.py:47`): only if `max_tool_calls >= 20` and
  `remaining = max_tool_calls − tool_calls_used ≤ 10`; adds to the tool's JSON response (if the
  key is absent) and re-serializes with `json.dumps(payload, ensure_ascii=False)`:
  `"budget_warning": "Only {remaining} tool call(s) remaining ({used}/{max} used). Finalize your edits and call submit_patch soon."`
- `get_status` (`swegemma/tools/execution.py:120`) fields, in order: `status`, `tool_calls_used`,
  `patch_submitted`, `patch_size`, then if set `tool_calls_remaining`, `max_tool_calls`; if a time
  budget is set `time_seconds_remaining`, `max_time_minutes`, `agent_elapsed_seconds`; if set
  `max_turns`; `command_timeout_seconds`.

## 6. Continuation nudges

`agent_runner.py` ~line 983. Sent as a user message when a turn ends without `submit_patch`; up to
3 consecutive nudges (reset after a turn with a valid tool call):
- truncated tool call (`<|tool_call>` in the last text): "Your previous response reached the token
  limit before the tool call finished closing (<|tool_call|> was cut off). Do NOT repeat your prior
  reasoning in thought—emit your next tool call immediately, and if calling edit_file or
  write_file, split the change into smaller incremental edits."
- finish reason `MAX_TOKENS`/`LENGTH`: "Your previous response reached the token limit while
  thinking before a tool call was completed. Do NOT repeat your analysis in thought—keep reasoning
  under a few sentences and emit your next tool call immediately, or call submit_patch when you
  have completed and verified your changes."
- otherwise: "Please continue your work using the available tools, or call submit_patch when you
  have completed and verified your changes."

## 7. Session end and patch extraction

- Time budget is a hard `asyncio.timeout(time_minutes × 60)` around the agent loop
  (`agent_runner.py` ~712); there is **no time warning**.
- After a timeout, turn limit, nudge limit or normal end without `submit_patch`, the harness runs
  `git add -N . && git diff --binary` and keeps the non-protected changes as the patch, which is
  still graded.
- Protected files stripped from patches: `test_*.py`, `*_test.py`, `.py` under `tests/`, `test/`,
  `testing/`, and runner configs (`conftest.py`, `pytest.ini`, `pyproject.toml`, `tox.ini`,
  `setup.cfg`, …).
- Resolved = pytest exit 0 and every required test node passed in JUnit XML.

## 8. Thoughts (`include_thoughts`)

`adk_submission/resolvers/generation.py:195` `install_litellm_reasoning_patch()`:
- assistant messages carry prior thoughts as `reasoning_content`, mirrored into `reasoning` (so
  vLLM keeps them across tool calls) unless `include_thoughts` is `false`;
- with `include_thoughts: false`, thought parts are removed from responses before they enter
  session history and `reasoning`/`reasoning_content` are stripped from outgoing messages
  (reasoning still happens on the current turn);
- tool parameter schemas with `any_of [T, null]` are flattened so Gemma 4's chat template gets a
  top-level `type`.
- Whether Gemma 4's chat template actually renders earlier `reasoning` is **VERIFY** (spec 02).

## 9. Open items (VERIFY in M0; fill in with evidence)

| # | Question | Answer |
| --- | --- | --- |
| 1 | Message fields and rendering of earlier thoughts | **From the chat template + vLLM source (2026-10-25):** ADK sends `reasoning_content` and `reasoning`; vLLM reads only `reasoning`. The template renders an assistant message's thinking only if it has tool calls **and** comes after the last user message, so a nudge or a compaction summary erases earlier thoughts; text-only turns never show their thinking in history. **Golden check passed** (Kaggle, vLLM 0.19.1, E4B, 2026-10-25): 63/63 prompts token-identical to `/v1/chat/completions/render`. Re-run with the localizer/fixer records added (2026-10-26): **141/141**. |
| 2 | Tool call / tool result serialization, parallel calls | LiteLLM sends arguments as JSON strings; vLLM `_postprocess_messages` parses them to dicts, and the template renders `<\|tool_call>call:name{key:<\|"\|>value<\|"\|>,...}<tool_call\|>` (keys sorted). Tool results (`role: tool`, JSON string content, unchanged from the harness) render inside the same model turn as `<\|tool_response>response:name{value:...}<tool_response\|>`. The whole trajectory is one model turn; a tool-call turn ends with the generated `<\|tool_response>` token. Visible text on a tool-call message is rendered **after** the tool responses. Content format detected by vLLM: `openai` (system prompt rendered with a trailing space). ADK appends `You are an agent. Your internal name is "<name>".` to the system instruction. |
| 3 | `agent_tool` schema seen by the root and result wrapper | **From ADK 1.39.1 + scripted replay of `agents/locfix` (2026-10-25):** the root sees a tool named after the sub-agent (`localizer`) with description = the sub-agent's `description` and parameters `{request: string}` (required). The result reaches the root as a tool message whose content is `{"result": "<sub-agent's final text>"}` (ADK wraps non-dict returns). See §15. |
| 4 | Sub-agent: prompt, compaction inside it, shared budget | **From ADK 1.39.1 + replay:** system = the sub-agent's instruction with `{problem_description}` filled from the copied state (a literal JSON example with braces is left intact) + `You are an agent. Your internal name is "localizer". The description about you is "<description>".`; user = the `request` text; tools = the sub-agent's own. The child `Runner` gets no compaction config, so **no compaction inside the sub-agent**. Its tool calls run the same harness tool callables, so they count toward the task's tool budget. Confirm on a live capture (session C). |
| 5 | Compaction request layout, summarizer model, trigger | **From ADK 1.39.1 code + replay:** trigger = latest model response's `usage.prompt_tokens` ≥ `token_threshold` (falls back to chars/4 only if usage is missing); summarizer = `LlmEventSummarizer` on the **root agent's own model (with its adapter)**; the summary enters later requests as a user message with parts `For context:` and `[model] said: <summary>`, followed by the last `event_retention_size` events. Confirm on a live capture (session D). |
| 6 | Nudge and warning appearance in requests | |
| 7 | vLLM and transformers versions in the Kaggle wheelhouse | **vLLM 0.19.1, transformers 5.13.1** (from the team, 2026-10-24) |
| 8 | `hints_text` present in dev/hidden tasks | |
| 9 | Graph data present for dev repositories | |
| 10 | Prefix-consistency result (one sequence vs per turn) | **Consistent within a segment** once (a) a tool-call turn's visible text is folded into its thinking and (b) records break at every new user message (nudge, compaction summary); the record's last message is rendered in generation format (keeps thinking for a final text turn). Verified on the four fixtures with the real tokenizer; token boundaries align at every target. |
| 11 | Does vLLM 0.19.1, started exactly as the harness starts it, load a LoRA adapter on the 31B and E4B `-ct` checkpoints? | **Yes, both** (Kaggle L4×4, 2026-10-25, `notebooks/kaggle_lora_check.py`): server started, adapter requests worked, outputs changed (max \|Δ logprob\| 1.52 on 31B, 2.97 on E4B). See §13. |

## 10. `agent_tool` (google-adk 1.39.1, `google/adk/tools/agent_tool.py`)

- Declaration: one required string parameter `request` (unless the sub-agent has an input schema).
- Run: a new `Runner` with an `InMemorySessionService`; the child session's state is a copy of the
  parent state (minus `_adk*` keys), so `{problem_description}` resolves in the sub-agent's
  instruction; the user message is the `request` text; state deltas are forwarded to the parent.
- Result: the sub-agent's last content parts that are **not thoughts**, joined with newlines
  (validated against an output schema if one is set).
- `skip_summarization: true` sets `tool_context.actions.skip_summarization`, which makes the
  function-response event a final response: the parent's LLM flow **stops** after the call, and
  the harness's outer loop then sends a continuation nudge (§15). Use `false` for a sub-agent whose
  answer the parent must act on.
- The child runner is built from the agent only (not the App), so the App's compaction config is
  expected **not** to apply inside it (VERIFY 4).

## 11. Compaction (ADK)

- Harness values (Kaggle `scripts/inference.py`, per README): `compaction_interval=15`,
  `overlap_size=2`, `token_threshold=14336`, `event_retention_size=5`; applied only when the eval
  config sets `events_compaction_config` (`agent_runner.py:694`), so local runs must set it too.
- Context caching: `min_tokens=2048`, `ttl_seconds=1800`, `cache_intervals=10`.
- Default summarizer `LlmEventSummarizer` (`google/adk/apps/llm_event_summarizer.py`); default
  prompt: "The following is a conversation history between a user and an AI agent. Please
  summarize the conversation, focusing on key information and decisions made, as well as any
  unresolved questions or tasks. The summary should be concise and capture the essence of the
  interaction.\n\n{conversation_history}". The summary event has role `model`.

## 12. Tool outputs (formats used by the observation renderer)

- `run_command`: `{"status":"ok","stdout","stderr","exit_code":0}`; non-zero exit →
  `{"status":"error","error_type":"CommandError","error_message", "details":{stdout,stderr,exit_code}}`;
  stdout and stderr each truncated to 5,000 characters; timeout 300 s per command.
- `read_file(filepath, start_line=None, end_line=None)`: at most 150 lines and 10,000 characters;
  fields `status, filepath, content, start_line, end_line, total_lines, is_truncated`; `end_line`
  is the last complete line returned (a requested `end_line` beyond the cap is ignored).
- `edit_file(filepath, old_string, new_string, allow_multiple=False)`: 3-tier matching (exact,
  flexible, regex) via `adk_eval_core.apply_replacement`; returns `occurrences`, `strategy`, `diff`
  (truncated at 5,000 characters), `is_truncated`; error `FileEditError` on no match, multiple
  matches, empty file or empty `old_string`.
- `write_file`: `{"status":"ok","filepath","size"}`.
- `submit_patch`: `{"status":"ok","patch_size","files_changed"}`.

## 13. vLLM 0.19.1 and Gemma 4 LoRA

**Measured on Kaggle (L4×4, competition vLLM 0.19.1 and transformers 5.13.1, flags as the harness,
`notebooks/kaggle_lora_check.py`):**

| | 31B | E4B |
| --- | --- | --- |
| `architectures` in config | `Gemma4ForConditionalGeneration` | `Gemma4ForConditionalGeneration` |
| Resolved by vLLM | `Gemma4ForConditionalGeneration` | same |
| LoRA adapter loads and is applied | yes (max \|Δ logprob\| 1.52) | yes (2.97) |
| Layers | 60; full attention every 6th, `attention_k_eq_v: true` on layers 5, 11, …, 59 | 42; `attention_k_eq_v: false`; `num_kv_shared_layers: 18` |
| LoRA modules (q,k,v,o,gate,up,down on the language model) | 410 (no `v_proj` in the 10 k_eq_v layers) | 258 (KV-shared layers have no k/v projections) |

- An earlier reading of the v0.19.1 source suggested the multimodal class lacks LoRA support; the
  measurement shows it works. vLLM logs that it "supports adding LoRA to the tower modules" and
  ignores only vision/audio tower and `embed_vision` modules, so language-model targets are applied.
- Adapter keys written with text-only names (`base_model.model.model.layers.N...`) were accepted.
- 31B on 4×L4: weights load ~103 s, server ready ~6.3 min after launch (compilation + CUDA graphs);
  6.0 GiB weights per GPU; **KV cache 37,568 tokens** in total (one full 32K session fits); vLLM
  forces the `TRITON_ATTN` backend (head_dim 256 local / 512 global).
- **k_eq_v layers (31B):** in HF, V is computed from the `k_proj` output (no `v_proj` module); vLLM
  keeps a packed `qkv_proj` and loads the `k_proj` weight into the V slot too. A `k_proj` LoRA
  therefore reaches only K in vLLM unless the adapter also has a `v_proj` LoRA for those layers.
  Export copies the trained `k_proj` LoRA to `v_proj` there; the parity test decides whether that
  matches training (the dummy adapter in the check already includes the copy).
- `use_rslora` is honoured by vLLM's `PEFTHelper`; we keep standard scaling.

## 14. Golden render check (Kaggle, 2026-10-25)

`notebooks/golden_render_check_standalone.py` on the E4B `-ct` model with vLLM 0.19.1: for every
trained turn of the four fixture records (63 prompts), `locfix.render.chat` produced exactly the
token ids of vLLM's `/v1/chat/completions/render`. vLLM logged "Detected the chat template content
format to be 'openai'", and that it uses the model's `generation_config.json` sampling defaults
(temperature 1.0, top_k 64, top_p 0.95) unless the request sets them. E4B on 4×L4: weights 3.3 GiB
per GPU, KV cache 298,464 tokens (45× concurrency at 32K), server ready ~5 min after launch.

## 15. `agent_tool` in the harness loop, and `read_file`'s character cap (2026-10-25)

- **`skip_summarization` must be `false` for the localizer.** ADK 1.39.1 `Event.is_final_response()`
  returns true when `actions.skip_summarization` is set, and `BaseLlmFlow.run_async` stops at a
  final response. With `true`, the fixer's `runner.run_async` ends right after the localizer's
  answer; the harness loop (`agent_runner.py:788-1023`) sees no submitted patch and sends the nudge
  "Please continue your work using the available tools, or call submit_patch ..." as a new user
  message, which also erases the fixer's earlier thoughts from the prompt (§9 item 1). With `false`
  the fixer is called again with the tool result, as for any tool. `agents/locfix/agent.yaml` uses
  `false`; `tests/test_loc_fix_records.py` checks that a fixer record has exactly one user message.
- Tool result as the fixer sees it: `{"result": "{\"locations\": [...]}"}` (the sub-agent's final
  non-thought text, JSON-escaped inside ADK's `result` wrapper).
- Routing in the scripted replay: requests carry `Your internal name is "<name>"` in the system
  message, so `locfix.harness.replay` serves one script per agent.
- **`read_file` character cap:** when the selected lines exceed `max_file_chars` (10,000), the
  harness cuts `content` at 10,000 characters, possibly in the middle of a line, sets
  `is_truncated: true` and still reports the line-capped `end_line` (`swegemma/tools/workspace.py`,
  `read_file`). `end_line` is therefore not "the last complete line shown". Synthetic reads
  (`locfix.data.render_obs.chunk_range`) are sized so that every output is complete.

**2026-10-26:** golden check re-run with `locfix_fixtures.jsonl` added (localizer system prompt, `agent_tool` call and `{"result": ...}` response, synthetic reads): 141/141 prompts identical, `ok: true`.
