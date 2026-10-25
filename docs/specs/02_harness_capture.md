# Spec 02 — Capturing real harness requests (golden set) and resolving VERIFY items

## Purpose

Training records must reproduce exactly what the harness sends to vLLM. Instead of guessing the
format (thought placement, tool-call syntax, sub-agent results, compaction summaries, nudges),
we record real requests once and test everything against them.

## 1. Capture proxy (`locfix harness capture-proxy`)

A small HTTP proxy between LiteLLM and vLLM:
- listens on a port, forwards every request to vLLM's OpenAI server unchanged, returns the
  response unchanged;
- appends to a JSONL file: timestamp, endpoint, full request body (`model` = adapter name,
  `messages`, `tools`, sampling params, `extra_body`), full response body (content,
  `reasoning_content`, `tool_calls`, `finish_reason`, usage);
- streaming must be passed through (store the reassembled response).

Run the harness locally (`swegemma eval ... --sandbox docker`) with the model registry pointed at
the proxy (the harness binds `LiteLlm(model="openai/<served>", api_base=...)`; point
`api_base` at the proxy, or attach to an already running endpoint as the harness supports).

## 2. Capture sessions (on dev tasks, E4B base is fine)

| Session | Agent config | Why |
| --- | --- | --- |
| A | single agent, our tools, `include_thoughts: true` | baseline format, thought history |
| B | same, `include_thoughts: false` | what history looks like without thoughts |
| C | fixer root + localizer as `agent_tool` (`skip_summarization: true`) | sub-agent input/output format |
| D | single agent on a task that runs past 14,336 prompt tokens (raise budgets if needed) | compaction format |
| E | a run that hits a nudge (e.g. `max_output_tokens` small) and a budget warning (`max_tool_calls: 22`) | nudge and warning formats |

Save each session's proxy log, harness trace (`results/.../traces/`) and the agent YAML to
`golden/` (private; small subset may become a test fixture).

## 3. Golden tests built from the capture

1. **Render test:** for every captured request, our renderer (`src/locfix/render/chat.py`) turns
   `messages + tools` into token ids; they must equal vLLM's `/tokenize` output for the same
   request (vLLM's tokenize endpoint accepts chat messages and applies the server's chat template).
2. **Prefix-consistency test (decides the training layout):** for consecutive requests in one
   session, check `tokens(request_{t+1})` starts with `tokens(request_t) + tokens(completion_t)`
   where `completion_t` is the raw generated text of turn t (as the model produced it, including
   `<|channel>thought…<channel|>` and the tool-call tokens).
   - If it holds: train each trajectory as **one sequence** with loss on all assistant turns.
   - If it fails (e.g. the template drops earlier thoughts, or re-serializes tool calls
     differently): train **per turn** (one example per assistant turn, its exact context), or per
     segment between the points where consistency breaks. Record which applies in
     `harness_facts.md`.
3. **Completion-format test:** re-render each captured response as a history message and compare
   with the generated tokens; any difference must be handled in the renderer (the loss is on what
   the model generates, so the target tokens must be the generation format).

## 4. VERIFY items to resolve (write answers into `docs/reference/harness_facts.md`)

0. **Done (2026-10-25): adapters load and apply on E4B and 31B.** Original item: start vLLM 0.19.1 exactly as the harness does (flags in `harness_facts.md`
   §1, no hf-overrides for 31B/E4B) with `--enable-lora` and a dummy adapter, for E4B and 31B. If it
   raises "does not support LoRA yet", stop and report: the model plan changes (see
   `harness_facts.md` §13).
1. Exact message fields for prior thoughts (`reasoning_content` and/or `reasoning`) and whether
   Gemma 4's chat template renders earlier thoughts (session A vs B).
2. How tool calls and tool results are serialized in `messages` (ids, `name`, JSON string args)
   and how parallel tool calls appear (the teacher made parallel `read_file` calls).
3. `agent_tool` call: the root sees a function named after the sub-agent with one string argument
   `request` (google-adk 1.39.1 `AgentTool._get_declaration`); confirm the tool schema and the
   result format the root receives (ADK returns the sub-agent's last non-thought text; check whether
   it is wrapped, e.g. `{"result": ...}`).
4. Sub-agent context: the sub-agent runs in a fresh in-memory session with a copy of the parent's
   state (so `{problem_description}` works in its instruction) and receives `request` as its user
   message. Confirm: does it get `build_agent_prompt` text? (Expected: no.) Is compaction active
   inside it? (Expected: no — it runs under its own `Runner`, not the App.) Do its tool calls count
   toward the same task budget? (Expected: yes, tools share the context.)
5. Compaction: what the request looks like after compaction (summary placement, role, which events
   are retained), which model wrote the summary (expected: the agent's own model and adapter via
   ADK's `LlmEventSummarizer`, default prompt in `google/adk/apps/llm_event_summarizer.py`), and the
   exact trigger (prompt tokens vs threshold).
6. Nudge messages and `budget_warning` fields appear as in `harness_facts.md` §5–6.
7. Versions: **answered — vLLM 0.19.1, transformers 5.13.1.**
8. Whether hidden/dev tasks have `hints_text` (if yes, the prompt has a `## Hints:` section).
9. Whether graph data exists for the dev repositories (if graph tools are declared, the prompt gets
   a "Code Intelligence Tools" section).

## Acceptance tests

- Proxy is transparent: a harness run through the proxy gives the same result as without it.
- Golden render test passes on 100% of captured requests for E4B and 31B.
- Prefix-consistency result recorded, and the trainer's layout (one sequence vs per turn) set
  accordingly in `configs/defaults.yaml`.
- All nine VERIFY items answered in `harness_facts.md`, with the evidence file named.
