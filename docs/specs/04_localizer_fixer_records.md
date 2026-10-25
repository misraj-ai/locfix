# Spec 04 — Localizer and fixer records (Stages 2–4)

## Purpose

Build localizer and fixer training examples from teacher trajectories (solved set) and from the
unsolved set, in the agent structure we deploy: **fixer = root agent**, **localizer = sub-agent
called as a tool**. All records are produced with the scripted replay of spec 03, so formatting is
exactly what the harness produces; this spec defines *what* is replayed.

## 1. Agent structure (deployed)

```
agents/locfix/agent.yaml              # fixer, adapter: fix_lora
  tools: run_command, read_file, edit_file, write_file, get_status, submit_patch,
         agent_tool: {config_path: sub_agents/localizer.yaml, skip_summarization: true}
agents/locfix/sub_agents/localizer.yaml   # localizer, adapter: loc_lora
  tools: run_command, read_file
  instruction: uses {problem_description}; ends by returning z as JSON (format below)
```

- The fixer calls the localizer with one string argument `request` (ADK `AgentTool`), first to get
  z, and again later if it needs more context ("find where X is defined").
- The localizer runs in a fresh session (copy of the parent's state), **without compaction**
  (VERIFY item 4, spec 02), and its tool calls count toward the same task budget. Its exploration
  must therefore stay short: cap at `em.localizer_max_calls` (default 12) and within 32K tokens.
- "Single model, two roles" ablation: same YAMLs, one adapter for both.
- Drafts of both instructions are in `agents/locfix/`. ADK fills `{name}` placeholders from
  session state; the localizer prompt also contains a literal JSON example with braces. Check in
  capture C (spec 02) that the instruction reaches the model with the JSON example intact and
  `{problem_description}` filled in.
- Include paths in `sub_agents/localizer.yaml` are relative to `sub_agents/` (the harness rejects
  `..`), hence the duplicated `sub_agents/configs/sampling.yaml`.

## 2. Location extraction (`src/locfix/data/locations.py`)

**Code at the base commit.** Needed for every task: the gold/edited files and every file the
teacher viewed, as the agent saw them (SWE-smith bug applied, setup patch applied). The unsolved
set already has `experiment/code/<id>.json`; **the same extraction must be run for the solved set**
(ask the data-pipeline agent; same code as notebook section 8).

**AST index:** per file, functions/classes/methods with line spans (Python `ast`; non-Python files
are file-level only). Module-level code between definitions is a pseudo-symbol `<module>:a-b`.

| Set | Definition |
| --- | --- |
| `L_edit` | symbols whose spans intersect lines changed by `teacher.patch` (solved) or the gold patch (unsolved); new files as file-level entries |
| `L_view` | in order, the files and line ranges the teacher viewed **before its first edit**: `read_file` ranges (use the returned `start_line`/`end_line`), `sed -n 'a,bp' f`, `cat f`, `head`/`tail` on a file; mapped to enclosing symbols. `grep` hits are searches, not views |
| Teacher searches | the teacher's `grep`/`find`/`rg` commands before the first edit, with their recorded outputs |

## 3. The localization z

```json
{"locations": [
  {"path": "arrow/factory.py", "symbol": "ArrowFactory.get", "lines": [90, 186], "role": "edit"},
  {"path": "arrow/arrow.py", "symbol": "Arrow.fromdatetime", "lines": [301, 336], "role": "context"}
]}
```

- At most `em.max_locations` (8) entries; ranked; total code of the spans ≤ `em.ctx_budget_tokens`
  (6K). Every symbol with `role: edit` is included in full.
- The localizer's final message is exactly this JSON (no other text), so the fixer and the scorer
  can parse it. Invalid JSON at inference: the fixer still sees the text; nothing breaks.

## 4. Localizer records

Replay script for the sub-agent stream:
1. user message = the fixer's `request` (template: "Find the code locations relevant to this
   issue.") with the instruction carrying `{problem_description}`;
2. the teacher's turns **up to its first edit** (thinking, `run_command`/`read_file` calls and
   their recorded outputs), excluding any call that writes inside `/workspace`;
3. final turn: z as JSON. Thinking for the final turn: the teacher's last thinking before its
   first edit, if it only discusses locations; otherwise a templated one-liner.

Labels per stage: `s2_loc_edit` (z = L_edit), `s2_loc_view` (z = L_view), `s3_loc_r0`
(z = L_edit ∪ L_view, pruned to the budget), `s3_loc_r{t}` (candidates with EM weights, spec 06).

Filters: exploration over `em.localizer_max_calls` counted calls or over 32K tokens → drop
(report counts).

## 5. Fixer records

Replay script for the root stream (initial prompt from `build_agent_prompt` with the fixer's
declared tools):
1. **Call the localizer:** a short thinking line + `localizer(request=...)`; the tool result is
   z (the sub-agent's final text, in whatever wrapper VERIFY item 3 shows).
2. **Read z:** one `read_file(path, start, end)` per location (≤ 150 lines per call, split longer
   spans). Output: reuse the teacher's recorded observation if it read the same file and range with
   the same tool; otherwise render offline (§7).
3. **Requests for what is missing** (only in records "with requests"): for each item in
   `N \ z` (N = L_edit ∪ L_view), in the teacher's viewing order:
   - if its path or symbol appears in what the fixer has seen so far → `read_file` on it;
   - otherwise a teacher `grep` whose recorded output contains that path, then `read_file`;
   - otherwise call the localizer again with a request naming what is needed, returning a z for
     that item (this is the "localizer answers requests" path);
   - rationale text: a templated line ("I need the definition of `X`, which `Y` calls."). No LLM
     writing here (avoids reintroducing recall).
4. **The teacher's trajectory from its first edit to the end:** edits, test runs, corrections,
   `submit_patch`, final text, with recorded outputs. These outputs are valid because the code state
   is identical from the first edit on.

Variants:

| Stage | z | Requests |
| --- | --- | --- |
| `s2_fix_gold` (2a) | L_edit ∪ L_view | none |
| `s2_fix_noisy` (2b) | perturbations (§6), ~3 per issue | yes, with `em.no_request_fraction` (40%) of records having none |
| `s3_fix_r{t}` | candidates with EM weights (spec 06) | as 2b |

**Validity checks (raise or drop, and count):**
- Impossible input: z misses an `L_edit` location and no request recovers it → drop.
- Reference check: every file path and every function/class name mentioned in the kept teacher
  reasoning (step 4) must appear somewhere in what the fixer has seen by then; otherwise add the
  missing request (if the teacher viewed it) or drop the record.

## 6. Candidate perturbations (round 0 and noisy fixer data)

From design doc §7.2: L_edit only; L_edit ∪ L_view; L_view minus one item; gold + callers/callees
of the edited symbols (static name references within the available files); gold + BM25
distractors (BM25 over symbols of the available files, query = issue text); one or two candidates
missing a gold file (negatives, used for localizer weighting, never as fixer inputs without a
recovering request). The distractor pool is limited to files in `code/<id>.json`; document this.

## 7. Offline observation rendering (`src/locfix/data/render_obs.py`)

`read_file` output for synthetic reads is rendered from the base-commit file text, reproducing the
harness exactly: JSON fields `status, filepath, content, start_line, end_line, total_lines,
is_truncated`, the 150-line and 10,000-character caps, `end_line` = last complete line returned,
`filepath` echoed as passed. Synthetic `grep` outputs are never generated; only recorded ones are
reused.

**Test:** for every teacher `read_file` call made before its first edit (thousands in the data),
rendering from `code/<id>.json` must reproduce the recorded output exactly.

## 8. Diff → `edit_file` converter (`src/locfix/data/edits.py`)

Used for scoring targets (spec 06) and for unsolved fixer records (gold patch). For each changed
region: `old_string` = the smallest block including the changed lines that is unique in the file
(extend context lines until unique), `new_string` = its replacement; new files → `write_file`;
deleted files are out of scope (drop the task). Apply the calls in order with
`adk_eval_core`'s `apply_replacement` (strategy must be `exact`) and require the result to equal
the patch applied with `git apply`.

## 9. Unsolved set (Stage 4)

Input: `experiment/unsolved_explorations.jsonl`, `experiment/code/<id>.json`, traces after
cleaning v2. Filters (from `DATA_PIPELINE.md`): one attempt per task (earliest reach of gold
files), non-empty `gold_locations`, `localization.path` ∈ {direct, searched, struggled} for the
main set, tasks with a code file.
- **Localizer:** prefix = Kimi's exploration up to the first time it opened a gold file;
  candidates = gold location + detour files (hard negatives); weights from the scorer.
- **Fixer:** z → reads (rendered) → gold patch as `edit_file` calls (§8) → `submit_patch` →
  templated final text. No verification segment.
- **never_found ablation:** insert a synthetic `read_file` of the gold file before the localizer's
  answer, mark it in `meta.synthetic_steps`, report separately.

## Acceptance tests

- `read_file` rendering reproduces 100% of eligible recorded teacher outputs.
- Diff converter: 100% of `teacher.patch` and gold patches either convert and re-apply exactly, or
  are dropped with a logged reason (target ≥ 98% converted).
- Sample traces (fixtures): arrow → L_edit = {`ArrowFactory.get`}; gunicorn → L_edit =
  {`Worker.init_process`} and L_view includes `gunicorn/reloader.py` and `tests/test_reload.py`.
- A replayed fixer record with requests passes the golden structure check (spec 03) and its
  localizer tool call/result match capture C (spec 02).
- Manifest per stage: records per source and variant, fraction with requests, drops per reason.
