"""Write notebooks/inspect_training_data.ipynb (CPU only; Colab or local).

    python tools/build_inspect_notebook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb_common import GET_CODE_CELL, TOKENIZER_CELL, packages_cell, write_notebook  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# How the training data is built — random samples from the real data

Picks a random solved task from `codeagent_final/`, builds its training records **live** with the
same code the pipeline uses, and shows every step:

```
teacher trace (Kimi-K3 log, cleaned/traces/<id>__a<N>.json)
   │  replay through the real harness code (ADK + harness tools + our agent YAML),
   │  with a scripted model that repeats the teacher's turns and recorded tool outputs
   ▼
record   = the exact OpenAI chat messages the model receives on Kaggle + which turns are trained
   │  Gemma 4 chat template (the same rendering vLLM does; golden check 141/141)
   ▼
text     = one long string with special tokens
   │  tokenizer
   ▼
tokens   = input_ids + labels (−100 = no loss) + loss weights   →  what the trainer receives
```

Two setups:
1. **Single agent** (Stage 1 baseline): one record per trajectory (more if it has nudges).
2. **Two agents** (our method): the fixer is the root agent and calls the localizer as a tool. One
   teacher trajectory gives **localizer records** (exploration → list of locations z) and **fixer
   records** (call localizer → read z → edit, test, submit).

Needs: the code (repo or `locfix_code.zip`), the three harness wheels (the replay runs the real
harness code), the data folder (Google Drive), and the Gemma 4 tokenizer files. CPU runtime is
enough. Re-run section 1 to get another random task.
""")

md("## 0. Setup")
code(GET_CODE_CELL)
code('''
# --- Where the data is (Google Drive)
# RECORDS_DIR: folder with records already built by `locfix data.agent-records` /
#   `data.locfix-records` (docs/DATA.md, step 2). Set it to only look at records: no harness wheels
#   needed. Leave None to build the records of a random task live (needs WHEELS_DIR).
RECORDS_DIR = None                                       # e.g. "/content/drive/MyDrive/records"
DATA_DIR = "/content/drive/MyDrive/codeagent_final"     # folder with cleaned/, experiment/, ...
WHEELS_DIR = "/content/drive/MyDrive/wheels"             # swegemma, adk_submission, adk_eval_core .whl

if IN_COLAB and any(str(p).startswith("/content/drive") for p in (RECORDS_DIR, DATA_DIR, WHEELS_DIR) if p):
    from google.colab import drive
    drive.mount("/content/drive")
for p in (RECORDS_DIR, DATA_DIR, WHEELS_DIR):
    if p:
        print(p, "found" if pathlib.Path(p).exists() else "NOT FOUND")
''')
code(packages_cell("RECORDS_DIR is None"))
code(TOKENIZER_CELL)
code('''
# --- Display helpers
import pandas as pd
from IPython.display import HTML, Markdown, display

from locfix.render import inspect as I

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", 300)


def table(rows):
    df = pd.DataFrame(rows)
    display(df.style.hide(axis="index").set_properties(**{"text-align": "left", "font-size": "12px"}))


def show_json(obj):
    display(Markdown("```json\\n" + json.dumps(obj, indent=1, ensure_ascii=False) + "\\n```"))
''')

md("""
## 1. Pick a random task and build its records

`SEED = None` gives a new task on every run; set an integer to reproduce one, or `INSTANCE` to a
task id. With `RECORDS_DIR` set, the task is drawn from the record files. Otherwise its records are
built live, and a task is skipped (another one drawn) when it cannot give both kinds of records;
the reason is printed (e.g. `over budget`: the trajectory needs more tool calls than our budget).

**Code at the base commit** (needed for the fixer's reads of z): taken from `code/<id>.json` when
the data has one; otherwise rebuilt from the lines the teacher printed, and reads are limited to
those lines (`code: trace` below).
""")
code('''
SEED = None            # None = random every run
INSTANCE = None        # e.g. "benoitc__gunicorn.bacbf8aa.lm_rewrite__9mzalywg" to pick one task
MAX_TRIES = 25

from locfix.config import load_config
from locfix.data.traces import load_trace

cfg = load_config()
rng = random.Random(SEED)
code_info = None

if RECORDS_DIR:
    by_task = {}
    for f in sorted(pathlib.Path(RECORDS_DIR).rglob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            by_task.setdefault(r["instance_id"], []).append(r)
    ids = sorted(i for i, rs in by_task.items()
                 if any(r["role"] == "agent" for r in rs) and any(r["role"] != "agent" for r in rs))
    if INSTANCE:
        ids = [i for i in ids if i.startswith(INSTANCE)]
    assert ids, "no task with both single-agent and localizer/fixer records in RECORDS_DIR"
    iid = rng.choice(ids)
    s1 = [r for r in by_task[iid] if r["role"] == "agent"]
    lf = [r for r in by_task[iid] if r["role"] != "agent"]
    counts = {}
    for r in lf:
        counts[r["stage"]] = counts.get(r["stage"], 0) + 1
    name = (s1[0]["meta"] or {}).get("trace", "")
    trace_path = pathlib.Path(DATA_DIR) / "cleaned" / "traces" / name
    trace = load_trace(trace_path) if name and trace_path.exists() else None
    print(f"{len(ids)} tasks in {RECORDS_DIR}")
else:
    from locfix.data.agent_records import BuildConfig, build_agent_records
    from locfix.data.dataset import SolvedData
    from locfix.data.loc_fix_records import LocFixConfig, TaskSkip, build_task_records
    from locfix.harness.budget import BudgetExceeded, time_scale_from_cfg
    from locfix.harness.prompts import Budget

    data = SolvedData(DATA_DIR, exclude_repos=cfg["data"]["exclude_repos"])
    print(f"{len(data.trace_paths())} traces, {len(data.rows)} solved rows; code folders:",
          [str(d) for d in data.code_dirs if d.exists()] or "none (code rebuilt from traces)")
    b, bl = cfg["budget"], cfg["budget_locfix"]
    bc = BuildConfig(agent_dir="agents/single", model_alias=cfg["models"]["31b"]["serve_id"],
                     budget=Budget(b["max_time_minutes"], b["max_tool_calls"], b["max_turns"]),
                     time_scale=time_scale_from_cfg(cfg, "budget"))
    lc = LocFixConfig(agent_dir="agents/locfix", model_alias=cfg["models"]["31b"]["serve_id"],
                      budget=Budget(bl["max_time_minutes"], bl["max_tool_calls"], bl["max_turns"]),
                      em=cfg["em"], seed=cfg["seed"], time_scale=time_scale_from_cfg(cfg, "budget_locfix"))
    paths = data.trace_paths()
    if INSTANCE:
        paths = [p for p in paths if p.name.startswith(INSTANCE)]
    rng.shuffle(paths)
    for trace_path in paths[:MAX_TRIES]:
        try:
            task, code_info = data.task(trace_path)
            s1 = build_agent_records(task.trace, bc, task.source, task.problem_statement)
            counts = {}
            lf = build_task_records(task, lc, counts)
        except BudgetExceeded as e:
            print(f"skip {trace_path.name}: over budget ({e})")
            continue
        except TaskSkip as e:
            print(f"skip {trace_path.name}: {e}")
            continue
        if not lf:
            print(f"skip {trace_path.name}: no localizer/fixer records {counts}")
            continue
        break
    else:
        raise RuntimeError("no usable task found; raise MAX_TRIES")
    trace, iid = task.trace, task.instance_id

r0 = s1[0]
show_json({"instance_id": iid, "trace": trace_path.name if trace else None,
           "dataset": r0["source"].get("dataset"), "fold": r0["source"].get("fold"),
           "teacher tool calls": trace.n_counted_calls if trace else None,
           "teacher turns": len(trace.turns) if trace else None,
           "code": code_info.source if code_info else (lf[0]["meta"].get("code_source") if lf else None),
           "records": {"single agent": len(s1), "localizer/fixer": counts}})
''')

md("""
## 1. The input: one teacher trace

The raw log of Kimi-K3 solving the task with **our harness** (after recall cleaning): system
instruction, task prompt, then `thinking`, `tool_call`, `tool_response` entries and the final
text. It was recorded with the teacher's budget (8 minutes, 100 tool calls).
""")
code('''
if trace is not None:
    table(I.trace_rows(trace.entries, 140))
else:
    print("trace file not found under DATA_DIR/cleaned/traces")
''')

md("""
## 2. Single agent (Stage 1)

### 2.1 The record

The trace is replayed through `agents/single` with **our** budget (5 minutes, 30 tool calls). The
replay captures the request the model would get at every step; consecutive requests that extend
each other form one record. Mapping:

| trace entry | record message |
| --- | --- |
| system_instruction | `system` (plus ADK's line `You are an agent. Your internal name is "swe_agent".`) |
| task_prompt | `user` (rebuilt by the harness's `build_agent_prompt` with our budget) |
| thinking + tool_call | `assistant` with `reasoning` and `tool_calls` |
| tool_response | `tool` (budget warnings recomputed for our budget) |
| final text | last `assistant` message with `content` |

`trained` marks the assistant turns that get loss. A turn can get weight 0 (e.g. a repeated
identical call).
""")
code('''
table([I.record_summary(r) | {"tools": len(r["tools"])} for r in s1])
rec = s1[0]                       # the first window (a nudge starts a new one)
show_json(I.record_summary(rec))
table(I.message_rows(rec))
''')
md("What the replay changed relative to the teacher's prompts:")
code('''
import difflib
sys_rec = I.content_text(rec["messages"][0])
user_rec = I.content_text(rec["messages"][1])
teacher_sys, teacher_user = (trace.system_instruction, trace.task_prompt) if trace else ("", "")
for name, a, b in (("system", teacher_sys, sys_rec), ("task prompt", teacher_user, user_rec)):
    print(f"--- {name}: teacher -> record")
    print("\\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)) or "(identical)")
''')

md("""
### 2.2 The training text

The record rendered with Gemma 4's chat template, exactly as vLLM renders it for the model.
**Grey** is context (no loss): system prompt, task, tool outputs. **Green** is what the model
learns to generate: its thinking (`<|channel>thought ... <channel|>`), its tool calls
(`<|tool_call>call:name{...}<tool_call|>`) and the token that ends a tool-call turn
(`<|tool_response>`), and the final answer. Long grey parts are shortened on screen only.
""")
code('''
R = render_record(rec, tok)
show_json(I.rendered_stats(R))
display(HTML(I.rendered_html(R)))
''')

md("""
### 2.3 Tokens and labels

The trainer sees token ids. `label = −100` means no loss on that position; otherwise the label is
the token itself (the model is trained to predict it from everything before it). Start of the
first trained turn:
""")
code('table(I.token_rows(R, tok, span=0, at="start"))')
md("End of the same turn: the turn ends with `<|tool_response>`, after which the harness inserts the tool output (grey again).")
code('table(I.token_rows(R, tok, span=0, at="end", before=8, after=6))')

md("""
### 2.4 What the trainer receives

Per example: `input_ids`, `labels`, `loss_weights` (all the same length). The loss is

  loss = Σₜ wₜ · CE(prediction at t−1, token t) / Σₜ wₜ  over positions with a label,

summed over all examples of a batch (across GPUs and gradient-accumulation steps) before dividing,
so long and short examples are weighted per token.
""")
code('''
b = I.trainer_batch(R)
print({k: len(v) for k, v in b.items()})
print("input_ids[:12]   ", b["input_ids"][:12])
print("labels[:12]      ", b["labels"][:12])
i0 = next(i for i, l in enumerate(b["labels"]) if l != -100)
print(f"first trained position {i0}: labels[{i0}:{i0 + 6}] =", b["labels"][i0:i0 + 6])
''')

md("""
## 3. Two agents (our method)

Deployed structure (`agents/locfix`): the **fixer** is the root agent; the **localizer** is a
sub-agent it calls as a tool (`localizer(request=...)`). The localizer explores and answers with
JSON z = a ranked list of code locations; the fixer reads them and fixes the issue.

The same teacher trace is split at its **first edit**:
- **localizer record** = the teacher's exploration before the first edit (only `run_command` /
  `read_file`), then a final answer z;
- **fixer record** = call the localizer → read the locations of z (`read_file` outputs produced
  by the harness's own `read_file` on the code at the base commit) → if z misses something,
  read it or reuse a teacher `grep` ("requests") → the teacher's trajectory from its first edit to
  `submit_patch`.

One trace gives several variants of z (the paper's Stage 2/3 data):

| variant | z | records |
| --- | --- | --- |
| gold | locations edited ∪ viewed by the teacher (pruned to 8 / 6K tokens) | fixer `s2_fix_gold`, localizer `s3_loc_r0` |
| edit_only | only the edited symbols | localizer `s2_loc_edit`, fixer `s2_fix_noisy` (with requests) |
| view_only | what the teacher viewed | localizer `s2_loc_view` (fixer only if z differs from gold) |
| gold_minus_one | gold minus one context item | fixer `s2_fix_noisy` |
""")
code('''
recs = lf
table([{"stage": r["stage"], "role": r["role"], "variant": r["meta"]["variant"],
        "messages": len(r["messages"]), "trained turns": len(r["loss"]),
        "requests": ", ".join(q["via"] for q in r["meta"].get("requests", [])),
        "z": ", ".join(f'{z["path"]}:{z["symbol"]} ({z["role"]})' for z in r["z"])} for r in recs])
''')

md("### 3.1 Localizer record (`s2_loc_edit`)\n\nIts own system prompt (`agents/locfix/sub_agents/prompts/localizer.md`, with the issue filled in), the fixer's request as the user message, the teacher's exploration, and z as the final answer.")
code('''
loc = next((r for r in recs if r["stage"] == "s2_loc_edit"), next(r for r in recs if r["role"] == "localizer"))
table(I.message_rows(loc))
print("final answer (z):")
show_json(json.loads(loc["messages"][-1]["content"]))
RL = render_record(loc, tok)
show_json(I.rendered_stats(RL))
display(HTML(I.rendered_html(RL)))
''')

md("""
### 3.2 Fixer record with an incomplete z (`s2_fix_noisy`)

z misses some of what the teacher looked at, so the fixer recovers it (`requests`: read what it
can already name, or reuse a teacher `grep`). Note message 3: the localizer's answer arrives wrapped by ADK as
`{"result": "<z JSON>"}`. Synthetic turns (call localizer, read z, requests) use short templated
thinking; from the first edit on, everything is the teacher's.
""")
code('''
fx = next((r for r in recs if r["stage"] == "s2_fix_noisy" and r["meta"].get("requests")),
          next(r for r in recs if r["role"] == "fixer"))
print("variant:", fx["meta"]["variant"], "| requests:", [q["via"] for q in fx["meta"].get("requests", [])])
table(I.message_rows(fx))
RF = render_record(fx, tok)
show_json(I.rendered_stats(RF))
display(HTML(I.rendered_html(RF)))
''')

md("### 3.3 Fixer record with the full z (`s2_fix_gold`): no requests needed")
code('''
fg = next((r for r in recs if r["stage"] == "s2_fix_gold"), None)
if fg is not None:
    table(I.message_rows(fg)[:10])
''')

md("## 4. All records of this task")
code('''
rows = []
for r in s1 + lf:
    x = render_record(r, tok)
    rows.append({"stage": r["stage"], "role": r["role"],
                 "variant": r["meta"].get("variant", ""), **I.rendered_stats(x)})
table(rows)
''')

md("""
## 5. What happens next (the trainer)

- One dataset per stage (`s1_agent`, `s2_fix_gold`, `s2_fix_noisy`, `s2_loc_*`, `s3_loc_r*`).
  The trainer renders records exactly as above (the same `render_record`), packs examples up to
  32K tokens, and trains a bf16 LoRA on the dequantized `-w4a16-ct` base with the weighted loss
  of §2.4.
- Single agent: one adapter. Two agents: a fixer adapter on fixer records and a localizer adapter
  on localizer records (or one shared adapter, as an ablation).
- In Stage 3 the localizer records' weights come from the fixer: how likely the fixer is to
  produce the teacher's verified patch given each candidate z.
""")


def main() -> None:
    write_notebook(CELLS, ROOT / "notebooks" / "inspect_training_data.ipynb")


if __name__ == "__main__":
    main()
