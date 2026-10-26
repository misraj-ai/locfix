"""Write notebooks/inspect_training_data.ipynb (CPU only; Colab or local).

    python tools/build_inspect_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

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
code('''
# --- Get the code: set REPO_URL to your public repo, or leave it empty and upload locfix_code.zip
REPO_URL = ""

import os, sys, json, random, pathlib, subprocess

IN_COLAB = "google.colab" in sys.modules


def find_root():
    for c in (pathlib.Path("/content/locfix"), pathlib.Path.cwd(), pathlib.Path.cwd().parent):
        if (c / "src" / "locfix").is_dir():
            return c
    return None


ROOT = find_root()
if ROOT is None and IN_COLAB:
    if REPO_URL:
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/locfix"], check=True)
    else:
        from google.colab import files
        print("Upload locfix_code.zip")
        uploaded = files.upload()
        subprocess.run(["unzip", "-q", "-o", next(iter(uploaded)), "-d", "/content"], check=True)
    ROOT = find_root()
assert ROOT is not None, "repository not found"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))          # use the code in place (no install needed)
print("code:", ROOT)
''')
code('''
# --- Data on Google Drive and the harness wheels
DATA_DIR = "/content/drive/MyDrive/codeagent_final"     # folder with cleaned/, experiment/, ...
WHEELS_DIR = "/content/drive/MyDrive/wheels"             # swegemma, adk_submission, adk_eval_core .whl

if IN_COLAB and DATA_DIR.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
for p in (DATA_DIR, WHEELS_DIR):
    print(p, "found" if pathlib.Path(p).exists() else "NOT FOUND")
''')
code('''
# --- Python packages: harness wheels + ADK (replay), transformers 5.13.1 (tokenizer/chat template)
import importlib.util

def missing(mod):
    return importlib.util.find_spec(mod) is None

need = []
if missing("swegemma") or missing("adk_submission"):
    wheels = sorted(str(p) for p in pathlib.Path(WHEELS_DIR).glob("*.whl"))
    assert wheels, f"no .whl files in {WHEELS_DIR}"
    need += [*wheels, "google-adk>=1.34,<2", "litellm", "mcp>=2"]
import importlib.metadata as md
try:
    tf_ok = md.version("transformers") == "5.13.1"
except md.PackageNotFoundError:
    tf_ok = False
if not tf_ok:
    need += ["transformers==5.13.1"]
if need:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need, "pyyaml", "jsonschema", "pandas"],
                   check=True)
print("transformers", md.version("transformers"), "| google-adk", md.version("google-adk"),
      "| swegemma", md.version("swegemma"))
''')
code('''
# --- Gemma 4 tokenizer + chat template (3 small files, no weights)
# TOKENIZER_SOURCE:
#   "kaggle"  download tokenizer.json, tokenizer_config.json, chat_template.jinja from the Kaggle
#             model (Kaggle credentials in Colab secrets or kagglehub.login(); Gemma 4 license
#             accepted on Kaggle)
#   "upload"  upload e4b.zip (the model folder without weights)
#   a folder path that already holds the three files (e.g. on Drive)
TOKENIZER_SOURCE = "kaggle"
KAGGLE_HANDLE = "google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]

local = ROOT / "work" / "model_meta" / "e4b"
if (local / "tokenizer.json").exists():
    TOKENIZER_DIR = str(local)
elif TOKENIZER_SOURCE == "kaggle":
    import kagglehub
    paths = [kagglehub.model_download(KAGGLE_HANDLE, path=f) for f in TOKENIZER_FILES]
    TOKENIZER_DIR = os.path.dirname(paths[0])
elif TOKENIZER_SOURCE == "upload":
    from google.colab import files
    print("Upload e4b.zip")
    uploaded = files.upload()
    subprocess.run(["unzip", "-q", "-o", next(iter(uploaded)), "-d", "/content/model_meta"], check=True)
    TOKENIZER_DIR = str(next(pathlib.Path("/content/model_meta").rglob("tokenizer.json")).parent)
else:
    TOKENIZER_DIR = TOKENIZER_SOURCE

from locfix.render.chat import load_tokenizer, render_record
tok = load_tokenizer(TOKENIZER_DIR)
print("tokenizer:", TOKENIZER_DIR, "| vocab", len(tok))
''')
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
task id. A task is skipped (and another one drawn) when it cannot give both kinds of records; the
reason is printed (e.g. `over budget`: the trajectory needs more tool calls than our budget).

**Code at the base commit** (needed for the fixer's reads of z): taken from `code/<id>.json` when
the data has one; otherwise rebuilt from the lines the teacher printed, and reads are limited to
those lines (`code: trace` below).
""")
code('''
SEED = None            # None = random every run
INSTANCE = None        # e.g. "benoitc__gunicorn.bacbf8aa.lm_rewrite__9mzalywg" to pick one task
MAX_TRIES = 25

from locfix.config import load_config
from locfix.data.agent_records import BuildConfig, build_agent_records
from locfix.data.dataset import SolvedData
from locfix.data.loc_fix_records import LocFixConfig, TaskSkip, build_task_records
from locfix.harness.budget import BudgetExceeded
from locfix.harness.prompts import Budget

cfg = load_config()
data = SolvedData(DATA_DIR, exclude_repos=cfg["data"]["exclude_repos"])
print(f"{len(data.trace_paths())} traces, {len(data.rows)} solved rows; code folders:",
      [str(d) for d in data.code_dirs if d.exists()] or "none (code rebuilt from traces)")

b, bl = cfg["budget"], cfg["budget_locfix"]
bc = BuildConfig(agent_dir="agents/single", model_alias=cfg["models"]["31b"]["serve_id"],
                 budget=Budget(b["max_time_minutes"], b["max_tool_calls"], b["max_turns"]))
lc = LocFixConfig(agent_dir="agents/locfix", model_alias=cfg["models"]["31b"]["serve_id"],
                  budget=Budget(bl["max_time_minutes"], bl["max_tool_calls"], bl["max_turns"]),
                  em=cfg["em"], seed=cfg["seed"])

rng = random.Random(SEED)
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

trace = task.trace
row = data.row(task.instance_id)
show_json({"instance_id": task.instance_id, "trace": trace_path.name, "repo": row.get("repo"),
           "dataset": row.get("verifier"), "fold": row.get("fold"),
           "teacher tool calls": trace.n_counted_calls, "teacher turns": len(trace.turns),
           "code": code_info.source, "code coverage (trace)": code_info.coverage,
           "records": {"single agent": len(s1), "localizer/fixer": counts}})
''')

md("""
## 1. The input: one teacher trace

The raw log of Kimi-K3 solving the task with **our harness** (after recall cleaning): system
instruction, task prompt, then `thinking`, `tool_call`, `tool_response` entries and the final
text. It was recorded with the teacher's budget (8 minutes, 100 tool calls).
""")
code('table(I.trace_rows(trace.entries, 140))')

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
teacher_sys, teacher_user = trace.system_instruction, trace.task_prompt
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
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    dest = ROOT / "notebooks" / "inspect_training_data.ipynb"
    dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {dest} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
