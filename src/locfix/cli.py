"""`locfix` command line. Each module registers its subcommands in `COMMANDS`.

Every subcommand accepts --config (repeatable) and --set key=value (repeatable).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from .config import load_config

# name -> "module:function" taking (args, cfg). Modules are imported lazily so that CPU-only
# commands do not import torch/vLLM.
COMMANDS: dict[str, tuple[str, str]] = {
    "config": ("locfix.cli:cmd_config", "print the resolved configuration"),
    "data.traces-stats": ("locfix.data.traces:cmd_stats", "statistics over a folder of traces"),
    "data.agent-records": ("locfix.data.agent_records:cmd_agent_records",
                           "Stage 1 records: replay teacher traces through the agent stack"),
    "data.stats": ("locfix.data.stats:cmd_stats",
                   "statistics of traces and records: budget, length, batching (writes stats.md)"),
    "data.check-records": ("locfix.render.inspect:cmd_check_records",
                           "validate record files and tokenize them (no harness needed)"),
    "quant.dequantize": ("locfix.quant.dequantize:cmd_dequantize",
                         "-w4a16-ct checkpoint -> bf16 text-only checkpoint for training (+ grid check)"),
    "quant.parity": ("locfix.quant.parity:cmd_parity",
                     "HF (dequantized + adapter) vs vLLM (-ct + LoRA) log-probs on trained tokens (GPU)"),
    "train": ("locfix.train.sft:cmd_train", "train one LoRA adapter (launch with accelerate)"),
    "train.plan": ("locfix.train.sft:cmd_plan", "dry run: tokenize sources, print steps/tokens/visits"),
    "data.locfix-records": ("locfix.data.loc_fix_records:cmd_locfix_records",
                            "Stage 2 / round-0 localizer and fixer records (fixer root + localizer agent_tool)"),
    "data.merge-records": ("locfix.data.build_common:cmd_merge_records",
                           "concatenate the --shard outputs of the record builders, sum their counts"),
    "data.code-paths": ("locfix.data.code_files:cmd_code_paths",
                        "files each solved task needs at the base commit (input of fetch_solved_code.ipynb)"),
    "data.fetch-code": ("locfix.data.fetch_code:cmd_fetch_code",
                        "base-commit code files for the solved tasks, with plain git (+ check against the traces)"),
    "data.check-code": ("locfix.data.code_files:cmd_check_code",
                        "do code/<id>.json files reproduce the teacher's recorded reads?"),
    "submit.package": ("locfix.submit.package:cmd_package",
                       "build a submission folder (agent YAMLs + adapters + budget) and check it"),
    "eval.run": ("locfix.eval.run:cmd_run",
                 "evaluate a submission with the harness's evaluator, vLLM and Kaggle's settings"),
    "eval.summary": ("locfix.eval.stats:cmd_summary", "resolve rate, 95%% CI, errors per results folder"),
    "eval.compare": ("locfix.eval.stats:cmd_compare", "paired comparison of two systems (exact McNemar)"),
    "data.split": ("locfix.data.build_common:cmd_split",
                   "split record files into training and validation files by task (data.val_fraction)"),
    "em.fold-split": ("locfix.em.folds:cmd_fold_split",
                      "Stage 3: per-fold record files (A/B) for cross-fitted localizers and fixers"),
    "em.candidates": ("locfix.em.candidates:cmd_candidates",
                      "Stage 3: sample K localizations per task from the other fold's localizer (GPU)"),
    "score.prepare": ("locfix.score.sequences:cmd_prepare",
                      "Stage 3: candidates -> scoring sequences (fixer context + canonical edits, y*+ positions)"),
    "score.run": ("locfix.score.vllm_scorer:cmd_run",
                  "Stage 3: log p of the verified patch's added tokens under the other fold's fixer (GPU)"),
    "score.weights": ("locfix.score.weights:cmd_weights",
                      "Stage 3: scores -> λ calibration, per-task τ (ESS target), w(z), z* and diagnostics"),
    "em.signal-test": ("locfix.em.signal:cmd_signal",
                       "Stage 3: fold fixer's own edit given z* vs z_T, paired against the verified patch (GPU)"),
}


def cmd_config(args: argparse.Namespace, cfg: dict) -> int:
    print(json.dumps(cfg, indent=2, default=str))
    return 0


def _resolve(target: str):
    mod, fn = target.split(":")
    return getattr(importlib.import_module(mod), fn)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="locfix")
    sub = p.add_subparsers(dest="command", required=True)
    for name, (target, help_) in COMMANDS.items():
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", action="append", default=[], help="run config YAML (repeatable)")
        sp.add_argument("--set", action="append", default=[], dest="overrides",
                        help="override key=value (repeatable)")
        add_args = getattr(importlib.import_module(target.split(":")[0]), "add_args_" + target.split(":")[1], None)
        if add_args:
            add_args(sp)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    return _resolve(COMMANDS[args.command][0])(args, cfg) or 0


if __name__ == "__main__":
    sys.exit(main())
