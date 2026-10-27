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
