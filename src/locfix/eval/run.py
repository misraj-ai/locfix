"""`locfix eval.run`: evaluate a packaged submission the way the Kaggle scorer runs it.

    locfix eval.run --submission work/submissions/S1-e4b --ct work/models/gemma-4-e4b-it-qat-w4a16-ct \
        --tasks <tasks.jsonl> --snapshots <snapshots dir> --out work/eval/S1-e4b/dev/seed0

What it does, using the harness's own code wherever the scorer does:
1. Checks the submission (`submit.package.validate_submission`): one allowed model, adapters with
   rank <= 128, the harness's `validate_directory`, < 3 GiB. Reads its `eval_config.yaml` budget.
2. Starts vLLM with `adk_submission.VllmServer` (the class the scorer uses): the `-ct` checkpoint,
   gemma4 tool-call and reasoning parsers, max_model_len 32768, every adapter in `adapters/` as a
   LoRA module (max_loras / max_lora_rank sized from the adapters), and the 12B hf-overrides.
   `--server-url` uses an already running server instead.
3. Runs `swegemma.evaluate.Evaluator`: agent container -> patch -> fresh verification container ->
   tests (resolved = the harness's own criterion). Unlike the plain `swegemma eval` command it
   applies the submission's budget and Kaggle's context compaction and caching
   (`harness.compaction`, `harness.context_cache` in configs/defaults.yaml).
4. Writes the harness's outputs (`task_results.jsonl`, `summary.json`, `patches/`, `traces/`,
   `logs/`, `test_outputs/`) plus `locfix_summary.json` (rate, 95% CI, errors, medians) and
   `run_meta.json`.

Re-running with the same --out continues: tasks already in task_results.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def select_tasks(tasks_path: Path, task_ids: list[str] | None = None, limit: int = 0,
                 skip: set[str] | None = None) -> list[str]:
    """Instance ids to run, in file order: `task_ids` (if given) or all; minus `skip`; first `limit`."""
    ids = []
    with open(tasks_path) as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["instance_id"])
    if task_ids:
        missing = sorted(set(task_ids) - set(ids))
        if missing:
            raise SystemExit(f"task ids not in {tasks_path}: {missing[:5]}")
        ids = [i for i in ids if i in set(task_ids)]
    if skip:
        ids = [i for i in ids if i not in skip]
    return ids[:limit] if limit else ids


def done_tasks(out: Path) -> set[str]:
    p = out / "task_results.jsonl"
    if not p.exists():
        return set()
    return {json.loads(x)["instance_id"] for x in p.read_text().splitlines() if x.strip()}


def merge_results(out: Path, previous: dict[str, str]) -> None:
    """The harness rewrites task_results.jsonl with only the tasks of the current run when it
    finishes; put back the rows of earlier runs (a task evaluated again keeps its newest row)."""
    p = out / "task_results.jsonl"
    rows = dict(previous)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                rows[json.loads(line)["instance_id"]] = line
    if rows:
        tmp = p.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(r + "\n" for r in rows.values()))
        tmp.replace(p)


def _rows(out: Path) -> dict[str, str]:
    p = out / "task_results.jsonl"
    if not p.exists():
        return {}
    return {json.loads(x)["instance_id"]: x for x in p.read_text().splitlines() if x.strip()}


def adk_harness_configs(cfg: dict, compaction: bool = True) -> dict[str, Any]:
    """Kaggle's compaction and context-cache settings as ADK objects (None when disabled)."""
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps.app import EventsCompactionConfig

    h = cfg["harness"]
    return {
        "events_compaction_config": EventsCompactionConfig(**h["compaction"]) if compaction else None,
        "context_cache_config": ContextCacheConfig(**h["context_cache"]) if h.get("context_cache") else None,
    }


def build_eval_config(sub: Path, tasks: Path, snapshots: Path, out: Path, models, cfg: dict, *,
                      budget: dict[str, Any], task_ids: list[str], concurrency: int, sandbox: str,
                      image: str, compaction: bool = True, display: str = "quiet"):
    from swegemma.config import EvalConfig

    h = cfg["harness"]
    return EvalConfig(
        tasks_path=tasks, snapshots_dir=snapshots, results_dir=out, submission_dir=sub, models=models,
        image=image, sandbox=sandbox, task_ids=task_ids, concurrency=concurrency, display_mode=display,
        max_time_minutes=budget.get("max_time_minutes"), max_tool_calls=budget.get("max_tool_calls"),
        max_turns=budget.get("max_turns"), timeout_seconds=h.get("command_timeout_seconds", 300),
        **adk_harness_configs(cfg, compaction),
    )


def vllm_config(ct: str, model_alias: str, cfg: dict, *, tp: int, gpu_util: float, port: int,
                extra_args: list[str] | None = None, startup_timeout: int = 1800):
    from adk_submission import VllmConfig

    h = cfg["harness"]
    extra = list(extra_args or [])
    if "12b" in model_alias:          # the scorer's MODEL_VLLM_EXTRA_ARGS for 12B
        extra += ["--hf-overrides", json.dumps({"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"})]
    return VllmConfig(model=ct, port=port, tool_call_parser=h["tool_call_parser"],
                      reasoning_parser=h["reasoning_parser"], max_model_len=h["max_model_len"],
                      gpu_memory_utilization=gpu_util, tensor_parallel_size=tp, extra_args=extra,
                      startup_timeout=startup_timeout)


def docker_preflight(image: str) -> None:
    """The sandbox image must exist locally: the requested tag, or one of the pre-built Kaggle images
    the harness falls back to (`swegemma.sandbox.docker.SDG_IMAGE_CANDIDATES`)."""
    from swegemma.sandbox.docker import SDG_IMAGE_CANDIDATES

    for name in (image, *SDG_IMAGE_CANDIDATES):
        try:
            r = subprocess.run(["docker", "image", "inspect", name], capture_output=True, text=True, timeout=60)
        except FileNotFoundError as e:
            raise SystemExit("docker is not installed or not on PATH (use --sandbox subprocess to run without it)") from e
        if r.returncode == 0:
            return
    raise SystemExit(f"no sandbox image: neither {image!r} nor {list(SDG_IMAGE_CANDIDATES)} is present. Pull one, "
                     f"e.g. `docker pull {SDG_IMAGE_CANDIDATES[0]} && docker tag {SDG_IMAGE_CANDIDATES[0]} {image}`")


def run_eval(args: argparse.Namespace, cfg: dict) -> dict[str, Any]:
    from adk_submission import discover_adapters
    from swegemma.evaluate import Evaluator

    from ..runmeta import write_run_meta
    from ..submit.package import validate_submission
    from .stats import summarize

    sub, out = Path(args.submission).resolve(), Path(args.out)
    tasks, snapshots = Path(args.tasks).resolve(), Path(args.snapshots).resolve()
    for p in (sub, tasks, snapshots):
        if not p.exists():
            raise SystemExit(f"{p}: not found")
    info = validate_submission(sub)
    budget = dict(info["budget"])
    for k in ("max_time_minutes", "max_tool_calls", "max_turns"):
        v = getattr(args, k, None)
        if v is not None:
            budget[k] = v
    out.mkdir(parents=True, exist_ok=True)
    ids = select_tasks(tasks, args.task_id or None, args.limit, skip=done_tasks(out))
    if args.shard:
        k, n = args.shard
        ids = [t for i, t in enumerate(ids) if i % n == k]
    print(json.dumps({"submission": str(sub), "model": info["model"], "adapters": info["adapters"],
                      "budget": budget, "tasks_to_run": len(ids), "already_done": len(done_tasks(out)),
                      "compaction": not args.no_compaction}, indent=2), flush=True)
    if not ids:
        print("nothing to run")
    else:
        if args.sandbox == "docker":
            docker_preflight(args.image)
        server = None
        previous = _rows(out)
        try:
            if args.server_url:
                from swegemma.cli import _build_cli_model_registry

                os.environ["LOCAL_INFERENCE_URL"] = args.server_url
                models = _build_cli_model_registry(submission_dir=sub)
            else:
                from adk_submission import VllmServer

                server = VllmServer(vllm_config(args.ct, info["model"], cfg, tp=args.tp,
                                                gpu_util=args.gpu_memory_utilization, port=args.port,
                                                extra_args=args.vllm_arg),
                                    adapter_manifest=discover_adapters(sub))
                print("starting vLLM:", " ".join(server.build_cmd()), flush=True)
                server.start()
                models = server.create_model_registry(aliases=[info["model"]])
            ec = build_eval_config(sub, tasks, snapshots, out.resolve(), models, cfg, budget=budget, task_ids=ids,
                                   concurrency=args.concurrency, sandbox=args.sandbox, image=args.image,
                                   compaction=not args.no_compaction, display=args.display)
            asyncio.run(Evaluator(ec).run())
        finally:
            if server is not None:
                server.stop()
            merge_results(out, previous)
    summary = summarize(out)
    summary.update({"submission": str(sub), "model": info["model"], "adapters": info["adapters"],
                    "budget": budget, "compaction": not args.no_compaction})
    (out / "locfix_summary.json").write_text(json.dumps(summary, indent=2))
    write_run_meta(out, cfg, {"eval": summary, "argv_args": {k: (v if not isinstance(v, Path) else str(v))
                                                             for k, v in vars(args).items()}})
    return summary


# --------------------------------------------------------------------------- CLI
def add_args_cmd_run(p: argparse.ArgumentParser) -> None:
    from ..data.build_common import parse_shard

    p.add_argument("--submission", required=True, help="folder made by `locfix submit.package`")
    p.add_argument("--ct", help="the -w4a16-ct checkpoint folder vLLM serves (not needed with --server-url)")
    p.add_argument("--tasks", required=True, help="tasks.jsonl in the harness's format")
    p.add_argument("--snapshots", required=True, help="repository snapshots folder for those tasks")
    p.add_argument("--out", required=True, help="results folder (re-running continues it)")
    p.add_argument("--task-id", action="append", default=[], help="only these instance ids (repeatable)")
    p.add_argument("--limit", type=int, default=0, help="first N tasks (a quick check)")
    p.add_argument("--shard", type=parse_shard, default=None, help="K/N: every N-th task starting at K")
    p.add_argument("--concurrency", type=int, default=8, help="tasks in parallel (each sandbox: 2 vCPU, 4 GB)")
    p.add_argument("--sandbox", choices=["docker", "subprocess"], default="docker")
    p.add_argument("--image", default="swebench-sandbox:latest", help="sandbox docker image")
    p.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel size")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--vllm-arg", action="append", default=[], help="extra vLLM flag (repeatable, e.g. --vllm-arg=--data-parallel-size=4)")
    p.add_argument("--server-url", help="use a running OpenAI-compatible server (e.g. http://127.0.0.1:8000/v1)")
    p.add_argument("--max-time-minutes", type=float, help="override the submission's budget (not recommended)")
    p.add_argument("--max-tool-calls", type=int)
    p.add_argument("--max-turns", type=int)
    p.add_argument("--no-compaction", action="store_true", help="disable Kaggle's context compaction")
    p.add_argument("--display", default="quiet", choices=["auto", "dashboard", "single", "quiet"])


def cmd_run(args: argparse.Namespace, cfg: dict) -> int:
    if not args.server_url and not args.ct:
        raise SystemExit("--ct is required (or --server-url for a running server)")
    s = run_eval(args, cfg)
    print(json.dumps({k: s[k] for k in ("tasks", "resolved", "resolve_rate", "ci95", "empty_patch", "errors")},
                     indent=2))
    return 0
