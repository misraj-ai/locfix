"""`locfix score.run`: score Stage 3 sequences with the cross-fitted fixer (spec 06 §4), GPU.

For each sequence of `score.prepare` (fold F task), the fixer trained on the **other** fold
(`--fix-A` scores fold B tasks, `--fix-B` fold A tasks) gives the log-probability of every y*₊ token
of the verified patch, given the context built from z_k. The mechanism is the one `quant.parity`
validates: vLLM on the real `-w4a16-ct` checkpoint (the class the harness loads), the adapter as a
`LoRARequest`, `prompt_logprobs=0` (the actual token's log-prob), `max_tokens=1`.

    s_k = mean over y*₊ tokens of log p        (spec 06 §4: per-token mean, comparable across tasks)

Output (one row per sequence, `scores.jsonl`): the item's identifiers and sizes plus
`logp_plus_sum`, `logp_plus_mean`, `fixer`. Every fold adapter must have a passing parity report
(`quant.parity --adapter ... --threshold`), as for packaging, unless `--no-parity-check`.

    CUDA_VISIBLE_DEVICES=0 locfix score.run --items work/s3r0/prepare/*/items.jsonl --ct <ct> \
        --fix-A runs/S3r0-e4b/fix_A/export/final --fix-B runs/S3r0-e4b/fix_B/export/final \
        --out work/s3r0/scores/part0.jsonl --shard 0/8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# prompt_logprobs takes a float32 log-softmax over the 262K vocabulary for every scheduled prompt
# token: 8,192 tokens per step need ~8.6 GB on top of weights and KV cache (fine on 80-141 GB GPUs).
SCORER_KW = {"max_num_batched_tokens": 8192, "max_num_seqs": 16, "enable_chunked_prefill": True,
             "enable_prefix_caching": False}
KEEP = ("instance_id", "fold", "cand_id", "source", "mode", "n_requests", "n_request_locations", "n_plus_tokens",
        "n_tokens", "n_z_tokens", "n_ctx_tokens", "n_edits")


def summarize_logprobs(lps: list[float]) -> dict[str, float]:
    if not lps:
        raise ValueError("no y*+ tokens")
    s = float(sum(lps))
    return {"logp_plus_sum": round(s, 5), "logp_plus_mean": round(s / len(lps), 6)}


def iter_items(paths: list[str], shard: tuple[int, int] | None) -> list[tuple[str, str]]:
    """(fold, raw JSON line) of the items in this shard (by task), in a stable order (instance,
    candidate, mode). Lines are parsed only when scored, so host memory stays small."""
    from ..data.build_common import in_shard

    keyed = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    keyed.append(((r["instance_id"], r["cand_id"], r["mode"]), r["fold"], line))
    tasks = sorted({k[0] for k, _, _ in keyed})
    keep = {t for i, t in enumerate(tasks) if in_shard(i, shard)}
    keyed.sort(key=lambda x: x[0])
    return [(fold, line) for k, fold, line in keyed if k[0] in keep]


def check_parity(adapter: str) -> None:
    from ..submit.package import PackageError, check_adapter

    try:
        check_adapter(Path(adapter), parity_check=True)
    except PackageError as e:
        raise SystemExit(f"scorer adapter: {e}") from e


def score(items: list[tuple[str, str]], *, ct: str, adapters: dict[str, str], max_model_len: int,
          gpu_util: float, tp: int, chunk: int, text_only: bool = False):
    """Yields score rows. `adapters`: {"A": fold-A fixer, "B": fold-B fixer}."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    from ..em.folds import other
    from ..quant.parity import TEXT_ONLY_OVERRIDE, vllm_lora_support

    if not text_only and not vllm_lora_support()["Gemma4ForConditionalGeneration"]:
        raise SystemExit("this vLLM build cannot serve LoRA on Gemma4ForConditionalGeneration (the harness "
                         "class): install the competition's vLLM wheel, or pass --vllm-text-only")
    rank = max(int(json.loads((Path(p) / "adapter_config.json").read_text())["r"]) for p in adapters.values())
    kw: dict[str, Any] = dict(model=ct, max_model_len=max_model_len, tensor_parallel_size=tp,
                              gpu_memory_utilization=gpu_util, enable_lora=True, max_loras=len(adapters),
                              max_lora_rank=rank, **SCORER_KW)
    if text_only:
        kw["hf_overrides"] = TEXT_ONLY_OVERRIDE
    llm = LLM(**kw)
    loras = {f: LoRARequest(f"fix_{f}", i + 1, p) for i, (f, p) in enumerate(sorted(adapters.items()))}
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    for fold in ("A", "B"):
        ts = [line for f, line in items if other(f) == fold]
        for s in range(0, len(ts), chunk):
            part = [json.loads(line) for line in ts[s:s + chunk]]
            ok = [it for it in part if len(it["input_ids"]) < max_model_len]
            for it in part:
                if len(it["input_ids"]) >= max_model_len:
                    yield {**{k: it.get(k) for k in KEEP}, "fixer": f"fix_{fold}", "error": "too_long"}
            outs = llm.generate([TokensPrompt(prompt_token_ids=it["input_ids"]) for it in ok], sp,
                                lora_request=loras[fold], use_tqdm=False) if ok else []
            for it, o in zip(ok, outs, strict=True):
                plp = o.prompt_logprobs
                lps = [float(plp[t][it["input_ids"][t]].logprob) for t in it["positions"]]
                yield {**{k: it.get(k) for k in KEEP}, "fixer": f"fix_{fold}", **summarize_logprobs(lps)}


# --------------------------------------------------------------------------- CLI
def add_args_cmd_run(p: argparse.ArgumentParser) -> None:
    from ..data.build_common import parse_shard

    p.add_argument("--items", nargs="+", required=True, help="items.jsonl file(s) from score.prepare")
    p.add_argument("--ct", required=True, help="the -w4a16-ct checkpoint")
    p.add_argument("--fix-A", required=True, help="exported fold-A fixer (scores fold B tasks)")
    p.add_argument("--fix-B", required=True, help="exported fold-B fixer (scores fold A tasks)")
    p.add_argument("--out", required=True, help="scores JSONL")
    p.add_argument("--shard", type=parse_shard, default=None, help="K/N by task (one GPU per shard)")
    p.add_argument("--chunk", type=int, default=64, help="sequences per vLLM call (bounds host memory)")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--no-parity-check", action="store_true", help="score with adapters that have no passing parity.json")
    p.add_argument("--vllm-text-only", action="store_true")


def cmd_run(args: argparse.Namespace, cfg: dict) -> int:
    from ..runmeta import write_run_meta

    adapters = {"A": args.fix_A, "B": args.fix_B}
    if not args.no_parity_check:
        for a in adapters.values():
            check_parity(a)
    items = iter_items(args.items, args.shard)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = {"items": len(items), "scored": 0, "errors": 0}
    with open(out, "w") as f:
        for row in score(items, ct=args.ct, adapters=adapters, max_model_len=int(cfg["harness"]["max_model_len"]),
                         gpu_util=args.gpu_memory_utilization, tp=args.tp, chunk=args.chunk,
                         text_only=args.vllm_text_only):
            f.write(json.dumps(row) + "\n")
            n["errors" if row.get("error") else "scored"] += 1
    write_run_meta(out.parent, cfg, {"score": n, "fix_A": args.fix_A, "fix_B": args.fix_B, "shard": args.shard})
    print(json.dumps(n, indent=2))
    return 0
