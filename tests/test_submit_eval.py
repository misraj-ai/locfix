"""Submission packaging, evaluation statistics and the evaluation runner's configuration.

The end-to-end checks replay real fixture trajectories through *packaged* submissions with the
harness's own agent stack: the agents compile from the packaged YAML, and every request reaches
the server under the adapter name of the agent that made it (that is how vLLM picks the LoRA)."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from locfix.config import load_config
from locfix.eval.stats import compare, mcnemar_exact, summarize, wilson
from locfix.submit.package import PackageError, package, validate_submission

ROOT = Path(__file__).resolve().parents[1]
E4B = "gemma-4-e4b-it-qat-w4a16-ct"


def fake_run(tmp: Path, name: str, rank: int = 8, parity_ok: bool | None = True) -> Path:
    """<tmp>/<name>/export/final (adapter) + <tmp>/<name>/parity/parity.json, as the recipe lays out."""
    d = tmp / name / "export" / "final"
    d.mkdir(parents=True)
    save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(rank, 16)},
              str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": 2 * rank, "peft_type": "LORA",
                                                        "target_modules": ["q_proj"]}))
    if parity_ok is not None:
        (tmp / name / "parity").mkdir()
        (tmp / name / "parity" / "parity.json").write_text(json.dumps(
            {"ok": parity_ok, "base": {"mean": 0.024}, "adapter": {"mean": 0.03}}))
    return d


def yaml_line(path: Path, key: str) -> str | None:
    for line in path.read_text().splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    return None


# --------------------------------------------------------------------------- packaging
def test_package_single(tmp_path):
    a = fake_run(tmp_path, "S1")
    rep = package("single", E4B, {"agent": a}, tmp_path / "sub", budget={"max_tool_calls": 28}, seed=3)
    sub = tmp_path / "sub"
    assert yaml_line(sub / "agent.yaml", "model") == E4B and yaml_line(sub / "agent.yaml", "adapter") == "agent_lora"
    assert sorted(p.name for p in (sub / "adapters" / "agent_lora").iterdir()) == ["adapter_config.json",
                                                                                   "adapter_model.safetensors"]
    assert rep["budget"] == {"max_time_minutes": 8, "max_tool_calls": 28, "max_turns": 300}
    assert yaml_line(sub / "configs" / "sampling.yaml", "seed") == "3"
    assert rep["adapters"]["agent_lora"]["parity"]["adapter_mean"] == 0.03
    assert (tmp_path / "sub.package.json").exists() and not (sub / "sub.package.json").exists()
    info = validate_submission(sub)
    assert info["model"] == E4B and info["adapters"] == {"agent_lora": 8} and info["budget"]["max_tool_calls"] == 28


def test_package_locfix_two_adapters_and_one_model(tmp_path):
    loc, fix = fake_run(tmp_path, "loc"), fake_run(tmp_path, "fix")
    package("locfix", E4B, {"localizer": loc, "fixer": fix}, tmp_path / "two")
    assert yaml_line(tmp_path / "two" / "agent.yaml", "adapter") == "fix_lora"
    assert yaml_line(tmp_path / "two" / "sub_agents" / "localizer.yaml", "adapter") == "loc_lora"
    assert validate_submission(tmp_path / "two")["adapters"] == {"fix_lora": 8, "loc_lora": 8}
    # the same folder for both roles: one adapter, used by both agents
    rep = package("locfix", E4B, {"localizer": fix, "fixer": fix}, tmp_path / "one")
    assert rep["shared_adapter"] and rep["roles"] == {"localizer": "locfix_lora", "fixer": "locfix_lora"}
    assert yaml_line(tmp_path / "one" / "agent.yaml", "adapter") == "locfix_lora"
    assert yaml_line(tmp_path / "one" / "sub_agents" / "localizer.yaml", "adapter") == "locfix_lora"
    assert [p.name for p in (tmp_path / "one" / "adapters").iterdir()] == ["locfix_lora"]
    # a sub-agent YAML keeps its !include lines untouched
    assert "!include prompts/localizer.md" in (tmp_path / "one" / "sub_agents" / "localizer.yaml").read_text()


def test_package_base_model(tmp_path):
    package("locfix", E4B, {}, tmp_path / "base")
    assert yaml_line(tmp_path / "base" / "agent.yaml", "adapter") is None
    assert not (tmp_path / "base" / "adapters").exists()
    assert validate_submission(tmp_path / "base")["adapters"] == {}


def test_package_refusals(tmp_path):
    good = fake_run(tmp_path, "good")
    with pytest.raises(PackageError, match="no parity report"):
        package("single", E4B, {"agent": fake_run(tmp_path, "noparity", parity_ok=None)}, tmp_path / "x1")
    with pytest.raises(PackageError, match="parity failed"):
        package("single", E4B, {"agent": fake_run(tmp_path, "bad", parity_ok=False)}, tmp_path / "x2")
    with pytest.raises(PackageError, match="rank 256"):
        package("single", E4B, {"agent": fake_run(tmp_path, "big", rank=256)}, tmp_path / "x3")
    with pytest.raises(PackageError, match="unknown role"):
        package("single", E4B, {"fixer": good}, tmp_path / "x4")
    with pytest.raises(PackageError, match="not one of"):
        package("single", "gemma-4-31b-it", {"agent": good}, tmp_path / "x5")
    package("single", E4B, {"agent": good}, tmp_path / "x6")
    with pytest.raises(PackageError, match="exists"):
        package("single", E4B, {"agent": good}, tmp_path / "x6")
    package("single", E4B, {"agent": good}, tmp_path / "x6", force=True)
    # no parity needed when asked explicitly
    package("single", E4B, {"agent": fake_run(tmp_path, "np2", parity_ok=None)}, tmp_path / "x7", parity_check=False)
    # a symlink inside a submission is refused
    (tmp_path / "x7" / "link.md").symlink_to(tmp_path / "x7" / "agent.yaml")
    with pytest.raises(PackageError, match="symlink"):
        validate_submission(tmp_path / "x7")


# --------------------------------------------------------------------------- replay through packaged submissions
def test_packaged_single_agent_runs_in_the_harness_with_its_adapter(tmp_path, gunicorn_path):
    from locfix.data.agent_records import BuildConfig, build_agent_records
    from locfix.harness import replay as rp
    from locfix.harness.prompts import Budget

    package("single", E4B, {"agent": fake_run(tmp_path, "S1")}, tmp_path / "sub", seed=7)
    seen: list[str] = []
    seeds: set = set()
    orig = rp._Player.handle

    def spy(self, req):
        seen.append(req.get("model"))
        seeds.add(req.get("seed"))
        return orig(self, req)

    rp._Player.handle = spy
    try:
        from locfix.data.traces import load_trace

        recs = build_agent_records(load_trace(gunicorn_path),
                                   BuildConfig(agent_dir=str(tmp_path / "sub"), model_alias=E4B,
                                               budget=Budget(8, 100, 300)),
                                   {"set": "solved", "dataset": "swe_smith", "fold": "A", "attempt": 1})
    finally:
        rp._Player.handle = orig
    assert recs and seen and set(seen) == {"agent_lora"}
    assert seeds == {7}                                   # the packaged sampling seed reaches the server


def test_packaged_locfix_routes_each_agent_to_its_adapter(tmp_path, gunicorn_path):
    from test_loc_fix_records import make_task

    from locfix.data.loc_fix_records import FIXER, LOCALIZER, LocFixConfig, replay_variant, solved_variants, \
        task_locations
    from locfix.harness.prompts import Budget

    package("locfix", E4B, {"localizer": fake_run(tmp_path, "loc"), "fixer": fake_run(tmp_path, "fix")},
            tmp_path / "sub")
    task = make_task(gunicorn_path)
    em = load_config()["em"]
    tl = task_locations(task)
    import random

    v = next(x for x in solved_variants(tl, task.files, em, random.Random(0)) if x.name == "gold")
    cfg = LocFixConfig(agent_dir=str(tmp_path / "sub"), model_alias=E4B, budget=Budget(8, 100, 300), em=em)
    res, _ = replay_variant(task, tl, v, cfg)
    models = {s: {e.request.get("model") for e, st in zip(res.agent_exchanges, res.streams, strict=True) if st == s}
              for s in (FIXER, LOCALIZER)}
    assert models == {FIXER: {"fix_lora"}, LOCALIZER: {"loc_lora"}}


# --------------------------------------------------------------------------- statistics
def write_results(d: Path, resolved: dict[str, bool], repo: str = "o/r") -> Path:
    d.mkdir(parents=True)
    (d / "task_results.jsonl").write_text("".join(
        json.dumps({"instance_id": k, "repo": repo, "resolved": v, "agent_patch_size": 10 if v else 0,
                    "duration_seconds": 100.0, "tool_calls": 12, "total_llm_calls": 15, "error": None}) + "\n"
        for k, v in resolved.items()))
    return d


def test_wilson_and_mcnemar_known_values():
    lo, hi = wilson(40, 100)
    assert round(lo, 3) == 0.309 and round(hi, 3) == 0.498
    assert wilson(0, 0) == (0.0, 0.0)
    assert round(mcnemar_exact(10, 2), 4) == 0.0386            # 2 * P(X <= 2), X ~ Bin(12, 0.5)
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(5, 5) == 1.0


def test_summarize_and_compare(tmp_path):
    tasks = [f"t{i}" for i in range(20)]
    a = {t: i < 8 for i, t in enumerate(tasks)}                       # A solves t0..t7
    b = {t: 2 <= i < 14 for i, t in enumerate(tasks)}                 # B solves t2..t13
    da, db = write_results(tmp_path / "a", a), write_results(tmp_path / "b", b)
    s = summarize(da)
    assert s["tasks"] == 20 and s["resolved"] == 8 and s["resolve_rate"] == 0.4 and s["empty_patch"] == 12
    c = compare([str(da)], [str(db)], "S2b", "S3")
    assert (c["both"], c["only_a"], c["only_b"], c["neither"]) == (6, 2, 6, 6)
    assert c["rate_S2b"] == 0.4 and c["rate_S3"] == 0.6 and c["difference_b_minus_a"] == 0.2
    assert c["p_mcnemar"] == round(mcnemar_exact(2, 6), 4)
    with pytest.raises(ValueError):
        compare([str(da)], [str(db), str(db)])


# --------------------------------------------------------------------------- runner configuration
def test_eval_config_uses_submission_budget_and_kaggle_settings(tmp_path):
    from adk_submission import ModelRegistry

    from locfix.eval.run import build_eval_config, done_tasks, merge_results, select_tasks, vllm_config

    cfg = load_config()
    package("locfix", E4B, {"fixer": fake_run(tmp_path, "fix")}, tmp_path / "sub", budget={"max_tool_calls": 30})
    info = validate_submission(tmp_path / "sub")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(json.dumps({"instance_id": f"t{i}"}) + "\n" for i in range(5)))
    assert select_tasks(tasks, limit=2) == ["t0", "t1"]
    assert select_tasks(tasks, ["t3", "t1"]) == ["t1", "t3"]
    assert select_tasks(tasks, skip={"t0", "t1"}) == ["t2", "t3", "t4"]
    ec = build_eval_config(tmp_path / "sub", tasks, tmp_path, tmp_path / "out", ModelRegistry(), cfg,
                           budget=info["budget"], task_ids=["t0"], concurrency=4, sandbox="subprocess",
                           image="x")
    assert (ec.budget.time_minutes, ec.budget.tool_calls, ec.budget.turns) == (8, 30, 300)
    comp = ec.events_compaction_config
    assert (comp.compaction_interval, comp.overlap_size, comp.token_threshold, comp.event_retention_size) == (15, 2, 14336, 5)
    cache = ec.context_cache_config
    assert (cache.min_tokens, cache.ttl_seconds, cache.cache_intervals) == (2048, 1800, 10)
    assert build_eval_config(tmp_path / "sub", tasks, tmp_path, tmp_path / "out", ModelRegistry(), cfg,
                             budget=info["budget"], task_ids=["t0"], concurrency=1, sandbox="subprocess",
                             image="x", compaction=False).events_compaction_config is None

    # vLLM command as the scorer builds it, with the submission's adapters
    from adk_submission import VllmServer, discover_adapters

    cmd = VllmServer(vllm_config("/models/ct", E4B, cfg, tp=1, gpu_util=0.9, port=8000),
                     adapter_manifest=discover_adapters(tmp_path / "sub")).build_cmd()
    s = " ".join(cmd)
    for part in ("--tool-call-parser gemma4", "--reasoning-parser gemma4", "--max-model-len 32768",
                 "--enable-lora", "--max-loras 1", "--lora-modules fix_lora=", '"reasoning_start_str": "<|channel>"'):
        assert part in s, part
    assert "--hf-overrides" not in s
    assert "--hf-overrides" in " ".join(VllmServer(vllm_config("/m", "gemma-4-12b-it-qat-w4a16-ct", cfg, tp=1,
                                                               gpu_util=0.9, port=8000)).build_cmd())

    # resuming keeps the rows of earlier runs (the harness rewrites the file with only the new run)
    out = tmp_path / "res"
    out.mkdir()
    (out / "task_results.jsonl").write_text(json.dumps({"instance_id": "t0", "resolved": True}) + "\n")
    prev = {"t0": json.dumps({"instance_id": "t0", "resolved": True})}
    (out / "task_results.jsonl").write_text(json.dumps({"instance_id": "t1", "resolved": False}) + "\n")
    merge_results(out, prev)
    assert done_tasks(out) == {"t0", "t1"}


def test_cli_help_renders():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "locfix.cli", "--help"], cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0 and "eval.run" in r.stdout and "submit.package" in r.stdout
