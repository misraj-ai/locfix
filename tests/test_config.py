from pathlib import Path

import pytest

from locfix.config import ConfigError, deep_merge, load_config, resolve_refs


def test_defaults_load_and_resolve():
    cfg = load_config()
    assert cfg["versions"]["vllm"] == "0.19.1"
    assert cfg["versions"]["transformers"] == "5.13.1"
    assert cfg["models"]["e4b"]["dequantized"].endswith("gemma4-e4b-w4a16-dequant-bf16")
    assert "${" not in cfg["models"]["31b"]["dequantized"]
    assert cfg["budget"]["max_tool_calls"] == 36
    assert cfg["budget_locfix"]["max_tool_calls"] == 44
    assert cfg["data"]["max_trace_tokens"] == cfg["train"]["max_seq_len"] == 32768


def test_time_scale_from_cfg():
    from locfix.harness.budget import time_scale_from_cfg

    cfg = load_config()
    assert time_scale_from_cfg(cfg, "budget") == 1.0                        # auto: 8-min teacher, 8-min budget
    assert time_scale_from_cfg(load_config(None, ["budget.max_time_minutes=5"])) == pytest.approx(5 / 8)
    assert time_scale_from_cfg(load_config(None, ["data.time_scale=2.5"])) == 2.5
    assert time_scale_from_cfg(load_config(None, ["data.teacher_time_minutes=null"])) == 1.0
    with pytest.raises(ValueError):
        time_scale_from_cfg(load_config(None, ["data.time_scale=0"]))


def test_overrides_and_run_config(tmp_path: Path):
    rc = tmp_path / "run.yaml"
    rc.write_text("train:\n  lr:\n    e4b: 4.0e-4\n")
    cfg = load_config([rc], ["budget.max_tool_calls=22", "em.K=16"])
    assert cfg["train"]["lr"]["e4b"] == 4.0e-4
    assert cfg["train"]["lr"]["31b"] == 1.0e-4  # untouched sibling
    assert cfg["budget"]["max_tool_calls"] == 22
    assert cfg["em"]["K"] == 16


def test_refs_keep_type_and_detect_missing():
    cfg = resolve_refs({"a": {"b": 3}, "c": "${a.b}", "d": "x-${a.b}"})
    assert cfg["c"] == 3 and cfg["d"] == "x-3"
    with pytest.raises(ConfigError):
        resolve_refs({"c": "${nope.key}"})


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    out = deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}} and out == {"a": {"b": 1, "c": 2}}


def test_bad_override():
    with pytest.raises(ConfigError):
        load_config(overrides=["no_equals_sign"])
