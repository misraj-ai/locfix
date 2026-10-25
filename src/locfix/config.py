"""Configuration loading.

A config is built from, in order (later wins):
  1. configs/defaults.yaml
  2. configs/paths.yaml (if present; machine-specific, gitignored)
  3. any run configs passed with --config
  4. dotted overrides passed as key=value (values parsed as YAML)

String values may reference other keys as ${dotted.key}; references are resolved after merging.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_REF = re.compile(r"\$\{([A-Za-z0-9_.\-]+)\}")


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, other: dict) -> dict:
    """Return a new dict: `other` merged into `base` (dicts merge recursively, others replace)."""
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_dotted(cfg: dict, key: str) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"missing config key: {key}")
        cur = cur[part]
    return cur


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
        if not isinstance(cur, dict):
            raise ConfigError(f"cannot set {key}: {part} is not a mapping")
    cur[parts[-1]] = value


def resolve_refs(cfg: dict) -> dict:
    """Resolve ${a.b} references inside string values (repeatedly, to allow chains)."""
    root = copy.deepcopy(cfg)

    def resolve(value: Any, depth: int = 0) -> Any:
        if depth > 20:
            raise ConfigError("reference cycle in config")
        if isinstance(value, str):
            full = _REF.fullmatch(value)
            if full:  # keep the referenced value's type
                return resolve(get_dotted(root, full.group(1)), depth + 1)
            return _REF.sub(lambda m: str(resolve(get_dotted(root, m.group(1)), depth + 1)), value)
        if isinstance(value, dict):
            return {k: resolve(v, depth) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v, depth) for v in value]
        return value

    return resolve(root)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def load_config(
    run_configs: list[str | Path] | None = None,
    overrides: list[str] | None = None,
    root: Path = REPO_ROOT,
    resolve: bool = True,
) -> dict:
    cfg = load_yaml(root / "configs" / "defaults.yaml")
    paths = root / "configs" / "paths.yaml"
    if paths.exists():
        cfg = deep_merge(cfg, load_yaml(paths))
    for rc in run_configs or []:
        cfg = deep_merge(cfg, load_yaml(rc))
    for ov in overrides or []:
        if "=" not in ov:
            raise ConfigError(f"override must be key=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return resolve_refs(cfg) if resolve else cfg
