"""Run metadata written next to every output: config, git SHA, versions, command line."""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import REPO_ROOT

TRACKED_PACKAGES = (
    "locfix", "swegemma", "adk-submission", "adk-eval-core", "google-adk", "litellm",
    "transformers", "peft", "trl", "accelerate", "torch", "vllm", "liger-kernel",
)


def git_sha(root: Path = REPO_ROOT) -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                               text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = _md.version(name)
        except _md.PackageNotFoundError:
            out[name] = None
    return out


def write_run_meta(out_dir: str | Path, config: dict, extra: dict[str, Any] | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "created": _dt.datetime.now(_dt.UTC).isoformat(),
        "argv": sys.argv,
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "versions": package_versions(),
        "config": config,
        **(extra or {}),
    }
    path = out / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    return path
