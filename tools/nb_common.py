"""Notebook cells shared by the Colab notebooks (code checkout, packages, tokenizer) and the writer.

Each `*_CELL` is the source of one code cell. The notebooks are generated files: edit the
builders in tools/, then re-run them.
"""
from __future__ import annotations

import json
from pathlib import Path

GET_CODE_CELL = r'''
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
'''

_PACKAGES = r'''
# --- Python packages
# From WHEELS_DIR only the three harness wheels are installed (pure Python, any Python >= 3.12);
# the rest of the Kaggle wheelhouse (vLLM, cp312 builds, ...) is not needed and would not install.
# Versions are pinned to the ones the code is tested with (also on Python 3.13).
import importlib.util
import importlib.metadata as md

HARNESS_WHEELS = ("swegemma-", "adk_submission-", "adk_eval_core-")
PINS = ["google-adk==1.39.1", "litellm==1.106.3", "transformers==5.13.1", "docker", "cachetools",
        "networkx", "python-dotenv", "rich", "pandas", "safetensors", "pyyaml", "jsonschema"]


def missing(mod):
    return importlib.util.find_spec(mod) is None


def pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise RuntimeError("pip install failed; see the output above")


if __NEED_HARNESS__ and (missing("swegemma") or missing("adk_submission") or missing("adk_eval_core")):
    wheels = [p for p in sorted(pathlib.Path(WHEELS_DIR).glob("*.whl")) if p.name.startswith(HARNESS_WHEELS)]
    found = sorted({w.name.split("-")[0] for w in wheels})
    assert len(found) == 3, f"need swegemma, adk_submission and adk_eval_core wheels in {WHEELS_DIR}; found {found}"
    pip("--no-deps", *map(str, wheels))   # their torch/torchvision/accelerate requirements are not used here
    pip(*PINS)
    pip("mcp==2.3.0")                     # litellm 1.106.3 needs mcp 2.x; google-adk declares mcp<2 but works
                                          # with it (pip prints a dependency warning: expected)
try:
    tf_ok = md.version("transformers") == "5.13.1"
except md.PackageNotFoundError:
    tf_ok = False
if not tf_ok:
    pip("transformers==5.13.1", "pyyaml", "jsonschema", "pandas")


def version(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return "-"


print("python", sys.version.split()[0],
      {p: version(p) for p in ("transformers", "swegemma", "adk-submission", "google-adk", "litellm", "mcp")})
'''


def packages_cell(need_harness: str = "True") -> str:
    """`need_harness`: Python expression; when true the three harness wheels are installed."""
    return _PACKAGES.replace("__NEED_HARNESS__", need_harness)


TOKENIZER_CELL = r'''
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
'''


def write_notebook(cells: list[tuple[str, str]], dest: Path) -> None:
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
          "cells": []}
    for i, (kind, src) in enumerate(cells):
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {dest} ({len(cells)} cells)")
