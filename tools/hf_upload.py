#!/usr/bin/env python3
"""Upload training outputs (folders or files) to a Hugging Face repository.

    export HF_TOKEN=hf_xxx                      # a WRITE token (safer than --token: not kept in shell history)
    python tools/hf_upload.py --user Hennara --repo locfix-e4b-runs \
        --path work/runs/S3r0-e4b --path work/s3r0 --path work/records/folds/folds.json

Each --path lands in the repository under its own name (work/runs/S3r0-e4b -> S3r0-e4b/,
work/s3r0 -> s3r0/, folds.json -> folds.json); `--path-in-repo` changes that for a single --path.

- The repository is created if it does not exist, **private** unless --public.
- --repo-type: model (default; adapters, logs) or dataset (records, scores).
- Folders go up with `upload_large_folder`: many files and large files (multi-GB JSONL) in parallel,
  and an interrupted upload continues where it stopped when started again with the same command
  (it keeps its progress in <folder's parent>/.cache/huggingface/). --single-commit uses
  `upload_folder` instead (one commit, nothing resumable; fine for small folders).
- --exclude skips files by pattern (repeatable), e.g. --exclude "*/epoch-*/*" to keep only
  export/final, or --exclude "*.log".
- --dry-run lists what would be uploaded, with sizes, and uploads nothing.

Needs only `huggingface_hub` (pip install -U huggingface_hub).
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path

DEFAULT_EXCLUDE = ["**/__pycache__/**", "**/.cache/**", "*.tmp"]


def files_of(path: Path, exclude: list[str]) -> list[tuple[str, int]]:
    """(path inside the upload, size) of every file that would be uploaded."""
    if path.is_file():
        return [(path.name, path.stat().st_size)]
    out = []
    for f in sorted(path.rglob("*")):
        if not f.is_file():
            continue
        rel = f"{path.name}/{f.relative_to(path).as_posix()}"
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(f.relative_to(path).as_posix(), pat) for pat in exclude):
            continue
        out.append((rel, f.stat().st_size))
    return out


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def upload_one(api, repo_id: str, repo_type: str, path: Path, dest: str, exclude: list[str],
               single_commit: bool, workers: int | None) -> None:
    if path.is_file():
        api.upload_file(path_or_fileobj=str(path), path_in_repo=dest, repo_id=repo_id, repo_type=repo_type,
                        commit_message=f"upload {dest}")
        return
    if single_commit or dest != path.name:
        api.upload_folder(folder_path=str(path), path_in_repo=dest, repo_id=repo_id, repo_type=repo_type,
                          ignore_patterns=exclude, commit_message=f"upload {dest}/")
        return
    # resumable, parallel: upload the folder from its parent so the files keep the folder's name
    api.upload_large_folder(repo_id=repo_id, folder_path=str(path.parent), repo_type=repo_type,
                            allow_patterns=[f"{path.name}/**"],
                            ignore_patterns=[f"{path.name}/{p}" if not p.startswith("**") else p for p in exclude]
                            + [".cache/**"], num_workers=workers)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True, help="Hugging Face user or organization, e.g. Hennara")
    p.add_argument("--repo", required=True, help="repository name, e.g. locfix-e4b-runs")
    p.add_argument("--path", action="append", required=True, help="folder or file to upload (repeatable)")
    p.add_argument("--token", help="WRITE token (default: the HF_TOKEN environment variable)")
    p.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    p.add_argument("--public", action="store_true", help="create the repository public (default: private)")
    p.add_argument("--path-in-repo", help="destination inside the repository (only with a single --path)")
    p.add_argument("--exclude", action="append", default=[], help="glob of files to skip (repeatable)")
    p.add_argument("--single-commit", action="store_true", help="folders in one commit (not resumable)")
    p.add_argument("--workers", type=int, help="parallel upload workers (default: chosen by huggingface_hub)")
    p.add_argument("--dry-run", action="store_true", help="list the files and sizes, upload nothing")
    a = p.parse_args(argv)

    paths = [Path(x).expanduser() for x in a.path]
    missing = [str(x) for x in paths if not x.exists()]
    if missing:
        print(f"not found: {missing}", file=sys.stderr)
        return 2
    if a.path_in_repo and len(paths) != 1:
        print("--path-in-repo needs exactly one --path", file=sys.stderr)
        return 2
    exclude = DEFAULT_EXCLUDE + a.exclude
    repo_id = f"{a.user}/{a.repo}"

    total = 0
    for x in paths:
        fs = files_of(x, exclude)
        size = sum(s for _, s in fs)
        total += size
        dest = a.path_in_repo or x.name
        print(f"{x}  ->  {repo_id}:{dest}{'/' if x.is_dir() else ''}   {len(fs)} file(s), {human(size)}")
        if a.dry_run:
            for rel, s in fs[:30]:
                print(f"    {rel}  ({human(s)})")
            if len(fs) > 30:
                print(f"    ... and {len(fs) - 30} more")
    print(f"total {human(total)}")
    if a.dry_run:
        return 0

    token = a.token or os.environ.get("HF_TOKEN")
    if not token:
        print("no token: pass --token or set HF_TOKEN (a token with WRITE access)", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()
    print(f"logged in as {who.get('name')}")
    api.create_repo(repo_id, repo_type=a.repo_type, private=not a.public, exist_ok=True)
    for x in paths:
        dest = a.path_in_repo or x.name
        print(f"uploading {x} -> {dest} ...", flush=True)
        upload_one(api, repo_id, a.repo_type, x, dest, exclude, a.single_commit, a.workers)
    base = "https://huggingface.co/" + ("datasets/" if a.repo_type == "dataset" else "")
    print(f"done: {base}{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
