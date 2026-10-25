"""Run the harness's own workspace tools (`read_file`, `edit_file`, `write_file`) on local files.

The swegemma tools only touch the sandbox through `ctx.docker.copy_from`, `ctx.docker.copy_to`
and `ctx.docker.exec("mkdir -p ...")`. `LocalSandbox` implements those on a local directory that
stands in for `/workspace`, so tool outputs are produced by the harness code itself (same
truncation, same JSON, same 3-tier edit matching and diff) without Docker.
"""
from __future__ import annotations

import json
import shlex
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

WORKSPACE = "/workspace"


class LocalSandbox:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _local(self, container_path: str) -> Path:
        p = str(container_path)
        if p == WORKSPACE:
            rel = ""
        elif p.startswith(WORKSPACE + "/"):
            rel = p[len(WORKSPACE) + 1:]
        else:
            raise ValueError(f"path outside /workspace: {p}")
        return self.root / rel

    # docker-like API used by swegemma.tools.workspace
    def copy_from(self, container_id: str, src: str, dst: str | Path) -> None:  # noqa: ARG002
        local = self._local(src)
        if local.is_file():
            shutil.copyfile(local, dst)

    def copy_to(self, container_id: str, src: str | Path, dst_dir: str) -> None:  # noqa: ARG002
        target_dir = self._local(dst_dir.rstrip("/"))
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target_dir / Path(src).name)

    def exec(self, container_id: str, command: str):  # noqa: ARG002
        toks = shlex.split(command)
        if toks[:2] == ["mkdir", "-p"] and len(toks) == 3:
            self._local(toks[2]).mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(exit_code=0, stdout="", stderr="")
        raise NotImplementedError(f"LocalSandbox.exec supports only 'mkdir -p': {command!r}")


@dataclass
class LocalToolContext:
    """Stand-in for SwegemmaContext with no budget (budget warnings are added separately)."""

    root: Path
    max_file_lines: int = 150
    max_file_chars: int = 10000
    max_stdout_chars: int = 5000
    docker: LocalSandbox = field(init=False)
    container_id: str = "local"
    tool_calls_used: int = 0
    tool_lock: Any = None

    def __post_init__(self) -> None:
        self.docker = LocalSandbox(self.root)
        self.harness = SimpleNamespace(max_file_lines=self.max_file_lines,
                                       max_file_chars=self.max_file_chars,
                                       max_stdout_chars=self.max_stdout_chars,
                                       command_timeout_seconds=300)
        self.budget = SimpleNamespace(tool_calls=None, turns=None, cost_usd=None,
                                      total_tokens=None, time_minutes=None)

    def check_budget(self, *, count_tool_call: bool = True) -> str | None:  # noqa: ARG002
        return None


class LocalWorkspace:
    """A temporary /workspace built from {relative_path: text}."""

    def __init__(self, files: dict[str, str] | None = None, root: str | Path | None = None):
        self._tmp = None
        if root is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="locfix_ws_")
            root = self._tmp.name
        self.root = Path(root)
        for rel, text in (files or {}).items():
            self.write_text(rel, text)
        self.ctx = LocalToolContext(self.root)

    def __enter__(self) -> "LocalWorkspace":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    @staticmethod
    def rel(path: str) -> str:
        p = path[len(WORKSPACE) + 1:] if path.startswith(WORKSPACE + "/") else path
        return p.lstrip("/")

    def write_text(self, rel: str, text: str) -> None:
        p = self.root / self.rel(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(text)

    def read_text(self, rel: str) -> str:
        with open(self.root / self.rel(rel), encoding="utf-8", newline="") as f:
            return f.read()

    def exists(self, rel: str) -> bool:
        return (self.root / self.rel(rel)).is_file()

    # harness tools -----------------------------------------------------------------
    def read_file(self, filepath: str, start_line: int | None = None, end_line: int | None = None) -> str:
        from swegemma.tools.workspace import read_file

        return read_file(self.ctx, filepath, start_line, end_line)

    def edit_file(self, filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str:
        from swegemma.tools.workspace import edit_file

        return edit_file(self.ctx, filepath, old_string, new_string, allow_multiple)

    def write_file(self, filepath: str, content: str) -> str:
        from swegemma.tools.workspace import write_file

        return write_file(self.ctx, filepath, content)

    def call(self, tool: str, args: dict[str, Any]) -> str:
        if tool == "read_file":
            return self.read_file(args["filepath"], args.get("start_line"), args.get("end_line"))
        if tool == "edit_file":
            return self.edit_file(args["filepath"], args["old_string"], args["new_string"],
                                  args.get("allow_multiple", False))
        if tool == "write_file":
            return self.write_file(args["filepath"], args["content"])
        raise ValueError(f"LocalWorkspace cannot run {tool!r}")


def response_ok(raw: str) -> bool:
    try:
        return json.loads(raw).get("status") == "ok"
    except (json.JSONDecodeError, AttributeError):
        return False
