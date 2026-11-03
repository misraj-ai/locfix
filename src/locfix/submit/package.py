"""`locfix submit.package`: build a submission directory from an agent template and adapters.

What is evaluated is exactly what would be submitted: `locfix eval.run` only accepts a directory
made here (or one that passes the same checks).

    locfix submit.package --structure single --model e4b \
        --adapter agent=work/runs/S1-e4b/export/final --out work/submissions/S1-e4b

    locfix submit.package --structure locfix --model e4b \
        --adapter localizer=work/runs/S2a-e4b/localizer/export/final \
        --adapter fixer=work/runs/S2b-e4b/fixer/export/final --out work/submissions/S2b-e4b

Structures and roles (agents/<structure>/):

| structure | role      | agent file                     | adapter name |
| single    | agent     | agent.yaml                     | agent_lora   |
| locfix    | fixer     | agent.yaml                     | fix_lora     |
| locfix    | localizer | sub_agents/localizer.yaml      | loc_lora     |

The same adapter folder given for both locfix roles ("one model, two roles") is packaged once,
as `locfix_lora`, and both agents point to it. A role without `--adapter` runs on the base model.

Checks (refuse on failure): every adapter has a passing parity report (`quant.parity`, "ok": true)
unless `--no-parity-check`; adapter rank <= 128; one declared model, from the competition's list;
the harness's own `validate_directory` (file types, symlinks, sizes); total size < 3 GiB; the
adapter names the harness discovers equal the names the agents use.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT

ALLOWED_MODELS = ("gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct",
                  "gemma-4-e4b-it-qat-w4a16-ct", "gemma-4-e2b-it-qat-w4a16-ct")
MAX_BYTES = 3 * 1024 ** 3
ROLES: dict[str, dict[str, tuple[str, str]]] = {        # structure -> role -> (agent file, adapter name)
    "single": {"agent": ("agent.yaml", "agent_lora")},
    "locfix": {"fixer": ("agent.yaml", "fix_lora"), "localizer": ("sub_agents/localizer.yaml", "loc_lora")},
}
SHARED_NAME = {"locfix": "locfix_lora"}
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PackageError(ValueError):
    pass


def parity_report_for(adapter_dir: Path) -> Path:
    """Where `quant.parity` writes its report for an exported adapter, following the recipe:
    <run>/export/<name>/ -> <run>/parity/parity.json; an explicit <adapter_dir>/parity.json wins."""
    local = adapter_dir / "parity.json"
    return local if local.exists() else adapter_dir.parents[1] / "parity" / "parity.json"


def check_adapter(adapter_dir: Path, parity_check: bool) -> dict[str, Any]:
    if not adapter_dir.is_dir():
        raise PackageError(f"{adapter_dir}: not a folder")
    for f in ADAPTER_FILES:
        if not (adapter_dir / f).is_file():
            raise PackageError(f"{adapter_dir}: missing {f} (use the exported adapter: <run>/export/<epoch>)")
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r = int(cfg.get("r", 0))
    if not 0 < r <= 128:
        raise PackageError(f"{adapter_dir}: rank {r} (vLLM serves rank 1..128)")
    info: dict[str, Any] = {"path": str(adapter_dir), "rank": r, "lora_alpha": cfg.get("lora_alpha")}
    if parity_check:
        rep = parity_report_for(adapter_dir)
        if not rep.exists():
            raise PackageError(f"{adapter_dir}: no parity report at {rep}; run `locfix quant.parity --adapter "
                               f"{adapter_dir} ... --out {rep.parent} --threshold 0.05` (or --no-parity-check)")
        pr = json.loads(rep.read_text())
        if pr.get("ok") is not True:
            raise PackageError(f"{adapter_dir}: parity failed or ran without --threshold ({rep}: ok={pr.get('ok')})")
        info["parity"] = {"report": str(rep), "base_mean": pr.get("base", {}).get("mean"),
                          "adapter_mean": pr.get("adapter", {}).get("mean")}
    return info


def _set_yaml_key(text: str, key: str, value: str | None, after: str = "model") -> str:
    """Set (or remove, value None) a top-level `key: value` line in an agent YAML, keeping
    everything else (comments, !include tags) as written. A commented `# key: ...` line is
    replaced in place; otherwise the line is added after the `after:` line."""
    pat = re.compile(rf"(?m)^#?\s*{re.escape(key)}:.*$")
    line = f"{key}: {value}" if value is not None else None
    if pat.search(text):
        return pat.sub(line if line is not None else f"# {key}: (none)", text, count=1)
    if line is None:
        return text
    anchor = re.compile(rf"(?m)^{re.escape(after)}:.*$")
    m = anchor.search(text)
    if not m:
        raise PackageError(f"no top-level `{after}:` line to place `{key}` after")
    return text[: m.end()] + "\n" + line + text[m.end():]


def _set_budget(path: Path, budget: dict[str, Any]) -> dict[str, Any]:
    from ..config import _yaml

    data = _yaml(path.read_text()) or {}
    ev = data.setdefault("evaluation", {})
    for k, v in budget.items():
        if v is not None:
            ev[k] = v
    path.write_text("evaluation:\n" + "".join(f"  {k}: {v}\n" for k, v in ev.items()))
    return ev


def dir_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def validate_submission(sub: Path, expect_adapters: dict[str, str] | None = None) -> dict[str, Any]:
    """The checks shared by packaging and `eval.run`. Returns model alias, adapters, budget, size."""
    from adk_submission import discover_adapters, discover_declared_models, validate_directory

    from ..config import _yaml

    if any(p.is_symlink() for p in [sub, *sub.rglob("*")]):
        raise PackageError(f"{sub}: contains symlinks")
    validate_directory(sub)
    models = {m.split("/", 1)[-1] for m in discover_declared_models(sub)}
    if len(models) != 1:
        raise PackageError(f"{sub}: agents declare {sorted(models)}; exactly one base model is allowed")
    (model,) = models
    if model not in ALLOWED_MODELS:
        raise PackageError(f"{sub}: model {model!r} is not one of the competition's {ALLOWED_MODELS}")
    found = {k: (v.rank or 0) for k, v in discover_adapters(sub).adapters.items()}
    for name, r in found.items():
        if not _NAME_RE.match(name):
            raise PackageError(f"adapter name {name!r} does not match {_NAME_RE.pattern}")
        if r > 128:
            raise PackageError(f"adapter {name}: rank {r} > 128")
    if expect_adapters is not None and set(found) != set(expect_adapters.values()):
        raise PackageError(f"adapters found by the harness {sorted(found)} != adapters the agents use "
                           f"{sorted(set(expect_adapters.values()))}")
    size = dir_size(sub)
    if size >= MAX_BYTES:
        raise PackageError(f"{sub}: {size / 1024 ** 3:.2f} GiB >= 3 GiB")
    ev = (_yaml((sub / "eval_config.yaml").read_text()) or {}).get("evaluation", {}) \
        if (sub / "eval_config.yaml").exists() else {}
    return {"model": model, "adapters": found, "budget": ev, "size_bytes": size}


def package(structure: str, model_alias: str, adapters: dict[str, Path], out: Path, *,
            budget: dict[str, Any] | None = None, seed: int | None = None, parity_check: bool = True,
            force: bool = False, template_root: Path | None = None) -> dict[str, Any]:
    if structure not in ROLES:
        raise PackageError(f"structure must be one of {sorted(ROLES)}")
    roles = ROLES[structure]
    unknown = set(adapters) - set(roles)
    if unknown:
        raise PackageError(f"unknown role(s) {sorted(unknown)} for {structure}; roles: {sorted(roles)}")
    if model_alias not in ALLOWED_MODELS:
        raise PackageError(f"model {model_alias!r} is not one of {ALLOWED_MODELS}")
    src = (template_root or REPO_ROOT / "agents") / structure
    if out.exists():
        if not force:
            raise PackageError(f"{out} exists (use --force to replace it)")
        shutil.rmtree(out)

    # adapter names: per role, or one shared name when every adapted role uses the same folder
    resolved = {role: Path(p).resolve() for role, p in adapters.items()}
    shared = len(resolved) > 1 and len(set(resolved.values())) == 1
    names = {role: (SHARED_NAME[structure] if shared else roles[role][1]) for role in resolved}
    infos = {}
    for role, path in resolved.items():
        infos[names[role]] = check_adapter(path, parity_check)

    shutil.copytree(src, out)
    for role, (rel, _) in roles.items():
        f = out / rel
        text = _set_yaml_key(f.read_text(), "model", model_alias)
        text = _set_yaml_key(text, "adapter", names.get(role))
        f.write_text(text)
    for role, path in resolved.items():
        dst = out / "adapters" / names[role]
        if dst.exists():                      # shared adapter: copied once
            continue
        dst.mkdir(parents=True)
        for fname in ADAPTER_FILES:
            shutil.copyfile(path / fname, dst / fname)
    ev = _set_budget(out / "eval_config.yaml", budget or {})
    if seed is not None:
        for samp in out.rglob("sampling.yaml"):
            samp.write_text(_set_yaml_key(samp.read_text(), "seed", str(int(seed)), after="max_output_tokens"))
    check = validate_submission(out, expect_adapters=names)
    report = {"structure": structure, "model": model_alias, "out": str(out), "roles": names,
              "adapters": infos, "shared_adapter": shared, "budget": ev, "seed": seed,
              "size_bytes": check["size_bytes"], "parity_checked": parity_check}
    (out.parent / f"{out.name}.package.json").write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- CLI
def _role_path(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--adapter takes role=folder, got {text!r}")
    role, path = text.split("=", 1)
    return role.strip(), path.strip()


def add_args_cmd_package(p: argparse.ArgumentParser) -> None:
    p.add_argument("--structure", required=True, choices=sorted(ROLES), help="agent template under agents/")
    p.add_argument("--model", default="e4b", help="model key in the config (e4b, 31b, 12b): sets the alias")
    p.add_argument("--adapter", action="append", default=[], type=_role_path, metavar="ROLE=FOLDER",
                   help="exported adapter for a role (single: agent; locfix: fixer, localizer); repeatable")
    p.add_argument("--out", required=True, help="submission folder to create")
    p.add_argument("--max-time-minutes", type=float, help="eval_config budget (default: the template's)")
    p.add_argument("--max-tool-calls", type=int)
    p.add_argument("--max-turns", type=int)
    p.add_argument("--seed", type=int, help="sampling seed written into every configs/sampling.yaml")
    p.add_argument("--no-parity-check", action="store_true", help="package adapters without a passing parity.json")
    p.add_argument("--force", action="store_true", help="replace --out if it exists")
    p.add_argument("--zip", action="store_true", help="also write <out>.zip")


def cmd_package(args: argparse.Namespace, cfg: dict) -> int:
    alias = cfg["models"][args.model]["serve_id"]
    t = args.max_time_minutes
    budget = {"max_time_minutes": (int(t) if t is not None and float(t).is_integer() else t),
              "max_tool_calls": args.max_tool_calls, "max_turns": args.max_turns}
    report = package(args.structure, alias, dict(args.adapter), Path(args.out), budget=budget, seed=args.seed,
                     parity_check=not args.no_parity_check, force=args.force)
    if args.zip:
        report["zip"] = shutil.make_archive(str(Path(args.out)), "zip", root_dir=args.out)
    print(json.dumps(report, indent=2))
    return 0
