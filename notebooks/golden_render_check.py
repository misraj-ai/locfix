"""Golden test: do our rendered prompts match vLLM's token ids exactly?

For every trained assistant turn i of every record, asks vLLM's
`/v1/chat/completions/render` endpoint for the prompt token ids of `messages[:i]` (with the record's
tools and chat_template_kwargs) and compares them with `locfix.render.chat` on the same input.
They must be identical; the training sequence must also start with that prompt.

Run on Kaggle (competition vLLM 0.19.1 + the Gemma 4 model attached; no install needed): upload
the repo (e.g. locfix_code.zip) as a Kaggle Dataset, add it and the model as notebook inputs, then

    !python /kaggle/input/<dataset>/locfix/notebooks/golden_render_check.py \
        --model /kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2

The script adds the repo's src/ to sys.path itself and uses the fixture records by default.
Writes golden_render_report.json (and vllm_golden.log) to /kaggle/working or the current folder.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = 8012
REPO = Path(__file__).resolve().parents[1]
# run from an uploaded copy of the repo without installing it (Kaggle notebooks are offline)
if (REPO / "src" / "locfix").is_dir() and str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
DEFAULT_RECORDS = ",".join(str(REPO / "tests" / "fixtures" / "records" / n) for n in ("s1_fixtures.jsonl", "locfix_fixtures.jsonl"))


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def _wait(proc: subprocess.Popen, timeout: int) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def load_records(records: str | Path) -> list[dict]:
    """Records from one JSONL file or a comma-separated list of files."""
    return [json.loads(line) for p in str(records).split(",") for line in open(p.strip())]


def check(model: str, records: str | Path = DEFAULT_RECORDS, tp: int = 0, timeout: int = 1500, limit: int = 0) -> dict:
    import torch

    from locfix.render.chat import load_tokenizer, render_record, render_text

    tok = load_tokenizer(model)
    recs = load_records(records)
    if limit:
        recs = recs[:limit]
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
           "--served-model-name", "base", "--port", str(PORT),
           "--tensor-parallel-size", str(tp or max(1, torch.cuda.device_count())),
           "--max-model-len", "32768", "--enable-auto-tool-choice",
           "--tool-call-parser", "gemma4", "--reasoning-parser", "gemma4"]
    report: dict = {"model": model, "records": records, "checked": 0, "mismatches": [], "errors": []}
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
    log = open(out_dir / "vllm_golden.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if not _wait(proc, timeout):
            report["errors"].append("vLLM did not start; see vllm_golden.log")
            return report
        for r in recs:
            kw = r["meta"]["request_params"].get("chat_template_kwargs") or {}
            full = render_record(r, tok)
            for e in r["loss"]:
                i = e["message_index"]
                body = {"model": "base", "messages": r["messages"][:i], "tools": r["tools"],
                        "chat_template_kwargs": kw, "add_generation_prompt": True}
                try:
                    theirs = _post("/v1/chat/completions/render", body)["token_ids"]
                except Exception as ex:  # noqa: BLE001
                    report["errors"].append({"record": r["id"], "message": i, "error": repr(ex)})
                    continue
                ours = tok(render_text(tok, r["messages"][:i], r["tools"], kw, True),
                           add_special_tokens=False)["input_ids"]
                report["checked"] += 1
                if ours != theirs:
                    k = next((j for j, (a, b) in enumerate(zip(ours, theirs, strict=False)) if a != b), min(len(ours), len(theirs)))
                    report["mismatches"].append({
                        "record": r["id"], "message": i, "ours_len": len(ours), "theirs_len": len(theirs),
                        "first_diff": k,
                        "ours_ctx": tok.decode(ours[max(0, k - 20):k + 20]),
                        "theirs_ctx": tok.decode(theirs[max(0, k - 20):k + 20])})
                elif full.input_ids[:len(theirs)] != theirs:
                    report["mismatches"].append({"record": r["id"], "message": i,
                                                 "error": "training sequence does not start with the prompt"})
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    report["ok"] = report["checked"] > 0 and not report["mismatches"] and not report["errors"]
    (out_dir / "golden_render_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=2))
    for m in report["mismatches"][:5]:
        print(json.dumps(m, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=False)
    p.add_argument("--records", default=str(DEFAULT_RECORDS), help="JSONL file(s), comma-separated")
    p.add_argument("--tp", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    a, _ = p.parse_known_args(argv)
    if not a.model:
        print('From a cell: check("<model dir or HF id>", "<records.jsonl>")')
        return
    check(a.model, a.records, a.tp, limit=a.limit)


if __name__ == "__main__":
    main()
