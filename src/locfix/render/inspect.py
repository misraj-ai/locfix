"""Human-readable views of traces, records and rendered training examples (for notebooks).

Nothing here is used by training; it shows what the data pipeline produces:

    teacher trace  ->  record (OpenAI messages + loss entries)  ->  rendered text  ->  tokens/labels
"""
from __future__ import annotations

import html
import json
from typing import Any

from .chat import Rendered

SPECIAL_MARKERS = ("<|turn>", "<turn|>", "<|tool_call>", "<tool_call|>", "<|tool_response>",
                   "<tool_response|>", "<|channel>", "<channel|>", "<|\"|>", "<bos>", "<|tool>", "<tool|>", "<|think|>")


def short(text: Any, n: int = 160) -> str:
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    s = s.replace("\n", "⏎ ")
    return s if len(s) <= n else s[: n - 1] + "…"


def content_text(m: dict[str, Any]) -> str:
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") for p in c if isinstance(p, dict))


# --------------------------------------------------------------------------- traces
def trace_rows(entries: list[dict[str, Any]], n: int = 160) -> list[dict[str, Any]]:
    """One row per raw trace entry (the teacher's log, before any processing)."""
    rows = []
    for i, e in enumerate(entries):
        t = e.get("type")
        if t == "tool_call":
            text = f"{e.get('tool')}({short(e.get('args'), n - 20)})"
        elif t == "tool_response":
            text = e.get("result")
        else:
            text = e.get("content")
        rows.append({"#": i, "t (s)": round(e.get("elapsed_s") or 0.0, 1), "type": t,
                     "author": e.get("author"), "content": short(text if text is not None else "", n)})
    return rows


# --------------------------------------------------------------------------- records
def message_rows(record: dict[str, Any], n: int = 160) -> list[dict[str, Any]]:
    """One row per message of a record, with its loss entry (if any)."""
    loss = {e["message_index"]: e for e in record["loss"]}
    rows = []
    for i, m in enumerate(record["messages"]):
        calls = [f"{tc['function']['name']}({short(tc['function']['arguments'], 80)})"
                 for tc in m.get("tool_calls", [])]
        e = loss.get(i)
        rows.append({
            "#": i,
            "role": m["role"],
            "thinking": short(m.get("reasoning") or m.get("reasoning_content") or "", 90),
            "text": short(content_text(m), n),
            "tool calls": " | ".join(calls),
            "trained": "" if e is None else (f"yes (w={e.get('weight', 1.0):g})" if e.get("train", True)
                                            and e.get("weight", 1.0) > 0 else f"no ({e.get('reason', 'w=0')})"),
        })
    return rows


def record_summary(record: dict[str, Any]) -> dict[str, Any]:
    m = record.get("meta") or {}
    out = {"id": record["id"], "stage": record["stage"], "role": record["role"],
           "instance_id": record["instance_id"], "messages": len(record["messages"]),
           "trained turns": sum(1 for e in record["loss"] if e.get("train", True) and e.get("weight", 1) > 0),
           "tools": [t["function"]["name"] for t in record.get("tools", [])],
           "record weight": record.get("weight", 1.0)}
    for k in ("variant", "budget", "counted_calls", "n_counted_calls", "localizer_calls", "window", "n_windows"):
        if k in m:
            out[k] = m[k]
    if record.get("z"):
        out["z"] = record["z"]
    return out


# --------------------------------------------------------------------------- rendered
def _segments(r: Rendered) -> list[tuple[str, float | None]]:
    """Split the rendered text into (text, weight) pieces; weight None = context (no loss)."""
    out: list[tuple[str, float | None]] = []
    pos = 0
    for s, e, w in sorted(r.target_spans):
        if s > pos:
            out.append((r.text[pos:s], None))
        out.append((r.text[s:e], w))
        pos = e
    if pos < len(r.text):
        out.append((r.text[pos:], None))
    return out


def _mark_special(escaped: str) -> str:
    for tok in SPECIAL_MARKERS:
        t = html.escape(tok)
        escaped = escaped.replace(t, f"<b style='color:#7c3aed'>{t}</b>")
    return escaped


def rendered_html(r: Rendered, max_chars_per_segment: int | None = 3000, height: str = "600px") -> str:
    """The exact training text: grey = context (no loss), green = trained. Special tokens in purple.
    Long context segments (tool outputs, system prompt) are shortened for display only."""
    parts = []
    for text, w in _segments(r):
        shown = text
        if w is None and max_chars_per_segment and len(text) > max_chars_per_segment:
            k = max_chars_per_segment // 2
            shown = text[:k] + f"\n[... {len(text) - 2 * k} characters not shown ...]\n" + text[-k:]
        esc = _mark_special(html.escape(shown))
        if w is None:
            parts.append(f"<span style='color:#6b7280'>{esc}</span>")
        else:
            bg = "#dcfce7" if w >= 1 else "#fef9c3"
            parts.append(f"<span style='background:{bg};color:#111' title='weight {w:g}'>{esc}</span>")
    legend = ("<div style='font:12px sans-serif;margin-bottom:6px'>"
              "<span style='color:#6b7280'>grey = context, no loss</span> &nbsp; "
              "<span style='background:#dcfce7'>green = trained (weight 1)</span> &nbsp; "
              "<span style='background:#fef9c3'>yellow = trained, weight &lt; 1</span> &nbsp; "
              "<b style='color:#7c3aed'>purple = special tokens</b></div>")
    body = "".join(parts)
    return (legend + f"<pre style='white-space:pre-wrap;font:12px monospace;max-height:{height};"
            f"overflow:auto;border:1px solid #ddd;padding:8px'>{body}</pre>")


def rendered_ascii(r: Rendered, max_chars_per_segment: int = 400) -> str:
    """Terminal version: trained parts between ⟦ and ⟧."""
    out = []
    for text, w in _segments(r):
        if w is None and len(text) > max_chars_per_segment:
            k = max_chars_per_segment // 2
            text = text[:k] + f" [...{len(text) - 2 * k} chars...] " + text[-k:]
        out.append(text if w is None else f"⟦{text}⟧")
    return "".join(out)


def token_rows(r: Rendered, tok, span: int = 0, at: str = "start", before: int = 6,
               after: int = 10) -> list[dict[str, Any]]:
    """Tokens around the start (or end) of trained span `span`: id, piece, label, weight."""
    s, e, _w = r.target_spans[span]
    enc = tok(r.text, add_special_tokens=False, return_offsets_mapping=True)
    if at == "start":
        first = next(i for i, (a, _b) in enumerate(enc["offset_mapping"]) if a >= s)
    else:
        first = next(i for i, (_a, b) in enumerate(enc["offset_mapping"]) if b >= e) + 1
    rows = []
    for i in range(max(0, first - before), min(len(r.input_ids), first + after)):
        rows.append({"position": i, "token id": r.input_ids[i], "piece": repr(tok.decode([r.input_ids[i]])),
                     "label": r.labels[i], "weight": f"{r.weights[i]:g}",
                     "in loss": "✓" if r.labels[i] != -100 else ""})
    return rows


def rendered_stats(r: Rendered) -> dict[str, Any]:
    n = len(r.input_ids)
    t = r.n_trained
    return {"tokens": n, "trained tokens": t, "trained %": round(100 * t / max(n, 1), 1),
            "trained turns": len(r.target_spans)}


def trainer_batch(r: Rendered) -> dict[str, list]:
    """What the trainer gets for one example (before padding/packing): the loss is
    sum_t weight_t * CE(logits_{t-1}, input_ids_t) / sum_t weight_t over positions with a label."""
    return {"input_ids": r.input_ids, "labels": r.labels, "loss_weights": r.weights}
