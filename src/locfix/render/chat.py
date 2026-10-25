"""Records -> token ids, labels and per-token loss weights, rendered exactly as vLLM renders them.

vLLM 0.19.1 (`vllm/entrypoints/chat_utils.py`, `vllm/renderers/hf.py`) builds the prompt by:
  1. converting each OpenAI message into a "conversation" message. The Gemma 4 template loops over
     `message['content']` items, so vLLM's auto-detected content format is "openai": string content
     becomes `[{"type": "text", "text": ...}]`, `None` becomes `[]`;
  2. keeping `tool_calls`, `tool_call_id`, `name`, and the assistant's `reasoning` (copied to both
     `reasoning` and `reasoning_content`; vLLM reads only `reasoning`);
  3. parsing tool-call `arguments` JSON strings into dicts (`_postprocess_messages`);
  4. applying the model's chat template with `tools`, `add_generation_prompt=True` and the request's
     `chat_template_kwargs` (e.g. `enable_thinking`).

Training targets are what the model *generated* at each trained turn: the text between the prompt
of that turn (`render(messages[:i], add_generation_prompt=True)`) and the end of the model's own
output: through the `<|tool_response>` token that ends a tool-call turn, or through `<turn|>` for a
text turn. For every trained turn except the last, the full rendering must contain exactly that
text at that position (prefix consistency); otherwise `RenderError` is raised and the record must be
split. The last message is rendered in generation format (thinking kept even for text-only turns).
Golden tests against vLLM's `/v1/chat/completions/render` run on a GPU machine (spec 02).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

TOOL_RESPONSE_START = "<|tool_response>"
TURN_END = "<turn|>"
THOUGHT_START = "<|channel>thought\n"
THOUGHT_END = "<channel|>"


class RenderError(ValueError):
    pass


def load_tokenizer(path: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


# --------------------------------------------------------------------------- vLLM conversion
def to_vllm_conversation(messages: list[dict[str, Any]], content_format: str = "openai") -> list[dict[str, Any]]:
    """OpenAI messages -> the conversation vLLM 0.19.1 passes to the chat template."""
    conv: list[dict[str, Any]] = []
    for m in messages:
        m = copy.deepcopy(m)
        role = m["role"]
        content = m.get("content")
        if content is None:
            parts: list[Any] = []
        elif isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            parts = [p if isinstance(p, dict) else {"type": "text", "text": p} for p in content]
        texts = [p.get("text", "") for p in parts if p.get("type") in
                 ("text", "input_text", "output_text", "refusal", "thinking")]
        if content_format == "openai":
            out: dict[str, Any] = {"role": role, "content": [{"type": "text", "text": t} for t in texts]}
        else:
            out = {"role": role, "content": "\n".join(texts)}
        if role == "assistant":
            if m.get("tool_calls") is not None:
                out["tool_calls"] = list(m["tool_calls"])
            reasoning = m.get("reasoning")
            if reasoning is not None:
                out["reasoning"] = reasoning
                out["reasoning_content"] = reasoning
        if role == "tool" and "tool_call_id" in m:
            out["tool_call_id"] = m["tool_call_id"]
        if isinstance(m.get("name"), str):
            out["name"] = m["name"]
        conv.append(out)
    # _postprocess_messages
    for msg in conv:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            tcs = msg.get("tool_calls")
            if not isinstance(tcs, list):
                continue
            if not tcs:
                msg.pop("tool_calls")
                continue
            for item in tcs:
                args = item["function"].get("arguments")
                if args:
                    if not isinstance(args, (dict, list)):
                        item["function"]["arguments"] = json.loads(args)
                else:
                    item["function"]["arguments"] = {}
    return conv


def render_text(tok, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                template_kwargs: dict[str, Any] | None, add_generation_prompt: bool) -> str:
    conv = to_vllm_conversation(messages)
    return tok.apply_chat_template(conv, tools=tools or None, tokenize=False,
                                   add_generation_prompt=add_generation_prompt,
                                   **(template_kwargs or {}))


def prompt_token_counter(tok, template_kwargs: dict[str, Any] | None = None) -> Callable[[dict], int]:
    """Prompt length in tokens of an OpenAI request, as vLLM would count it (for compaction)."""
    def count(req: dict[str, Any]) -> int:
        kw = req.get("chat_template_kwargs", template_kwargs) or {}
        text = render_text(tok, req.get("messages", []), req.get("tools"), kw, True)
        return len(tok(text, add_special_tokens=False)["input_ids"])
    return count


# --------------------------------------------------------------------------- targets
def generation_text(message: dict[str, Any], rendered_history_tail: str | None) -> str:
    """The text the model generates for `message` (an assistant message).

    For tool-call messages the history rendering already has the generation form, so the caller
    passes it (`rendered_history_tail`). Text-only messages are built here because the template
    drops their thinking in history: `<|channel>thought\\n{r}\\n<channel|>{content}<turn|>`.
    """
    if rendered_history_tail is not None:
        return rendered_history_tail
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = "".join(p.get("text", "") for p in content)
    out = ""
    if reasoning:
        out += THOUGHT_START + reasoning + "\n" + THOUGHT_END
    return out + content.strip() + TURN_END


@dataclass
class Rendered:
    text: str
    input_ids: list[int]
    labels: list[int]                     # -100 where not trained
    weights: list[float]                  # 0 where not trained
    target_spans: list[tuple[int, int, float]] = field(default_factory=list)   # char spans + weight

    @property
    def n_trained(self) -> int:
        return sum(1 for w in self.weights if w > 0)


def _target_for(tok, msgs, i, tools, kw, is_last: bool) -> tuple[str, str]:
    """(prompt_text, target_text) for assistant message i."""
    prompt = render_text(tok, msgs[:i], tools, kw, add_generation_prompt=True)
    m = msgs[i]
    if m.get("tool_calls"):
        upto = render_text(tok, msgs[: i + 1], tools, kw, add_generation_prompt=False)
        if not upto.startswith(prompt):
            raise RenderError(f"message {i}: rendering with the message does not extend its prompt")
        tail = upto[len(prompt):]
        if not tail.endswith(TOOL_RESPONSE_START):
            raise RenderError(f"message {i}: tool-call turn does not end with {TOOL_RESPONSE_START}")
        return prompt, generation_text(m, tail)
    if not is_last:
        # text-only turns lose their thinking in history; they must end a record (split at user msgs)
        reasoning = m.get("reasoning") or m.get("reasoning_content")
        if reasoning:
            raise RenderError(f"message {i}: text-only turn with thinking is not the last message")
        upto = render_text(tok, msgs[: i + 1], tools, kw, add_generation_prompt=False)
        tail = upto[len(prompt):]
        return prompt, tail[: tail.find(TURN_END) + len(TURN_END)]
    return prompt, generation_text(m, None)


def render_record(record: dict[str, Any], tok) -> Rendered:
    msgs = record["messages"]
    tools = record.get("tools")
    kw = (record.get("meta") or {}).get("request_params", {}).get("chat_template_kwargs") or {}
    rec_w = float(record.get("weight", 1.0))
    loss = {e["message_index"]: e for e in record["loss"]}
    last = len(msgs) - 1
    if msgs[last]["role"] != "assistant":
        raise RenderError("record must end with an assistant message")

    # full text: history rendering of everything before the last message + its generation text
    prompt_last, target_last = _target_for(tok, msgs, last, tools, kw, is_last=True)
    full = prompt_last + target_last

    spans: list[tuple[int, int, float]] = []
    for i, e in sorted(loss.items()):
        if msgs[i]["role"] != "assistant":
            raise RenderError(f"loss on non-assistant message {i}")
        w = float(e.get("weight", 1.0)) * rec_w if e.get("train", True) else 0.0
        if i == last:
            prompt, target = prompt_last, target_last
        else:
            prompt, target = _target_for(tok, msgs, i, tools, kw, is_last=False)
        start = len(prompt)
        if full[start:start + len(target)] != target or not full.startswith(prompt):
            raise RenderError(f"message {i}: not prefix-consistent with the full rendering")
        if w > 0:
            spans.append((start, start + len(target), w))

    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    labels = [-100] * len(ids)
    weights = [0.0] * len(ids)
    starts = {s for s, _ in offsets}
    ends = {e for _, e in offsets}
    for s, e, w in spans:
        if s not in starts or e not in ends:
            raise RenderError(f"target span {s}-{e} does not fall on token boundaries")
        for t, (a, b) in enumerate(offsets):
            if a >= s and b <= e and b > a:
                labels[t] = ids[t]
                weights[t] = w
    return Rendered(full, ids, labels, weights, spans)
