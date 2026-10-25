"""A minimal OpenAI-compatible chat-completions server for scripted replay and request capture.

Every POST to /v1/chat/completions is passed to `handler(request_body) -> response_body`
(a dict in OpenAI chat-completion format) and recorded with the request, so the exact JSON that
LiteLLM would send to vLLM is captured. Streaming requests get the same response as one SSE chunk
sequence. GET /v1/models and /health are answered for clients that probe the server.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

Handler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Exchange:
    request: dict[str, Any]
    response: dict[str, Any] | None
    error: str | None = None
    t: float = field(default_factory=time.time)


def chat_response(model: str, content: str | None, reasoning: str | None,
                  tool_calls: list[dict[str, Any]] | None, finish_reason: str | None = None,
                  usage: dict[str, int] | None = None) -> dict[str, Any]:
    """Build a chat.completion body the way vLLM returns it (reasoning in `reasoning`)."""
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning"] = reasoning
        msg["reasoning_content"] = reasoning
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:16],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": msg, "logprobs": None,
                     "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop")}],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _stream_chunks(resp: dict[str, Any]) -> list[dict[str, Any]]:
    choice = resp["choices"][0]
    msg = choice["message"]
    base = {"id": resp["id"], "object": "chat.completion.chunk", "created": resp["created"],
            "model": resp["model"]}
    delta: dict[str, Any] = {"role": "assistant"}
    if msg.get("reasoning"):
        delta["reasoning"] = msg["reasoning"]
        delta["reasoning_content"] = msg["reasoning"]
    if msg.get("content"):
        delta["content"] = msg["content"]
    if msg.get("tool_calls"):
        delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(msg["tool_calls"])]
    return [
        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
         "usage": resp.get("usage")},
    ]


class ScriptedServer:
    def __init__(self, handler: Handler, host: str = "127.0.0.1", port: int = 0):
        self.handler = handler
        self.exchanges: list[Exchange] = []
        self._lock = threading.Lock()
        server = self

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, json.dumps({"object": "list", "data": [
                        {"id": "scripted", "object": "model"}]}).encode())
                else:
                    self._send(200, b"{}")

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._send(404, b'{"error": "only /v1/chat/completions"}')
                    return
                try:
                    resp = server.handler(req)
                    with server._lock:
                        server.exchanges.append(Exchange(req, resp))
                except Exception as e:  # noqa: BLE001 - report to the client and keep the record
                    with server._lock:
                        server.exchanges.append(Exchange(req, None, repr(e)))
                    self._send(500, json.dumps({"error": {"message": repr(e)}}).encode())
                    return
                if req.get("stream"):
                    body = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in _stream_chunks(resp))
                    self._send(200, body + b"data: [DONE]\n\n", "text/event-stream")
                else:
                    self._send(200, json.dumps(resp).encode())

        self._httpd = ThreadingHTTPServer((host, port), _H)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "ScriptedServer":
        for var in ("NO_PROXY", "no_proxy"):
            cur = os.environ.get(var, "")
            if "127.0.0.1" not in cur:
                os.environ[var] = ",".join(x for x in (cur, "127.0.0.1", "localhost") if x)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
