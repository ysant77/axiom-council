"""Ollama native chat client that returns timing breakdowns for bake-offs.

The OpenAI-compatible endpoint hides load and prompt-processing time; the native /api/chat
reports them (in nanoseconds), which is what TTFT and swap-cost decisions need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

NS = 1e9


@dataclass(frozen=True)
class ChatMetrics:
    load_seconds: float
    prompt_tokens: int
    prompt_seconds: float
    output_tokens: int
    decode_seconds: float
    total_seconds: float

    @property
    def ttft_seconds(self) -> float:
        """Engine time to first token: model load plus prompt processing."""
        return self.load_seconds + self.prompt_seconds

    @property
    def decode_tps(self) -> float:
        return self.output_tokens / self.decode_seconds if self.decode_seconds else 0.0


@dataclass(frozen=True)
class ChatReply:
    content: str
    thinking: str
    metrics: ChatMetrics
    done_reason: str | None


class OllamaChat:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        timeout: float = 1800.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        think: bool | str | None,
        num_ctx: int,
        num_predict: int,
        temperature: float,
    ) -> ChatReply:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": "15m",
            "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature},
        }
        if think is not None:
            payload["think"] = think
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        body = resp.json()
        message = body.get("message") or {}
        return ChatReply(
            content=message.get("content") or "",
            thinking=message.get("thinking") or "",
            metrics=ChatMetrics(
                load_seconds=body.get("load_duration", 0) / NS,
                prompt_tokens=int(body.get("prompt_eval_count", 0)),
                prompt_seconds=body.get("prompt_eval_duration", 0) / NS,
                output_tokens=int(body.get("eval_count", 0)),
                decode_seconds=body.get("eval_duration", 0) / NS,
                total_seconds=body.get("total_duration", 0) / NS,
            ),
            done_reason=body.get("done_reason"),
        )

    def unload(self, model: str) -> None:
        """Evict a model so the next call measures a cold load."""
        self._client.post("/api/generate", json={"model": model, "keep_alive": 0})
