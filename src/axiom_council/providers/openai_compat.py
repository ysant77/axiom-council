"""Client for OpenAI-compatible chat endpoints: Ollama, llama.cpp server, vLLM and SGLang."""

from __future__ import annotations

import time
from typing import Any

import httpx

from axiom_council.contracts import InferenceResult, InferenceUsage
from axiom_council.registry import NodeSpec


class ProviderError(RuntimeError):
    pass


class OpenAICompatProvider:
    def __init__(
        self,
        node: NodeSpec,
        *,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.node = node
        self._client = httpx.Client(
            base_url=node.base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def list_models(self) -> list[str]:
        try:
            resp = self._client.get("/v1/models")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.node.node_id}: {exc}") from exc
        return sorted(m["id"] for m in resp.json().get("data", []))

    def chat(
        self,
        *,
        task_id: str,
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.2,
        extra: dict[str, Any] | None = None,
    ) -> InferenceResult:
        if model_id not in self.node.serves:
            raise ProviderError(f"node {self.node.node_id} is not configured to serve {model_id}")
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            **(extra or {}),
        }
        started = time.perf_counter()
        try:
            resp = self._client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.node.node_id}/{model_id}: {exc}") from exc
        wall = time.perf_counter() - started

        return self._parse(resp.json(), task_id, model_id, wall)

    def _parse(
        self, body: dict[str, Any], task_id: str, model_id: str, wall: float
    ) -> InferenceResult:
        try:
            choice = body["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"malformed response from {self.node.node_id}") from exc
        if not text.strip() and choice.get("finish_reason") == "length":
            raise ProviderError(
                f"{model_id} exhausted its output budget before answering"
                " (likely hidden reasoning); raise reasoning_overhead_tokens"
            )
        usage = body.get("usage") or {}
        return InferenceResult(
            task_id=task_id,
            model_id=model_id,
            node_id=self.node.node_id,
            output=text,
            usage=InferenceUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                wall_seconds=wall,
            ),
            finish_reason=choice.get("finish_reason"),
        )
