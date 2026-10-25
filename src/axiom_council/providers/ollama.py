"""Client for Ollama's native API: reasoning control, load and decode timings, model builds.

The OpenAI-compatible endpoint hides load and prompt-processing time and the model digest;
the native API reports both, which run provenance (D-12) and swap-cost decisions need.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from axiom_council.contracts import InferenceResult, InferenceUsage, ModelProvenance
from axiom_council.providers.openai_compat import ProviderError
from axiom_council.registry import NodeSpec

NS = 1e9


def _seconds(body: dict[str, Any], key: str) -> float | None:
    value = body.get(key)
    return None if value is None else value / NS


class OllamaProvider:
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

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._client.request(method, path, **kwargs)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.node.node_id}{path}: {exc}") from exc
        return resp.json()

    def list_models(self) -> list[str]:
        return sorted(m["name"] for m in self._request("GET", "/api/tags").get("models", []))

    def describe(self, model_id: str) -> ModelProvenance:
        tags = self._request("GET", "/api/tags").get("models", [])
        digest = next((m.get("digest") for m in tags if m.get("name") == model_id), None)
        details: dict[str, Any] = {}
        if digest is not None:
            details = self._request("POST", "/api/show", json={"model": model_id}).get(
                "details", {}
            )
        return ModelProvenance(
            runtime=self.node.runtime,
            runtime_version=self._request("GET", "/api/version").get("version"),
            digest=digest,
            quantization=details.get("quantization_level"),
        )

    def unload(self, model_id: str) -> bool:
        """Evict a model so the next call measures a cold load."""
        self._request("POST", "/api/generate", json={"model": model_id, "keep_alive": 0})
        return True

    @staticmethod
    def _payload(
        model_id: str, messages: list[dict[str, str]], max_tokens: int, temperature: float,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:  # fmt: skip
        """`think` wins over `reasoning_effort`; every other extra field is an engine option."""
        options = dict(extra or {})
        think = options.pop("think", None)
        effort = options.pop("reasoning_effort", None)
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "keep_alive": "15m",
            "options": {"num_predict": max_tokens, "temperature": temperature, **options},
        }
        if think is None:
            think = effort
        if think is not None:
            payload["think"] = think
        return payload

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
        payload = self._payload(model_id, messages, max_tokens, temperature, extra)
        started = time.perf_counter()
        body = self._request("POST", "/api/chat", json=payload)
        wall = time.perf_counter() - started
        message = body.get("message") or {}
        return InferenceResult(
            task_id=task_id,
            model_id=model_id,
            node_id=self.node.node_id,
            output=message.get("content") or "",
            usage=InferenceUsage(
                prompt_tokens=int(body.get("prompt_eval_count", 0)),
                completion_tokens=int(body.get("eval_count", 0)),
                wall_seconds=wall,
                load_seconds=_seconds(body, "load_duration"),
                prompt_seconds=_seconds(body, "prompt_eval_duration"),
                decode_seconds=_seconds(body, "eval_duration"),
            ),
            finish_reason=body.get("done_reason"),
            reasoning_chars=len(message.get("thinking") or ""),
        )
