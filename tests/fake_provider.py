"""A `ChatProvider` fake for bake-off tests: canned replies, recorded requests, no network."""

from __future__ import annotations

from typing import Any

from axiom_council.contracts import InferenceResult, InferenceUsage, ModelProvenance


class FakeProvider:
    """Replies by kind (a dict keyed "plan"/"review") or in order (a list).

    `timings=False` behaves like a runtime that reports no load or decode times.
    """

    def __init__(
        self,
        replies: dict[str, str] | list[str],
        *,
        digests: dict[str, str] | None = None,
        quantization: str | None = None,
        timings: bool = True,
        finish_reason: str = "stop",
    ):
        self.replies = replies
        self.digests = digests or {}
        self.quantization = quantization
        self.timings = timings
        self.finish_reason = finish_reason
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.conversations: list[list[dict[str, str]]] = []
        self.unloaded: list[str] = []

    def list_models(self) -> list[str]:
        return []

    def close(self) -> None:
        return None

    def describe(self, model_id: str) -> ModelProvenance:
        return ModelProvenance(
            runtime="ollama",
            runtime_version="0.0-test",
            digest=self.digests.get(model_id),
            quantization=self.quantization,
        )

    def unload(self, model_id: str) -> bool:
        self.unloaded.append(model_id)
        return True

    def _content(self, messages: list[dict[str, str]]) -> str:
        if isinstance(self.replies, list):
            return self.replies[min(len(self.conversations), len(self.replies)) - 1]
        text = messages[-1]["content"]
        return self.replies["review"] if "UNTRUSTED DIFF" in text else self.replies["plan"]

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
        self.calls.append((model_id, dict(extra or {})))
        self.conversations.append(list(messages))
        load = 5.0 if len(self.calls) == 1 else 0.0
        timed = {"load_seconds": load, "prompt_seconds": 0.5, "decode_seconds": 2.0}
        return InferenceResult(
            task_id=task_id,
            model_id=model_id,
            node_id="fake",
            output=self._content(messages),
            usage=InferenceUsage(
                prompt_tokens=100,
                completion_tokens=50,
                wall_seconds=3.0 + load,
                **(timed if self.timings else {}),
            ),
            finish_reason=self.finish_reason,
            reasoning_chars=0 if self.timings else None,
        )
