"""Provider factory: a node's `runtime` field picks the client class that talks to it.

Adding a node or a model is a config change. Adding a runtime with a different wire protocol
is one new class registered here; callers never name a provider class directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from axiom_council.contracts import InferenceResult, ModelProvenance
from axiom_council.providers.ollama import OllamaProvider
from axiom_council.providers.openai_compat import OpenAICompatProvider, ProviderError
from axiom_council.registry import NodeSpec


class ChatProvider(Protocol):
    def list_models(self) -> list[str]: ...

    def chat(
        self,
        *,
        task_id: str,
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.2,
        extra: dict[str, Any] | None = None,
    ) -> InferenceResult: ...

    def describe(self, model_id: str) -> ModelProvenance: ...

    def unload(self, model_id: str) -> bool: ...

    def close(self) -> None: ...


ProviderClass = Callable[..., ChatProvider]
_RUNTIMES: dict[str, ProviderClass] = {}


def register_runtime(*names: str) -> Callable[[ProviderClass], ProviderClass]:
    """Class decorator (or plain call) that maps runtime names to a provider class."""

    def register(cls: ProviderClass) -> ProviderClass:
        for name in names:
            if name in _RUNTIMES and _RUNTIMES[name] is not cls:
                raise ValueError(f"runtime {name!r} is already registered")
            _RUNTIMES[name] = cls
        return cls

    return register


def runtimes() -> list[str]:
    return sorted(_RUNTIMES)


def provider_for(node: NodeSpec, **kwargs: Any) -> ChatProvider:
    """Build the client for a node from its configured runtime; unknown runtimes fail closed."""
    cls = _RUNTIMES.get(node.runtime)
    if cls is None:
        raise ProviderError(
            f"node {node.node_id} uses unknown runtime {node.runtime!r}; known: {runtimes()}"
        )
    return cls(node, **kwargs)


# Ollama uses its native API (reasoning control, timings, model digest); the other
# runtimes speak the OpenAI-compatible chat API.
register_runtime("ollama")(OllamaProvider)
register_runtime("llama.cpp", "vllm", "sglang")(OpenAICompatProvider)
