"""The native Ollama provider: reasoning control, timings, model build and eviction."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from axiom_council.contracts import TaskSpec
from axiom_council.council import Council
from axiom_council.providers import (
    OllamaProvider,
    OpenAICompatProvider,
    ProviderError,
    provider_for,
)
from axiom_council.registry import Registry

DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"


def _ollama(seen: list[tuple[str, Any]], content: str = "hi", reason: str = "stop") -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.url.path, body))
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.11"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b", "digest": DIGEST}]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"details": {"quantization_level": "Q4_K_M"}})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={
            "message": {"content": content, "thinking": "hmm"}, "done_reason": reason,
            "load_duration": 2e9, "prompt_eval_count": 10, "prompt_eval_duration": 5e8,
            "eval_count": 40, "eval_duration": 2e9, "total_duration": 4.5e9,
        })  # fmt: skip

    return httpx.MockTransport(handler)


def test_ollama_runtime_uses_the_native_provider(registry: Registry) -> None:
    node = registry.nodes["mac-m5"]
    provider = provider_for(node, timeout=1)
    assert isinstance(provider, OllamaProvider)
    provider.close()
    llama = provider_for(node.model_copy(update={"runtime": "llama.cpp"}), timeout=1)
    assert isinstance(llama, OpenAICompatProvider)
    llama.close()


def test_chat_maps_reasoning_and_options_and_reports_timings(registry: Registry) -> None:
    seen: list[tuple[str, Any]] = []
    provider = OllamaProvider(registry.nodes["mac-m5"], transport=_ollama(seen))
    extra = {"reasoning_effort": "low", "num_ctx": 2048, "seed": 7}
    result = provider.chat(
        task_id="t", model_id="qwen3:4b", messages=[], max_tokens=64, temperature=0.1, extra=extra
    )
    _, body = seen[-1]
    assert body["think"] == "low" and body["options"] == {
        "num_predict": 64, "temperature": 0.1, "num_ctx": 2048, "seed": 7,
    }  # fmt: skip
    usage = result.usage
    assert (usage.prompt_tokens, usage.completion_tokens) == (10, 40)
    assert (usage.load_seconds, usage.prompt_seconds, usage.decode_seconds) == (2.0, 0.5, 2.0)
    assert result.reasoning_chars == 3 and result.finish_reason == "stop"
    provider.chat(task_id="t", model_id="qwen3:4b", messages=[], max_tokens=8,
                  extra={"reasoning_effort": "low", "think": False})  # fmt: skip
    assert seen[-1][1]["think"] is False  # an explicit think wins over reasoning_effort
    provider.chat(task_id="t", model_id="qwen3:4b", messages=[], max_tokens=8)
    assert "think" not in seen[-1][1]
    provider.close()


def test_describe_reports_build_quantisation_and_version(registry: Registry) -> None:
    provider = OllamaProvider(registry.nodes["mac-m5"], transport=_ollama([]))
    prov = provider.describe("qwen3:4b")
    assert (prov.runtime, prov.runtime_version) == ("ollama", "0.40.11")
    assert prov.digest == DIGEST and prov.quantization == "Q4_K_M"
    assert provider.describe("gemma3:4b").digest is None  # not pulled
    assert provider.list_models() == ["qwen3:4b"]
    seen: list[tuple[str, Any]] = []
    provider = OllamaProvider(registry.nodes["mac-m5"], transport=_ollama(seen))
    assert provider.unload("qwen3:4b") is True
    assert seen[-1] == ("/api/generate", {"model": "qwen3:4b", "keep_alive": 0})


def test_unserved_model_and_http_errors_fail_closed(registry: Registry) -> None:
    provider = OllamaProvider(registry.nodes["mac-m5"], transport=_ollama([]))
    with pytest.raises(ProviderError, match="not configured"):
        provider.chat(task_id="t", model_id="ghost:1", messages=[], max_tokens=8)
    down = OllamaProvider(
        registry.nodes["mac-m5"], transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    with pytest.raises(ProviderError):
        down.chat(task_id="t", model_id="qwen3:4b", messages=[], max_tokens=8)
    with pytest.raises(ProviderError):
        down.describe("qwen3:4b")


def test_openai_compat_describes_what_it_can(registry: Registry) -> None:
    node = registry.nodes["mac-m5"].model_copy(update={"runtime": "vllm"})
    provider = OpenAICompatProvider(
        node, transport=httpx.MockTransport(lambda r: httpx.Response(200))
    )
    prov = provider.describe("qwen3:4b")
    assert prov.runtime == "vllm" and prov.digest is None and prov.runtime_version is None
    assert provider.unload("qwen3:4b") is False


def test_openai_compat_drops_the_native_think_field(registry: Registry) -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    node = registry.nodes["mac-m5"].model_copy(update={"runtime": "vllm"})
    provider = OpenAICompatProvider(node, transport=httpx.MockTransport(handler))
    provider.chat(task_id="t", model_id="qwen3:4b", messages=[], max_tokens=8,
                  extra={"think": False, "reasoning_effort": "low"})  # fmt: skip
    assert "think" not in sent[0] and sent[0]["reasoning_effort"] == "low"


def test_council_sends_the_card_reasoning_and_fails_on_an_empty_answer(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    seen: list[tuple[str, Any]] = []
    council = Council(
        registry, tmp_path,
        lambda node: OllamaProvider(node, transport=_ollama(seen, content="", reason="length")),
    )  # fmt: skip
    outcome = council.run(make_task(), "Summarise: attention replaces recurrence.")
    council.close()
    assert outcome.state.value == "failed" and "output budget" in outcome.reasons[0]
    chat = next(body for path, body in seen if path == "/api/chat")
    model = registry.models[chat["model"]]
    expected = (
        model.think if model.think is not None else model.request_defaults.get("reasoning_effort")
    )
    assert chat.get("think") == expected
