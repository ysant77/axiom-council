from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from axiom_council.contracts import Budget, RiskClass, TaskSpec, TaskState
from axiom_council.council import Council, open_registry
from axiom_council.providers import OpenAICompatProvider, ProviderError
from axiom_council.registry import NodeSpec, Registry

from .conftest import CONFIG_DIR


def _ok(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "qwen3:4b"}, {"id": "gemma3:4b"}]})
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": f"echo:{body['model']}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    )


def _factory(handler: Callable[[httpx.Request], httpx.Response]) -> Any:
    return lambda node: OpenAICompatProvider(node, transport=httpx.MockTransport(handler))


def test_open_registry_seeds_once(tmp_path: Path) -> None:
    first = open_registry(tmp_path, CONFIG_DIR)
    second = open_registry(tmp_path, tmp_path / "missing-seed-dir")
    assert first.models.keys() == second.models.keys()


def test_low_risk_task_runs_and_is_audited(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    council = Council(registry, tmp_path, _factory(_ok))
    outcome = council.run(make_task(), "Summarise: attention replaces recurrence.")
    council.close()
    assert outcome.state is TaskState.COMPLETED and outcome.result is not None
    assert outcome.result.output.startswith("echo:")
    assert outcome.result.usage.total_tokens == 15
    events = [r["event"] for r in council.audit.for_task(outcome.result.task_id)]
    assert events == ["task_submitted", "routed", "inference_completed"]
    assert council.audit.verify() == 3


def test_policy_rejection_never_reaches_a_model(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    council = Council(registry, tmp_path, _factory(lambda r: pytest.fail("model called")))
    outcome = council.run(make_task(workspace_id="nowhere"), "anything at all")
    rogue = council.run(make_task(), "anything at all", role="ghost")
    council.close()
    assert outcome.state is TaskState.POLICY_REJECTED
    assert rogue.state is TaskState.POLICY_REJECTED and "unknown role" in rogue.reasons[0]


def test_risky_task_waits_for_director(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    council = Council(registry, tmp_path, _factory(lambda r: pytest.fail("model called")))
    task = make_task(workspace_id="pagetrace", risk_class=RiskClass.MEDIUM)
    outcome = council.run(task, "Answer from evidence", output_tokens=256)
    council.close()
    assert outcome.state is TaskState.AWAITING_APPROVAL and outcome.inbox_item
    assert council.inbox.get(outcome.inbox_item).artifact_hash == task.digest()


def test_no_eligible_model_escalates(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    council = Council(registry, tmp_path, _factory(_ok))
    # Any model needs more than 200 tokens for a 1024-token answer, so nothing is eligible.
    outcome = council.run(make_task(budget=Budget(max_tokens=200, max_wall_seconds=60)), "Patch it")
    council.close()
    assert outcome.state is TaskState.AWAITING_CLARIFICATION


def test_provider_failure_is_recorded(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    council = Council(registry, tmp_path, _factory(lambda r: httpx.Response(500)))
    outcome = council.run(make_task(), "Summarise this please")
    council.close()
    assert outcome.state is TaskState.FAILED
    assert council.audit.for_task(outcome.decision.task_id)[-1]["event"] == "inference_failed"  # type: ignore[union-attr]


def test_provider_guards(registry: Registry) -> None:
    node: NodeSpec = registry.nodes["mac-m5"]
    good = OpenAICompatProvider(node, transport=httpx.MockTransport(_ok))
    assert good.list_models() == ["gemma3:4b", "qwen3:4b"]
    with pytest.raises(ProviderError, match="not configured"):
        good.chat(task_id="t", model_id="ghost", messages=[], max_tokens=1)
    good.close()
    junk = OpenAICompatProvider(
        node, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": []}))
    )
    with pytest.raises(ProviderError, match="malformed"):
        junk.chat(task_id="t", model_id="qwen3:4b", messages=[], max_tokens=1)
    down = OpenAICompatProvider(node, transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(ProviderError):
        down.list_models()


def test_budget_exhausted_while_reasoning_is_an_error(registry: Registry) -> None:
    def thinking(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
        )

    provider = OpenAICompatProvider(
        registry.nodes["mac-m5"], transport=httpx.MockTransport(thinking)
    )
    with pytest.raises(ProviderError, match="exhausted its output budget"):
        provider.chat(task_id="t", model_id="qwen3:4b", messages=[], max_tokens=10)


def test_reasoning_overhead_and_defaults_reach_the_request(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    seen: list[dict[str, Any]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _ok(request)

    council = Council(registry, tmp_path, _factory(capture))
    outcome = council.run(make_task(), "Summarise briefly", output_tokens=100)
    council.close()
    assert outcome.result is not None
    model = registry.models[outcome.result.model_id]
    assert seen[0]["max_tokens"] == 100 + model.reasoning_overhead_tokens
    for key, value in model.request_defaults.items():
        assert seen[0][key] == value
