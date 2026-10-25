"""Config-driven construction: providers by runtime, tunables from the settings store."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from axiom_council.configstore import ConfigStore
from axiom_council.contracts import TaskSpec
from axiom_council.energy import EstimateMeter
from axiom_council.evals import session
from axiom_council.evals.bakeoff import Bakeoff, Contestant
from axiom_council.evals.fixtures import load_fixtures
from axiom_council.ledger import CapabilityLedger
from axiom_council.providers import (
    OpenAICompatProvider,
    ProviderError,
    provider_for,
    register_runtime,
    runtimes,
)
from axiom_council.registry import Registry
from axiom_council.router import Router
from axiom_council.settings import Settings

from .test_bakeoff import GOOD_PLAN, REPO, FakeChat


def test_provider_is_chosen_by_node_runtime(registry: Registry) -> None:
    node = registry.nodes["mac-m5"]
    provider = provider_for(node, timeout=1)
    assert isinstance(provider, OpenAICompatProvider)
    provider.close()
    assert {"ollama", "llama.cpp", "vllm", "sglang"} <= set(runtimes())

    odd = node.model_copy(update={"runtime": "tensorrt"})
    with pytest.raises(ProviderError, match="unknown runtime 'tensorrt'"):
        provider_for(odd)


def test_new_runtime_is_one_registration(registry: Registry) -> None:
    class Probe(OpenAICompatProvider):
        pass

    register_runtime("probe-runtime")(Probe)
    node = registry.nodes["mac-m5"].model_copy(update={"runtime": "probe-runtime"})
    provider = provider_for(node)
    assert isinstance(provider, Probe)
    provider.close()
    with pytest.raises(ValueError, match="already registered"):
        register_runtime("ollama")(Probe)


def test_settings_come_from_the_store_and_version(
    store: ConfigStore, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    assert Registry.from_store(store).settings.router.min_success == 0.55
    router = dict(store.active("setting")["router"])
    entry = store.put("setting", "router", {**router, "min_success": 0.99})
    assert entry.version == 2
    registry = Registry.from_store(store)
    assert registry.settings.router.min_success == 0.99

    ledger = CapabilityLedger(tmp_path / "ledger.sqlite3")
    try:
        decision = Router(registry, ledger).route(make_task(), prompt_tokens=100)
    finally:
        ledger.close()
    assert decision.chosen is None  # a stricter gate now rejects every model, no code change


def test_unknown_or_invalid_settings_fail_loudly() -> None:
    with pytest.raises(ValidationError):
        Settings.from_entries({"routr": {"name": "routr"}})
    with pytest.raises(ValidationError):
        Settings.from_entries({"router": {"name": "router", "min_success": 2}})
    assert Settings.from_entries({}).bakeoff.kinds["code"].num_predict == 8000


def test_bakeoff_kind_must_name_a_known_prompt(registry: Registry) -> None:
    kinds = dict(registry.settings.bakeoff.kinds)
    kinds["plan"] = kinds["plan"].model_copy(update={"prompt_id": "ghost.prompt"})
    bad = registry.settings.model_copy(
        update={"bakeoff": registry.settings.bakeoff.model_copy(update={"kinds": kinds})}
    )
    with pytest.raises(ValueError, match=r"ghost\.prompt"):
        registry.model_copy(update={"settings": bad}).validate_links()


def test_bakeoff_uses_model_card_think_unless_overridden(
    registry: Registry, tmp_path: Path
) -> None:
    chat = FakeChat({"plan": json.dumps(GOOD_PLAN), "review": "{}"})
    ledger = CapabilityLedger(tmp_path / "ledger.sqlite3")
    bake = Bakeoff(
        registry,
        load_fixtures(REPO / "evals/fixtures"),
        chat,  # type: ignore[arg-type]
        ledger,
        tmp_path / "r.jsonl",
        "n1",
        EstimateMeter(30),
    )
    bake.run([Contestant("gemma4:26b"), Contestant("gpt-oss:20b", "high")])
    ledger.close()
    thinks = {model: think for model, think in chat.calls}
    assert thinks == {"gemma4:26b": False, "gpt-oss:20b": "high"}


def test_bakeoff_node_comes_from_config(registry: Registry) -> None:
    assert session._node_for(registry, ["gpt-oss:20b"]).node_id == "mac-m5"
    with pytest.raises(ValueError, match="no enabled ollama node"):
        session._node_for(registry, ["gpt-oss:20b", "ghost:1"])
