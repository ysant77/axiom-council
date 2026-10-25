from __future__ import annotations

from pathlib import Path

import pytest

from axiom_council.configstore import ConfigStore
from axiom_council.contracts import TaskClass
from axiom_council.registry import Registry


def test_seed_loads_every_kind(store: ConfigStore, registry: Registry) -> None:
    assert {"gpt-oss:20b", "qwen3:4b", "gemma3:4b"} <= registry.models.keys()
    assert registry.nodes["win-4060"].enabled is False
    assert "worker.grounded_answer" in registry.prompts
    assert not store.is_empty()


def test_put_versions_only_on_change(store: ConfigStore) -> None:
    body = dict(store.get("prompt", "worker.default").body)  # type: ignore[union-attr]
    same = store.put("prompt", "worker.default", body)
    assert same.version == 1
    changed = store.put("prompt", "worker.default", {**body, "system": "Be terse."})
    assert changed.version == 2
    history = store.history("prompt", "worker.default")
    assert [e.version for e in history] == [1, 2]
    assert store.active("prompt")["worker.default"]["system"] == "Be terse."


def test_reseeding_is_idempotent(store: ConfigStore) -> None:
    store.seed(Path(__file__).resolve().parents[1] / "config")
    assert store.get("model", "qwen3:4b").version == 1  # type: ignore[union-attr]


def test_unknown_kind_is_rejected(store: ConfigStore) -> None:
    with pytest.raises(ValueError, match="unknown config kind"):
        store.put("secret", "x", {})


def test_broken_links_are_rejected(store: ConfigStore) -> None:
    node = dict(store.active("node")["mac-m5"])
    store.put("node", "mac-m5", {**node, "serves": ["ghost:1b"]})
    with pytest.raises(ValueError, match="unknown models"):
        Registry.from_store(store)


def test_prompt_with_unknown_role_is_rejected(store: ConfigStore) -> None:
    store.put("prompt", "bad", {"prompt_id": "bad", "role": "ghost", "system": "x"})
    with pytest.raises(ValueError, match="unknown role"):
        Registry.from_store(store)


def test_prompt_selection_and_rendering(registry: Registry) -> None:
    grounded = registry.prompt_for("worker", TaskClass.GROUNDED_ANSWER)
    assert grounded is not None and grounded.prompt_id == "worker.grounded_answer"
    fallback = registry.prompt_for("worker", TaskClass.TRIAGE)
    assert fallback is not None and fallback.prompt_id == "worker.default"
    assert registry.prompt_for("builder", TaskClass.TRIAGE) is None
    messages = grounded.render(input="What is attention?")
    assert messages[1]["content"] == "Question: What is attention?"
    with pytest.raises(KeyError):
        grounded.render()
