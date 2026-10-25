from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from axiom_council.configstore import ConfigStore
from axiom_council.contracts import Budget, Capability, TaskClass, TaskSpec
from axiom_council.registry import Registry

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ConfigStore]:
    s = ConfigStore(tmp_path / "config.sqlite3")
    s.seed(CONFIG_DIR)
    yield s
    s.close()


@pytest.fixture
def registry(store: ConfigStore) -> Registry:
    return Registry.from_store(store)


@pytest.fixture
def make_task() -> Callable[..., TaskSpec]:
    def factory(**overrides: Any) -> TaskSpec:
        fields: dict[str, Any] = {
            "workspace_id": "sandbox",
            "objective": "Summarise the abstract of a paper",
            "task_class": TaskClass.TRIAGE,
            "actor": "director",
            "allowed_capabilities": frozenset({Capability.LLM_INFERENCE}),
            "budget": Budget(max_tokens=16_000, max_wall_seconds=600),
            "acceptance_checks": ("director_review",),
        }
        fields.update(overrides)
        return TaskSpec(**fields)

    return factory
