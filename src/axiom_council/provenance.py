"""Run provenance and placement identity (defects D-12 and D-04).

A placement is what routing evidence is about: one model build, at one quantisation, on one
runtime and node, with one reasoning setting. The harness is everything else that shapes the
answer (prompts, settings, roles, routes, workspaces, sandboxes); see RFC 0005.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from axiom_council.contracts import content_hash
from axiom_council.registry import ModelCard, Registry

_SOURCE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Placement:
    model_id: str
    node_id: str
    runtime: str | None = None
    model_digest: str | None = None
    quantization: str | None = None
    reasoning: str | None = None


def think_value(card: ModelCard, override: bool | str | None = None) -> bool | str | None:
    """The reasoning control sent to the engine: override, then card `think`, then the
    card's `reasoning_effort`; None leaves the engine default."""
    if override is not None:
        return override
    if card.think is not None:
        return card.think
    effort = card.request_defaults.get("reasoning_effort")
    return None if effort is None else str(effort)


def reasoning_setting(card: ModelCard, override: bool | str | None = None) -> str | None:
    value = think_value(card, override)
    if value is None:
        return None
    return str(value).lower()


def placement_for(
    registry: Registry, model_id: str, node_id: str, *, think: bool | str | None = None
) -> Placement:
    card, node = registry.models[model_id], registry.nodes[node_id]
    return Placement(
        model_id=model_id,
        node_id=node_id,
        runtime=node.runtime,
        model_digest=card.digest,
        quantization=card.quantization,
        reasoning=reasoning_setting(card, think),
    )


def _canonical(value: Any) -> Any:
    """Sets become sorted lists, so a hash never depends on PYTHONHASHSEED."""
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, set | frozenset):
        return sorted((_canonical(v) for v in value), key=repr)
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    return value


def _hash(model: BaseModel, exclude: set[str] | None = None) -> str:
    return content_hash(_canonical(model.model_dump(exclude=exclude)))


def harness_hash(registry: Registry) -> str:
    """Content hash of everything except the placement (models and nodes)."""
    return _hash(registry, {"models", "nodes"})


def config_hash(registry: Registry) -> str:
    return _hash(registry)


def fixture_hash(fixture: BaseModel) -> str:
    return _hash(fixture)


def _git(*args: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(  # noqa: S603 - argv list, no shell
        [git, "-C", str(_SOURCE_ROOT), *args],
        capture_output=True, text=True, check=False, timeout=30,
    )  # fmt: skip
    return result.stdout.strip() if result.returncode == 0 else None


@cache
def source_commit() -> str | None:
    """The council's own commit, with `-dirty` when tracked files differ; None outside git."""
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return f"{commit}-dirty" if dirty else commit
