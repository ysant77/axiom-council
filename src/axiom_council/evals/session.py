"""Assemble a bake-off from config: fixtures, ledger, results file and (optionally) the
PageTrace sandbox for code fixtures."""

from __future__ import annotations

from pathlib import Path

from axiom_council.contracts import utcnow
from axiom_council.evals.bakeoff import Bakeoff, CodeContext, Contestant, RunRecord
from axiom_council.evals.fixtures import load_fixtures
from axiom_council.evals.ollama_metrics import OllamaChat
from axiom_council.ledger import CapabilityLedger
from axiom_council.registry import Registry
from axiom_council.sandbox import DockerSandbox, create_snapshot


def parse_contestants(spec: str) -> list[Contestant]:
    """`model[=think]` comma-separated; think is true, false, low, medium or high."""
    contestants: list[Contestant] = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        model, _, think = item.partition("=")
        value: bool | str | None = None
        if think in {"true", "false"}:
            value = think == "true"
        elif think:
            value = think
        contestants.append(Contestant(model, value))
    return contestants


def _code_context(registry: Registry, home: Path, sandbox_id: str) -> CodeContext:
    profile = registry.sandboxes[sandbox_id]
    sandbox = DockerSandbox(profile)
    stamp = utcnow().strftime("%Y%m%d%H%M%S")
    base = create_snapshot(profile.repo_path(), "HEAD", home / "bakeoff" / f"base-{stamp}")
    sandbox.prepare(base)
    return CodeContext(sandbox, base, home / "bakeoff" / f"work-{stamp}")


def run_bakeoff(
    registry: Registry,
    home: Path,
    fixtures_dir: Path,
    contestants: list[Contestant],
    *,
    with_code: bool,
    sandbox_id: str = "pagetrace",
) -> tuple[list[RunRecord], Path]:
    unknown = [c.model_id for c in contestants if c.model_id not in registry.models]
    if unknown:
        raise ValueError(f"models without a model card: {unknown}")
    results = home / "bakeoff" / f"results-{utcnow().strftime('%Y%m%d%H%M%S')}.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    node = registry.nodes["mac-m5"]
    chat = OllamaChat(node.base_url)
    ledger = CapabilityLedger(home / "ledger.sqlite3")
    try:
        bakeoff = Bakeoff(
            registry=registry,
            fixtures=load_fixtures(fixtures_dir),
            chat=chat,
            ledger=ledger,
            results_path=results,
            code=_code_context(registry, home, sandbox_id) if with_code else None,
        )
        return bakeoff.run(contestants), results
    finally:
        chat.close()
        ledger.close()
