"""Regression tests for D-12 (run provenance) and D-04 (routing granularity).

D-12: every bake-off record names the exact model build, runtime, node, prompt, parameters,
config and harness behind it; a measurement that was not taken is None, never zero.
D-04: routing evidence is kept per placement (model build, quantisation, runtime, node and
reasoning setting), so evidence for one placement never decides another.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from axiom_council.contracts import TaskClass, TaskSpec
from axiom_council.energy import EstimateMeter
from axiom_council.evals.bakeoff import Bakeoff, Contestant, RunRecord
from axiom_council.evals.fixtures import FixtureSet, load_fixtures
from axiom_council.ledger import CapabilityLedger
from axiom_council.provenance import (
    Placement,
    config_hash,
    fixture_hash,
    harness_hash,
    placement_for,
    reasoning_setting,
)
from axiom_council.registry import Registry
from axiom_council.router import Router

from .fake_provider import FakeProvider
from .test_bakeoff import GOOD_PLAN, REPO

QWEN = "qwen3:4b"


def _bake(
    registry: Registry, tmp_path: Path, provider: FakeProvider, fixtures: FixtureSet | None = None
) -> tuple[Bakeoff, CapabilityLedger]:
    ledger = CapabilityLedger(tmp_path / "ledger.sqlite3")
    bake = Bakeoff(
        registry,
        fixtures or load_fixtures(REPO / "evals" / "fixtures"),
        provider,
        ledger,
        tmp_path / "results-x.jsonl",
        "mac-m5",
        EstimateMeter(30),
    )
    return bake, ledger


def _plans_only(registry: Registry, tmp_path: Path, **kw: object) -> list[RunRecord]:
    fixtures = load_fixtures(REPO / "evals" / "fixtures")
    provider = FakeProvider({"plan": json.dumps(GOOD_PLAN), "review": "{}"}, **kw)  # type: ignore[arg-type]
    bake, ledger = _bake(registry, tmp_path, provider, FixtureSet(plan=fixtures.plan[:1]))
    try:
        return bake.run([Contestant(QWEN)])
    finally:
        ledger.close()


# --- D-12 -------------------------------------------------------------------------------


def test_every_record_carries_full_provenance(registry: Registry, tmp_path: Path) -> None:
    record = _plans_only(registry, tmp_path)[0]
    card, node = registry.models[QWEN], registry.nodes["mac-m5"]
    prov = record.provenance
    assert prov.run_id == "results-x" and prov.node_id == "mac-m5"
    assert prov.model_digest == card.digest and prov.quantization == card.quantization
    assert prov.runtime == node.runtime and prov.runtime_version == "0.0-test"
    assert prov.hardware == node.hardware and prov.reasoning == reasoning_setting(card)
    assert prov.config_hash == config_hash(registry)
    assert prov.harness_hash == harness_hash(registry)
    assert prov.source_commit  # this checkout is a git repository
    assert record.prompt_id == "dispatcher.plan" and len(record.prompt_hash) == 64
    assert record.fixture_hash and record.ts
    assert record.params["temperature"] == registry.settings.bakeoff.temperature
    assert record.params["num_ctx"] == registry.settings.bakeoff.num_ctx
    assert "seed" in record.params


def test_results_file_has_the_provenance_block(registry: Registry, tmp_path: Path) -> None:
    _plans_only(registry, tmp_path)
    row = json.loads((tmp_path / "results-x.jsonl").read_text().splitlines()[0])
    assert row["provenance"]["harness_hash"] == harness_hash(registry)
    assert row["provenance"]["model_digest"] == registry.models[QWEN].digest


def test_missing_measurements_are_none_not_zero(registry: Registry, tmp_path: Path) -> None:
    record = _plans_only(registry, tmp_path, timings=False)[0]
    assert record.load_seconds is None and record.ttft_seconds is None
    assert record.decode_tps is None and record.thinking_chars is None
    assert record.total_seconds > 0  # wall time is always measured


def test_harness_hash_tracks_prompts_and_settings_not_models(registry: Registry) -> None:
    prompts = dict(registry.prompts)
    first = next(iter(prompts))
    prompts[first] = prompts[first].model_copy(update={"system": "changed"})
    new_prompt = registry.model_copy(update={"prompts": prompts})
    router = registry.settings.router.model_copy(update={"min_success": 0.9})
    new_setting = registry.model_copy(
        update={"settings": registry.settings.model_copy(update={"router": router})}
    )
    card = registry.models[QWEN].model_copy(update={"memory_gb": 9.0})
    new_card = registry.model_copy(update={"models": {**registry.models, QWEN: card}})
    base = harness_hash(registry)
    assert harness_hash(new_prompt) != base and harness_hash(new_setting) != base
    assert harness_hash(new_card) == base  # the model is the placement, not the harness
    assert config_hash(new_card) != config_hash(registry)


def test_fixture_hash_changes_when_a_fixture_is_edited() -> None:
    plan = load_fixtures(REPO / "evals" / "fixtures").plan[0]
    edited = plan.model_copy(update={"hours_budget": plan.hours_budget + 1})
    assert fixture_hash(plan) != fixture_hash(edited) and fixture_hash(plan) == fixture_hash(plan)


def test_bakeoff_refuses_a_runtime_build_that_differs_from_the_card(
    registry: Registry, tmp_path: Path
) -> None:
    provider = FakeProvider({"plan": "{}", "review": "{}"}, digests={QWEN: "f" * 64})
    bake, ledger = _bake(registry, tmp_path, provider)
    with pytest.raises(ValueError, match="digest"):
        bake.run([Contestant(QWEN)])
    ledger.close()
    assert provider.calls == []  # refused before any inference


def test_bakeoff_refuses_a_card_without_a_pinned_digest(registry: Registry, tmp_path: Path) -> None:
    card = registry.models[QWEN].model_copy(update={"digest": None})
    unpinned = registry.model_copy(update={"models": {**registry.models, QWEN: card}})
    provider = FakeProvider({"plan": "{}", "review": "{}"}, digests={QWEN: "a" * 64})
    bake, ledger = _bake(unpinned, tmp_path, provider)
    with pytest.raises(ValueError, match="pin"):
        bake.run([Contestant(QWEN)])
    ledger.close()


def test_matching_runtime_digest_is_accepted(registry: Registry, tmp_path: Path) -> None:
    digest = registry.models[QWEN].digest
    assert digest is not None
    record = _plans_only(registry, tmp_path, digests={QWEN: digest})[0]
    assert record.provenance.runtime_digest == digest


# --- D-04 -------------------------------------------------------------------------------


def _fail(ledger: CapabilityLedger, placement: Placement, count: int = 40) -> None:
    for i in range(count):
        ledger.record(task_id=f"f{i}", placement=placement, task_class=TaskClass.TRIAGE,
                      success=False, verified_by="pytest", wall_seconds=1.0,
                      total_tokens=10)  # fmt: skip


def test_evidence_is_kept_per_placement(tmp_path: Path) -> None:
    ledger = CapabilityLedger(tmp_path / "l.sqlite3")
    base = Placement("m", "mac-m5", "ollama", "abc", "Q4_K_M", "low")
    _fail(ledger, base, 3)
    variants = [
        Placement("m", "win-4060", "ollama", "abc", "Q4_K_M", "low"),
        Placement("m", "mac-m5", "llama.cpp", "abc", "Q4_K_M", "low"),
        Placement("m", "mac-m5", "ollama", "def", "Q4_K_M", "low"),
        Placement("m", "mac-m5", "ollama", "abc", "Q8_0", "low"),
        Placement("m", "mac-m5", "ollama", "abc", "Q4_K_M", "false"),
    ]
    assert ledger.stats(base, TaskClass.TRIAGE).trials == 3
    assert all(ledger.stats(v, TaskClass.TRIAGE).trials == 0 for v in variants)
    ledger.close()


def test_two_reasoning_settings_of_one_model_share_a_run(tmp_path: Path) -> None:
    ledger = CapabilityLedger(tmp_path / "l.sqlite3")
    for reasoning in ("low", "false"):
        ledger.record(task_id=f"r-{reasoning}", placement=Placement("m", "n", reasoning=reasoning),
                      task_class=TaskClass.PLAN, success=True, verified_by="scorer",
                      wall_seconds=1.0, total_tokens=1, split="calibration", run_id="run-1",
                      item_key="plan-a")  # fmt: skip
    with pytest.raises(ValueError, match="already recorded"):
        ledger.record(task_id="again", placement=Placement("m", "n", reasoning="low"),
                      task_class=TaskClass.PLAN, success=True, verified_by="scorer",
                      wall_seconds=1.0, total_tokens=1, split="calibration", run_id="run-1",
                      item_key="plan-a")  # fmt: skip
    ledger.close()


def test_harness_filter_is_optional(tmp_path: Path) -> None:
    ledger = CapabilityLedger(tmp_path / "l.sqlite3")
    where = Placement("m", "n")
    ledger.record(task_id="t", placement=where, task_class=TaskClass.PLAN, success=True,
                  verified_by="scorer", wall_seconds=1.0, total_tokens=1,
                  harness_hash="h1")  # fmt: skip
    assert ledger.stats(where, TaskClass.PLAN).trials == 1
    assert ledger.stats(where, TaskClass.PLAN, harness_hash="h1").trials == 1
    assert ledger.stats(where, TaskClass.PLAN, harness_hash="h2").trials == 0
    ledger.close()


def test_router_reads_evidence_for_the_exact_placement(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    ledger = CapabilityLedger(tmp_path / "l.sqlite3")
    exact = placement_for(registry, QWEN, "mac-m5")
    other_build = Placement(QWEN, "mac-m5", exact.runtime, "0" * 64, exact.quantization,
                            exact.reasoning)  # fmt: skip
    _fail(ledger, other_build)
    router = Router(registry, ledger)
    chosen = {c.model_id for c in router.route(make_task(), prompt_tokens=100).ranked}
    assert QWEN in chosen  # failures of another build do not count against this one
    _fail(ledger, exact)
    decision = router.route(make_task(), prompt_tokens=100)
    assert QWEN not in {c.model_id for c in decision.ranked}
    assert any(r.model_id == QWEN and "40 verified" in r.reason for r in decision.rejected)
    ledger.close()


def test_router_can_require_the_current_harness(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    ledger = CapabilityLedger(tmp_path / "l.sqlite3")
    exact = placement_for(registry, QWEN, "mac-m5")
    for i in range(40):
        ledger.record(task_id=f"o{i}", placement=exact, task_class=TaskClass.TRIAGE,
                      success=False, verified_by="pytest", wall_seconds=1.0, total_tokens=10,
                      harness_hash="an-older-harness")  # fmt: skip
    strict = registry.settings.router.model_copy(update={"match_harness": True})
    ranked = Router(registry, ledger, strict).route(make_task(), prompt_tokens=100).ranked
    assert QWEN in {c.model_id for c in ranked}  # old-harness failures are ignored
    ranked = Router(registry, ledger).route(make_task(), prompt_tokens=100).ranked
    assert QWEN not in {c.model_id for c in ranked}  # default: any harness counts
    ledger.close()


def test_placement_reasoning_follows_card_then_override(registry: Registry) -> None:
    gemma, qwen_small = registry.models["gemma4:26b"], registry.models[QWEN]
    assert reasoning_setting(gemma) == "false"
    assert reasoning_setting(qwen_small) == "low"  # from request_defaults.reasoning_effort
    assert reasoning_setting(gemma, "high") == "high"
    assert placement_for(registry, QWEN, "mac-m5", think=False).reasoning == "false"


def test_bakeoff_ledger_rows_use_the_contestant_placement(
    registry: Registry, tmp_path: Path
) -> None:
    fixtures = load_fixtures(REPO / "evals" / "fixtures")
    provider = FakeProvider({"plan": json.dumps(GOOD_PLAN), "review": "{}"})
    bake, ledger = _bake(registry, tmp_path, provider, FixtureSet(plan=fixtures.plan[:2]))
    bake.run([Contestant(QWEN), Contestant(QWEN, False)])
    low = placement_for(registry, QWEN, "mac-m5")
    off = placement_for(registry, QWEN, "mac-m5", think=False)
    assert ledger.stats(low, TaskClass.PLAN).trials == 2
    assert ledger.stats(off, TaskClass.PLAN).trials == 2
    ledger.close()
    assert provider.calls[0][1]["think"] == "low" and provider.calls[-1][1]["think"] is False


def test_bakeoff_refuses_a_different_quantisation(registry: Registry, tmp_path: Path) -> None:
    provider = FakeProvider({"plan": "{}", "review": "{}"}, quantization="Q8_0")
    bake, ledger = _bake(registry, tmp_path, provider)
    with pytest.raises(ValueError, match="quantisation"):
        bake.run([Contestant(QWEN)])
    ledger.close()


def test_an_empty_answer_at_the_length_limit_is_its_own_failure_category(
    registry: Registry, tmp_path: Path
) -> None:
    record = _plans_only_reply(registry, tmp_path, "", finish_reason="length")
    assert not record.passed and record.failure_category == "empty_output"
    record = _plans_only_reply(registry, tmp_path / "b", json.dumps({"estimate_hours": 1}))
    assert record.failure_category in {"public_checks", "hidden_checks"}


def _plans_only_reply(
    registry: Registry, tmp_path: Path, reply: str, finish_reason: str = "stop"
) -> RunRecord:
    fixtures = load_fixtures(REPO / "evals" / "fixtures")
    provider = FakeProvider([reply], finish_reason=finish_reason)
    bake, ledger = _bake(registry, tmp_path, provider, FixtureSet(plan=fixtures.plan[:1]))
    try:
        return bake.run([Contestant(QWEN)])[0]
    finally:
        ledger.close()


def test_source_commit_outside_git_and_with_local_edits(monkeypatch: pytest.MonkeyPatch) -> None:
    from axiom_council import provenance

    provenance.source_commit.cache_clear()
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert provenance.source_commit() is None
    provenance.source_commit.cache_clear()
    answers = iter(["abc123", " M src/x.py"])
    monkeypatch.setattr(provenance, "_git", lambda *args: next(answers))
    assert provenance.source_commit() == "abc123-dirty"
    provenance.source_commit.cache_clear()


def test_hashes_do_not_depend_on_the_hash_seed() -> None:
    import subprocess
    import sys

    config = str(REPO / "config")
    script = (
        "import tempfile; from pathlib import Path\n"
        "from axiom_council.configstore import ConfigStore\n"
        "from axiom_council.registry import Registry\n"
        "from axiom_council.provenance import harness_hash\n"
        f"s = ConfigStore(Path(tempfile.mkdtemp()) / 'c.db'); s.seed(Path({config!r}))\n"
        "print(harness_hash(Registry.from_store(s)))\n"
    )
    seen = set()
    for seed in ("1", "2", "3"):
        env = {"PYTHONHASHSEED": seed, "PATH": ""}
        run = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env
        )
        seen.add(run.stdout)
    assert len(seen) == 1


def test_a_ledger_with_the_old_unique_index_is_migrated(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "ledger.sqlite3"
    CapabilityLedger(path).close()
    db = sqlite3.connect(path)
    with db:  # recreate the pre-D-04 state: old index, no placement columns used
        db.execute("DROP INDEX ux_outcomes_run_item_placement")
        db.execute(
            "CREATE UNIQUE INDEX ux_outcomes_run_item ON outcomes (run_id, item_key, model_id)"
            " WHERE item_key IS NOT NULL"
        )
    db.close()
    ledger = CapabilityLedger(path)
    for reasoning in ("low", "false"):  # the old index would reject the second row
        ledger.record(task_id=reasoning, placement=Placement("m", "n", reasoning=reasoning),
                      task_class=TaskClass.PLAN, success=True, verified_by="scorer",
                      wall_seconds=1.0, total_tokens=1, split="calibration", run_id="r",
                      item_key="i")  # fmt: skip
    ledger.close()
