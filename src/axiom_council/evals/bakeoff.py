"""Bake-off runner: same fixtures, same settings, one model at a time.

Each model starts cold (previous model evicted) so its first call measures load time; the
rest run warm. Every answer is scored deterministically and recorded twice: a JSONL row for
the report and a verified outcome in the capability ledger, so the router learns from it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean, median
from typing import Any

from axiom_council.bus.spotlight import fence
from axiom_council.contracts import TaskClass
from axiom_council.energy import EnergyMeter, EnergyReading
from axiom_council.evals.code_task import code_messages, verify_code
from axiom_council.evals.fixtures import CodeFixture, FixtureSet, PlanFixture, ReviewFixture
from axiom_council.evals.ollama_metrics import ChatReply, OllamaChat
from axiom_council.evals.scoring import Score, score_plan, score_review
from axiom_council.ledger import CapabilityLedger
from axiom_council.registry import PromptTemplate, Registry
from axiom_council.sandbox import DockerSandbox, Snapshot

Fixture = PlanFixture | ReviewFixture | CodeFixture


@dataclass(frozen=True)
class Contestant:
    """A model under test; think=None uses the model card's reasoning setting."""

    model_id: str
    think: bool | str | None = None


@dataclass(frozen=True)
class RunRecord:
    model_id: str
    fixture_id: str
    kind: str
    passed: bool
    checks: dict[str, bool]
    note: str
    cold: bool
    load_seconds: float
    ttft_seconds: float
    prompt_tokens: int
    output_tokens: int
    decode_tps: float
    total_seconds: float
    thinking_chars: int
    done_reason: str | None
    output: str = ""
    energy_wh: float = 0.0
    mean_watts: float = 0.0
    energy_method: str = "none"
    energy_scope: str = "none"
    energy_estimated: bool = True
    co2e_g: float = 0.0


@dataclass
class CodeContext:
    sandbox: DockerSandbox
    base: Snapshot
    work_root: Path


@dataclass
class Bakeoff:
    registry: Registry
    fixtures: FixtureSet
    chat: OllamaChat
    ledger: CapabilityLedger
    results_path: Path
    node_id: str
    meter: EnergyMeter
    code: CodeContext | None = None
    records: list[RunRecord] = field(default_factory=list)

    def _prompt(self, kind: str) -> PromptTemplate:
        return self.registry.prompts[self.registry.settings.bakeoff.kinds[kind].prompt_id]

    def _messages(self, kind: str, fixture: Fixture) -> list[dict[str, str]]:
        prompt = self._prompt(kind)
        if isinstance(fixture, PlanFixture):
            return prompt.render(
                request=fixture.request,
                candidates=", ".join(fixture.candidates),
                hours=f"{fixture.hours_budget:g}",
            )
        if isinstance(fixture, ReviewFixture):
            return prompt.render(diff=fence(fixture.diff, "DIFF"))
        assert self.code is not None
        return code_messages(prompt, fixture, self.code.base)

    def _score(self, fixture: Fixture, reply: ChatReply, run_id: str) -> Score:
        if isinstance(fixture, PlanFixture):
            return score_plan(fixture, reply.content)
        if isinstance(fixture, ReviewFixture):
            return score_review(fixture, reply.content)
        assert self.code is not None
        return verify_code(
            fixture, reply.content, self.code.base, self.code.sandbox, self.code.work_root, run_id
        )

    def _tasks(self) -> list[tuple[str, Fixture]]:
        tasks: list[tuple[str, Fixture]] = [("plan", f) for f in self.fixtures.plan]
        tasks += [("review", f) for f in self.fixtures.review]
        if self.code is not None:
            tasks += [("code", f) for f in self.fixtures.code]
        return tasks

    def run(self, contestants: list[Contestant]) -> list[RunRecord]:
        for contestant in contestants:
            for other in contestants:
                self.chat.unload(other.model_id)
            for index, (kind, fixture) in enumerate(self._tasks()):
                self._run_one(contestant, kind, fixture, cold=index == 0)
        return self.records

    def _think(self, contestant: Contestant) -> bool | str | None:
        if contestant.think is not None:
            return contestant.think
        return self.registry.models[contestant.model_id].think

    def _run_one(self, contestant: Contestant, kind: str, fixture: Fixture, cold: bool) -> None:
        cfg = self.registry.settings.bakeoff
        messages = self._messages(kind, fixture)
        with self.meter.measure() as measured:  # inference only, not the sandbox gates
            reply = self.chat.chat(
                contestant.model_id,
                messages,
                think=self._think(contestant),
                num_ctx=cfg.num_ctx,
                num_predict=cfg.kinds[kind].num_predict,
                temperature=cfg.temperature,
            )
        assert measured.reading is not None
        run_id = f"{contestant.model_id.replace(':', '-')}-{fixture.id}"
        score = self._score(fixture, reply, run_id)
        m = reply.metrics
        record = RunRecord(
            model_id=contestant.model_id, fixture_id=fixture.id, kind=kind,
            passed=score.passed, checks=score.checks, note=score.note[:600], cold=cold,
            load_seconds=m.load_seconds, ttft_seconds=m.ttft_seconds,
            prompt_tokens=m.prompt_tokens, output_tokens=m.output_tokens,
            decode_tps=m.decode_tps, total_seconds=m.total_seconds,
            thinking_chars=len(reply.thinking), done_reason=reply.done_reason,
            output=reply.content, **self._energy(measured.reading),
        )  # fmt: skip
        self._persist(record, cfg.kinds[kind].task_class)

    def _energy(self, reading: EnergyReading) -> dict[str, Any]:
        factor = self.registry.settings.energy.grid_kg_co2e_per_kwh
        return {
            "energy_wh": round(reading.watt_hours, 4),
            "mean_watts": round(reading.mean_watts, 2),
            "energy_method": reading.method,
            "energy_scope": reading.scope,
            "energy_estimated": reading.estimated,
            "co2e_g": round(reading.watt_hours * factor, 4),  # Wh x kg/kWh = g
        }

    def _persist(self, record: RunRecord, task_class: TaskClass) -> None:
        self.records.append(record)
        with self.results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")
        self.ledger.record(
            task_id=f"bakeoff-{record.fixture_id}",
            model_id=record.model_id,
            node_id=self.node_id,
            task_class=task_class,
            success=record.passed,
            verified_by="bakeoff-scorer",
            wall_seconds=record.total_seconds,
            total_tokens=record.prompt_tokens + record.output_tokens,
        )


def summarise(records: list[RunRecord]) -> list[dict[str, Any]]:
    by_model: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_model[record.model_id].append(record)
    rows: list[dict[str, Any]] = []
    for model, runs in by_model.items():
        warm = [r for r in runs if not r.cold] or runs
        row: dict[str, Any] = {"model": model, "passed": sum(r.passed for r in runs)}
        row["total"] = len(runs)
        for kind in dict.fromkeys(r.kind for r in runs):
            subset = [r for r in runs if r.kind == kind]
            if subset:
                row[kind] = f"{sum(r.passed for r in subset)}/{len(subset)}"
        row["cold_load_s"] = round(max((r.load_seconds for r in runs if r.cold), default=0), 1)
        row["median_ttft_s"] = round(median(r.ttft_seconds for r in warm), 2)
        row["decode_tps"] = round(fmean(r.decode_tps for r in runs), 1)
        row["total_minutes"] = round(sum(r.total_seconds for r in runs) / 60, 1)
        row["energy_wh"] = round(sum(r.energy_wh for r in runs), 2)
        row["co2e_g"] = round(sum(r.co2e_g for r in runs), 2)
        row["energy_methods"] = sorted({r.energy_method for r in runs})
        rows.append(row)
    return rows
