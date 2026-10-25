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
from axiom_council.evals.repair import feedback
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
    attempts: int = 1
    passed_first: bool = False


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
    split: str = "calibration"  # development | calibration | holdout
    repair_attempts: int | None = None  # None: use the bake-off settings
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

    def _ask(
        self, contestant: Contestant, kind: str, messages: list[dict[str, str]]
    ) -> tuple[ChatReply, EnergyReading]:
        cfg = self.registry.settings.bakeoff
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
        return reply, measured.reading

    def _attempts_allowed(self) -> int:
        extra = self.repair_attempts
        return 1 + (self.registry.settings.bakeoff.repair_attempts if extra is None else extra)

    def _run_one(self, contestant: Contestant, kind: str, fixture: Fixture, cold: bool) -> None:
        """First answer, then bounded repairs driven only by failed public checks."""
        public = self.registry.settings.bakeoff.kinds[kind].public_checks
        messages = self._messages(kind, fixture)
        run_id = f"{contestant.model_id.replace(':', '-')}-{fixture.id}"
        tries: list[tuple[ChatReply, EnergyReading]] = []
        scores: list[Score] = []
        for attempt in range(self._attempts_allowed()):
            reply, reading = self._ask(contestant, kind, messages)
            tries.append((reply, reading))
            scores.append(self._score(fixture, reply, f"{run_id}-{attempt}"))
            hint = None if scores[-1].passed else feedback(fixture, scores[-1], public)
            if hint is None:
                break
            messages = [
                *messages,
                {"role": "assistant", "content": reply.content},
                {"role": "user", "content": hint},
            ]
        record = self._record(contestant.model_id, kind, fixture, cold, tries, scores)
        self._persist(record, self.registry.settings.bakeoff.kinds[kind].task_class)

    def _record(
        self, model_id: str, kind: str, fixture: Fixture, cold: bool,
        tries: list[tuple[ChatReply, EnergyReading]], scores: list[Score],
    ) -> RunRecord:  # fmt: skip
        """One row per fixture; time, tokens and energy are summed over every attempt."""
        replies = [reply for reply, _ in tries]
        first, last, score = replies[0].metrics, replies[-1], scores[-1]
        out_tokens = sum(r.metrics.output_tokens for r in replies)
        decode_s = sum(r.metrics.decode_seconds for r in replies)
        return RunRecord(
            model_id=model_id, fixture_id=fixture.id, kind=kind,
            passed=score.passed, checks=score.checks, note=score.note[:600], cold=cold,
            load_seconds=first.load_seconds, ttft_seconds=first.ttft_seconds,
            prompt_tokens=sum(r.metrics.prompt_tokens for r in replies),
            output_tokens=out_tokens,
            decode_tps=out_tokens / decode_s if decode_s else 0.0,
            total_seconds=sum(r.metrics.total_seconds for r in replies),
            thinking_chars=sum(len(r.thinking) for r in replies),
            done_reason=last.done_reason, output=last.content,
            attempts=len(tries), passed_first=scores[0].passed,
            **self._energy([reading for _, reading in tries]),
        )  # fmt: skip

    def _energy(self, readings: list[EnergyReading]) -> dict[str, Any]:
        factor = self.registry.settings.energy.grid_kg_co2e_per_kwh
        wh = sum(r.watt_hours for r in readings)
        seconds = sum(r.seconds for r in readings)
        return {
            "energy_wh": round(wh, 4),
            "mean_watts": round(wh * 3600 / seconds, 2) if seconds else 0.0,
            "energy_method": readings[0].method,
            "energy_scope": readings[0].scope,
            "energy_estimated": any(r.estimated for r in readings),
            "co2e_g": round(wh * factor, 4),  # Wh x kg/kWh = g
        }

    @property
    def run_id(self) -> str:
        """One results file per run; its name identifies the run in the ledger."""
        return self.results_path.stem

    def _persist(self, record: RunRecord, task_class: TaskClass) -> None:
        self.records.append(record)
        with self.results_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({**asdict(record), "run_id": self.run_id, "split": self.split}) + "\n"
            )
        if self.split == "holdout":
            return  # holdout results are reported, never fed to routing
        self.ledger.record(
            task_id=f"{self.run_id}-{record.fixture_id}",
            model_id=record.model_id,
            node_id=self.node_id,
            task_class=task_class,
            success=record.passed,
            verified_by="bakeoff-scorer",
            wall_seconds=record.total_seconds,
            total_tokens=record.prompt_tokens + record.output_tokens,
            split=self.split,
            run_id=self.run_id,
            item_key=f"bakeoff:{record.fixture_id}",
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
        row["first_try"] = sum(r.passed_first for r in runs)
        row["mean_attempts"] = round(fmean(r.attempts for r in runs), 2)
        row["energy_wh"] = round(sum(r.energy_wh for r in runs), 2)
        row["co2e_g"] = round(sum(r.co2e_g for r in runs), 2)
        row["energy_methods"] = sorted({r.energy_method for r in runs})
        rows.append(row)
    return rows
