"""Bake-off runner: same fixtures, same settings, one model at a time.

Each model starts cold (previous model evicted, where the runtime allows it) so its first
call measures load time; the rest run warm. Every answer is scored deterministically and
recorded twice: a JSONL row with full provenance for the report, and a verified outcome in
the capability ledger under the contestant's exact placement, so the router learns from it.
Inference goes through the provider factory like every other model call.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import fmean, median
from typing import Any

from axiom_council.bus.spotlight import fence
from axiom_council.contracts import InferenceResult, TaskClass, content_hash, utcnow
from axiom_council.energy import EnergyMeter, EnergyReading
from axiom_council.evals.code_task import code_messages, verify_code
from axiom_council.evals.fixtures import CodeFixture, FixtureSet, PlanFixture, ReviewFixture
from axiom_council.evals.repair import feedback, public_failures
from axiom_council.evals.run_provenance import RunProvenance, run_provenance
from axiom_council.evals.scoring import Score, score_plan, score_review
from axiom_council.ledger import CapabilityLedger
from axiom_council.provenance import fixture_hash, think_value
from axiom_council.providers import ChatProvider
from axiom_council.registry import PromptTemplate, Registry
from axiom_council.sandbox import DockerSandbox, Snapshot

Fixture = PlanFixture | ReviewFixture | CodeFixture
Attempt = tuple[InferenceResult, EnergyReading]


@dataclass(frozen=True)
class Contestant:
    """A model under test; think=None uses the model card's reasoning setting."""

    model_id: str
    think: bool | str | None = None


@dataclass(frozen=True)
class RunRecord:
    """One fixture for one contestant. A measurement that was not taken is None, not 0."""

    model_id: str
    fixture_id: str
    kind: str
    passed: bool
    checks: dict[str, bool]
    note: str
    cold: bool
    load_seconds: float | None
    ttft_seconds: float | None
    prompt_tokens: int
    output_tokens: int
    decode_tps: float | None
    total_seconds: float
    thinking_chars: int | None
    done_reason: str | None
    ts: str
    fixture_hash: str
    prompt_id: str
    prompt_hash: str
    params: dict[str, Any]
    provenance: RunProvenance
    output: str = ""
    energy_wh: float = 0.0
    mean_watts: float = 0.0
    energy_method: str = "none"
    energy_scope: str = "none"
    energy_estimated: bool = True
    co2e_g: float = 0.0
    attempts: int = 1
    passed_first: bool = False
    retry_reasons: tuple[str, ...] = ()
    failure_category: str | None = None  # None | empty_output | public_checks | hidden_checks


@dataclass
class CodeContext:
    sandbox: DockerSandbox
    base: Snapshot
    work_root: Path


def _timings(results: list[InferenceResult]) -> dict[str, Any]:
    """Sum over attempts; any attempt without a timing makes the total unknown."""
    usages = [r.usage for r in results]
    first = usages[0]
    ttft = (
        first.load_seconds + first.prompt_seconds
        if first.load_seconds is not None and first.prompt_seconds is not None
        else None
    )
    decodes = [u.decode_seconds for u in usages]
    decode_s = None if None in decodes else sum(d for d in decodes if d is not None)
    output_tokens = sum(u.completion_tokens for u in usages)
    thinking = [r.reasoning_chars for r in results]
    return {
        "load_seconds": first.load_seconds,
        "ttft_seconds": ttft,
        "prompt_tokens": sum(u.prompt_tokens for u in usages),
        "output_tokens": output_tokens,
        "decode_tps": output_tokens / decode_s if decode_s else None,
        "total_seconds": sum(u.wall_seconds for u in usages),
        "thinking_chars": None if None in thinking else sum(t for t in thinking if t),
    }


def _failure(passed: bool, last: InferenceResult, public_failed: list[str]) -> str | None:
    if passed:
        return None
    if not last.output.strip() and last.finish_reason == "length":
        return "empty_output"
    return "public_checks" if public_failed else "hidden_checks"


@dataclass
class Bakeoff:
    registry: Registry
    fixtures: FixtureSet
    provider: ChatProvider
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

    def _score(self, fixture: Fixture, answer: str, run_id: str) -> Score:
        if isinstance(fixture, PlanFixture):
            return score_plan(fixture, answer)
        if isinstance(fixture, ReviewFixture):
            return score_review(fixture, answer)
        assert self.code is not None
        return verify_code(
            fixture, answer, self.code.base, self.code.sandbox, self.code.work_root, run_id
        )

    def _tasks(self) -> list[tuple[str, Fixture]]:
        tasks: list[tuple[str, Fixture]] = [("plan", f) for f in self.fixtures.plan]
        tasks += [("review", f) for f in self.fixtures.review]
        if self.code is not None:
            tasks += [("code", f) for f in self.fixtures.code]
        return tasks

    def run(self, contestants: list[Contestant]) -> list[RunRecord]:
        for contestant in contestants:
            prov = run_provenance(
                self.registry, self.provider, model_id=contestant.model_id,
                think=contestant.think, node_id=self.node_id, run_id=self.run_id,
                target_commit=self.code.base.commit if self.code else None,
            )  # fmt: skip
            models = dict.fromkeys(c.model_id for c in contestants)
            evicted = all([self.provider.unload(model) for model in models])
            for index, (kind, fixture) in enumerate(self._tasks()):
                self._run_one(contestant, prov, kind, fixture, cold=index == 0 and evicted)
        return self.records

    def _params(self, contestant: Contestant, kind: str) -> dict[str, Any]:
        """Every inference parameter, recorded as sent."""
        cfg = self.registry.settings.bakeoff
        card = self.registry.models[contestant.model_id]
        return {
            "temperature": cfg.temperature,
            "num_ctx": cfg.num_ctx,
            "num_predict": cfg.kinds[kind].num_predict,
            "think": think_value(card, contestant.think),
            "seed": cfg.seed,
            "request_defaults": dict(card.request_defaults),
        }

    def _ask(
        self, contestant: Contestant, params: dict[str, Any], messages: list[dict[str, str]],
        task_id: str,
    ) -> Attempt:  # fmt: skip
        extra: dict[str, Any] = {**params["request_defaults"], "num_ctx": params["num_ctx"]}
        for key in ("think", "seed"):
            if params[key] is not None:
                extra[key] = params[key]
        with self.meter.measure() as measured:  # inference only, not the sandbox gates
            result = self.provider.chat(
                task_id=task_id, model_id=contestant.model_id, messages=messages,
                max_tokens=params["num_predict"], temperature=params["temperature"],
                extra=extra,
            )  # fmt: skip
        assert measured.reading is not None
        return result, measured.reading

    def _attempts_allowed(self) -> int:
        extra = self.repair_attempts
        return 1 + (self.registry.settings.bakeoff.repair_attempts if extra is None else extra)

    def _run_one(
        self, contestant: Contestant, prov: RunProvenance, kind: str, fixture: Fixture,
        cold: bool,
    ) -> None:  # fmt: skip
        """First answer, then bounded repairs driven only by failed public checks."""
        public = self.registry.settings.bakeoff.kinds[kind].public_checks
        params = self._params(contestant, kind)
        messages = self._messages(kind, fixture)
        run_id = f"{contestant.model_id.replace(':', '-')}-{prov.reasoning}-{fixture.id}"
        tries: list[Attempt] = []
        scores: list[Score] = []
        reasons: list[str] = []
        for attempt in range(self._attempts_allowed()):
            result, reading = self._ask(contestant, params, messages, f"{run_id}-{attempt}")
            tries.append((result, reading))
            scores.append(self._score(fixture, result.output, f"{run_id}-{attempt}"))
            hint = None if scores[-1].passed else feedback(fixture, scores[-1], public)
            if hint is None or attempt + 1 == self._attempts_allowed():
                break
            reasons.append(",".join(public_failures(scores[-1], public)))
            messages = [
                *messages,
                {"role": "assistant", "content": result.output},
                {"role": "user", "content": hint},
            ]
        record = self._record(contestant, prov, kind, fixture, cold, tries, scores, params)
        record = replace(
            record,
            retry_reasons=tuple(reasons),
            failure_category=_failure(
                record.passed, tries[-1][0], public_failures(scores[-1], public)
            ),
        )
        self._persist(record, self.registry.settings.bakeoff.kinds[kind].task_class)

    def _record(
        self, contestant: Contestant, prov: RunProvenance, kind: str, fixture: Fixture,
        cold: bool, tries: list[Attempt], scores: list[Score], params: dict[str, Any],
    ) -> RunRecord:  # fmt: skip
        """One row per fixture; time, tokens and energy are summed over every attempt."""
        results = [result for result, _ in tries]
        last, score, prompt = results[-1], scores[-1], self._prompt(kind)
        return RunRecord(
            model_id=contestant.model_id, fixture_id=fixture.id, kind=kind,
            passed=score.passed, checks=score.checks, note=score.note[:600], cold=cold,
            done_reason=last.finish_reason, output=last.output,
            attempts=len(tries), passed_first=scores[0].passed,
            ts=utcnow().isoformat(), fixture_hash=fixture_hash(fixture),
            prompt_id=prompt.prompt_id, prompt_hash=content_hash(prompt.model_dump()),
            params=params, provenance=prov,
            **_timings(results),
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
        prov = record.provenance
        self.ledger.record(
            task_id=f"{self.run_id}-{record.model_id}-{prov.reasoning}-{record.fixture_id}",
            placement=prov.placement,
            task_class=task_class,
            success=record.passed,
            verified_by="bakeoff-scorer",
            wall_seconds=record.total_seconds,
            total_tokens=record.prompt_tokens + record.output_tokens,
            split=self.split,
            run_id=self.run_id,
            item_key=f"bakeoff:{record.fixture_id}",
            harness_hash=prov.harness_hash,
        )


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(fmean(known), 2) if known else None


def summarise(records: list[RunRecord]) -> list[dict[str, Any]]:
    """One row per placement tested (model and reasoning setting); unknowns stay None."""
    by_model: dict[tuple[str, str | None], list[RunRecord]] = defaultdict(list)
    for record in records:
        by_model[(record.model_id, record.provenance.reasoning)].append(record)
    rows: list[dict[str, Any]] = []
    for (model, reasoning), runs in by_model.items():
        warm = [r for r in runs if not r.cold] or runs
        row: dict[str, Any] = {"model": model, "reasoning": reasoning}
        row["passed"] = sum(r.passed for r in runs)
        row["total"] = len(runs)
        for kind in dict.fromkeys(r.kind for r in runs):
            subset = [r for r in runs if r.kind == kind]
            row[kind] = f"{sum(r.passed for r in subset)}/{len(subset)}"
        loads = [r.load_seconds for r in runs if r.cold and r.load_seconds is not None]
        ttfts = [r.ttft_seconds for r in warm if r.ttft_seconds is not None]
        row["cold_load_s"] = round(max(loads), 1) if loads else None
        row["median_ttft_s"] = round(median(ttfts), 2) if ttfts else None
        row["decode_tps"] = _mean([r.decode_tps for r in runs])
        row["total_minutes"] = round(sum(r.total_seconds for r in runs) / 60, 1)
        row["first_try"] = sum(r.passed_first for r in runs)
        row["mean_attempts"] = round(fmean(r.attempts for r in runs), 2)
        row["energy_wh"] = round(sum(r.energy_wh for r in runs), 2)
        row["co2e_g"] = round(sum(r.co2e_g for r in runs), 2)
        row["energy_methods"] = sorted({r.energy_method for r in runs})
        rows.append(row)
    return rows
