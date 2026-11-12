"""Number-grounding interceptor: every number in a message must appear in its cited evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from axiom_council.audit import AuditLog
from axiom_council.bus import Broker, Envelope, MessageKind, Trust, seal
from axiom_council.bus.envelope import EvidenceItem
from axiom_council.bus.evidence import InMemoryEvidenceLedger
from axiom_council.bus.interceptors import (
    NumberGrounding,
    Route,
    default_chain,
)
from axiom_council.bus.numbers import payload_text, ungrounded_numbers
from axiom_council.registry import Registry
from axiom_council.settings import GroundingSettings, Settings
from tests.grounding_helpers import FLAGGED, PLAIN, issue, make_env

ZW = "\u200b"  # zero-width space
NBSP = "\u00a0"
THIN = "\u202f"  # narrow no-break space

# (message, evidence texts, settings overrides, expected ungrounded numbers in order)
Case = tuple[str, tuple[str, ...], dict[str, Any], list[str]]

CASES: dict[str, Case] = {
    "integer-grounded": ("We ran 12 tests.", ("12 tests passed",), {}, []),
    "integer-ungrounded": ("We ran 13 tests.", ("12 tests passed",), {}, ["13"]),
    "short-not-grounded-by-longer": ("12 items", ("112 items",), {}, ["12"]),
    "long-not-grounded-by-shorter": ("112 items", ("12 items",), {}, ["112"]),
    "decimal-exact": ("ratio 0.5", ("ratio 0.5",), {}, []),
    "decimal-rounded-to-shown-digits": ("score 0.67", ("score 0.6712",), {}, []),
    "decimal-coarser-needs-tolerance": ("score 0.7", ("score 0.6712",), {}, ["0.7"]),
    "decimal-coarser-with-tolerance": ("score 0.7", ("score 0.6712",), {"tol": 0.05}, []),
    "decimal-wrong-rounding": ("score 0.68", ("score 0.6712",), {}, ["0.68"]),
    "decimal-half-rounds-up": ("score 0.68", ("score 0.675",), {}, []),
    "decimal-trailing-zero-in-message": ("p 0.50", ("p 0.5",), {}, []),
    "decimal-trailing-zero-in-evidence": ("p 0.5", ("p 0.50",), {}, []),
    "decimal-long-rounded": ("pi 3.14159", ("pi 3.14159265",), {}, []),
    "decimal-leading-dot": ("rate .5", ("rate 0.5",), {}, []),
    "decimal-min-digits-lowered": ("score 0.7", ("score 0.6712",), {"min": 1}, []),
    "integer-never-rounded": ("13 ms", ("12.6 ms",), {"min": 1}, ["13"]),
    "thousands-in-message": ("1,234 rows", ("1234 rows",), {}, []),
    "thousands-in-evidence": ("1234 rows", ("1,234 rows",), {}, []),
    "thousands-narrow-space": (f"1{THIN}234 rows", ("1234 rows",), {}, []),
    "split-by-space-is-two-numbers": ("1 234 rows", ("1234 rows",), {}, ["1", "234"]),
    "split-by-bad-comma-is-two-numbers": ("12,34 rows", ("1234 rows",), {}, ["12", "34"]),
    "split-by-zero-width-joins": (f"1{ZW}2 rows", ("1 and 2",), {}, ["12"]),
    "negative-not-grounded-by-positive": ("delta -5", ("delta 5",), {}, ["-5"]),
    "positive-not-grounded-by-negative": ("delta 5", ("delta -5",), {}, ["5"]),
    "negative-grounded": ("delta -5", ("delta -5",), {}, []),
    "unicode-minus": ("delta \u22125", ("delta -5",), {}, []),
    "negative-decimal-rounded": ("delta -0.67", ("delta -0.6712",), {}, []),
    "negative-decimal-wrong-sign": ("delta -0.67", ("delta 0.6712",), {}, ["-0.67"]),
    "plus-sign-ignored": ("up +5", ("up 5",), {}, []),
    "range-hyphen-is-not-minus": ("10-20 files", ("10 and 20 files",), {}, []),
    "range-needs-positive-evidence": ("10-20 files", ("10 and -20",), {}, ["20"]),
    "percent-grounded": ("pass rate 67%", ("pass rate 67%",), {}, []),
    "percent-word": ("pass rate 67 percent", ("pass rate 67%",), {}, []),
    "percent-not-grounded-by-plain": ("pass rate 67%", ("67 tests",), {}, ["67%"]),
    "plain-not-grounded-by-percent": ("67 tests", ("pass rate 67%",), {}, ["67"]),
    "percent-not-grounded-by-fraction": ("rate 67%", ("rate 0.67",), {}, ["67%"]),
    "percent-grounded-by-fraction-when-allowed": (
        "rate 67%",
        ("rate 0.67",),
        {"pct": True},
        [],
    ),
    "percent-fraction-rounded-when-allowed": (
        "rate 67.12%",
        ("rate 0.67124",),
        {"pct": True},
        [],
    ),
    "percent-integer-not-rounded-from-fraction": (
        "rate 67%",
        ("rate 0.6712",),
        {"pct": True},
        ["67%"],
    ),
    "percent-fraction-wrong-value": ("rate 68%", ("rate 0.6712",), {"pct": True}, ["68%"]),
    "unicode-percent-sign": ("rate 67\uff05", ("rate 67%",), {}, []),
    "unit-matches-on-number": ("3.5 GB and 12 ms", ("3.5 and 12",), {}, []),
    "unit-attached": ("took 12ms", ("12 ms",), {}, []),
    "unit-wrong-number": ("took 12 ms", ("took 13 ms",), {}, ["12"]),
    "list-markers-ignored": ("1. First step\n2) Second step\n  3. Third", (), {}, []),
    "ordinals-ignored": ("the 3rd run and the 21st run", (), {}, []),
    "finding-ids-ignored": ("see R1.2, F-12 and p99 and #47", (), {}, []),
    "percentile-names-ignored": ("p95 and p99 were 12", ("were 12",), {}, []),
    "iso-date-ignored": ("on 2024-05-12", (), {}, []),
    "numeric-date-ignored": ("on 12/05/2024 and 12.05.2024", (), {}, []),
    "written-date-ignored": ("on March 5, 2024 and 5 March 2024", (), {}, []),
    "version-ignored": ("version 1.2.3, v2.0 and 1.2.3.4 on tool version 3.12", (), {}, []),
    "arabic-indic-digits": ("\u0663\u0664 items", ("34 items",), {}, []),
    "fullwidth-digits": ("\uff11\uff12 items", ("12 items",), {}, []),
    "fullwidth-digits-wrong": ("\uff11\uff13 items", ("12 items",), {}, ["13"]),
    "repeated-number-named-once": ("7 here and 7 there", (), {}, ["7"]),
    "leading-zeros": ("007 agents", ("7 agents",), {}, []),
    "tolerance-relative-passes": ("100 users", ("103 users",), {"tol": 0.05}, []),
    "tolerance-relative-fails": ("100 users", ("103 users",), {"tol": 0.01}, ["100"]),
    "tolerance-zero-evidence-needs-exact": ("0.1", ("0",), {"tol": 0.5}, ["0.1"]),
    "nbsp-separator-groups": (f"1{NBSP}234 rows", ("1234 rows",), {}, []),
    "no-numbers": ("All checks passed with no issues.", (), {}, []),
    "spelled-numbers-out-of-scope": ("twelve tests passed", (), {}, []),
    # Identifier skip is narrow: when in doubt, the digits are read as a number.
    "percent-joined-by-hyphen": ("accuracy-95% on 18 tasks", (), {}, ["95%", "18"]),
    "percent-word-joined-by-hyphen": ("accuracy-95 percent", (), {}, ["95%"]),
    "capitalised-word-joined-percent": ("Accuracy-95% overall", (), {}, ["95%"]),
    "decimal-joined-by-dot": ("score.95 on 18 tasks", (), {}, [".95", "18"]),
    "digits-after-one-lowercase-letter": ("x95 tasks passed", (), {}, ["95"]),
    "digits-after-lowercase-word": ("tasks95 passed", (), {}, ["95"]),
    "digits-after-underscore": ("pass_rate_95 holds", (), {}, ["95"]),
    "lowercase-word-hyphen-number": ("tasks-18 done", (), {}, ["18"]),
    "lowercase-v-without-dots-is-a-number": ("v95 tasks passed", (), {}, ["95"]),
    "unlisted-name-is-a-number": ("node-5 scored 3", (), {}, ["5", "3"]),
    "listed-name-from-setting": ("node-5 scored 3", ("3",), {"allow": ["node-5"]}, []),
    "allowlist-is-case-insensitive": ("P99 scored 3", ("3",), {}, []),
    "allowlist-needs-whole-token": ("p9999 scored 3", ("3",), {}, ["9999"]),
    "caller-model-id-with-tag": ("gemma4:26b passed 18", ("18",), {"ids": ["gemma4:26b"]}, []),
    "caller-model-id-with-dots": ("qwen3.8:27b passed 18", ("18",), {"ids": ["qwen3.8:27b"]}, []),
    "model-id-without-caller-list-is-numbers": ("gemma4:26b passed 18", ("18",), {}, ["4", "26"]),
    "bare-number-next-to-model-id-fails": (
        "gemma4:26b ran 26",
        ("ran",),
        {"ids": ["gemma4:26b"]},
        ["26"],
    ),
    "partial-model-id-is-numbers": ("gemma4:26 passed", (), {"ids": ["gemma4:26b"]}, ["4", "26"]),
    "model-id-is-case-insensitive": ("Gemma4:26B passed", (), {"ids": ["gemma4:26b"]}, []),
    "model-id-then-percent-is-measured": (
        "gemma4:26b scored 90%",
        (),
        {"ids": ["gemma4:26b"]},
        ["90%"],
    ),
    "uppercase-codes-ignored": ("R1.2, PM2.5, F-12, K100 and COVID-19", (), {}, []),
    "mixed-case-codes-ignored": ("pH7 and eBox15 and Node3-8B", (), {}, []),
    "dotted-lowercase-version-ignored": ("v2.0 and V1.4.2", (), {}, []),
    "circled-digits": ("\u2460\u2467 items", ("18 items",), {}, []),
    "circled-digits-wrong": ("\u2460\u2468 items", ("18 items",), {}, ["19"]),
    "vulgar-fraction-is-two-numbers": ("\u00bd cup", (), {}, ["1", "2"]),
    "superscript-digit": ("x\u00b2", (), {}, ["2"]),
    "hair-space-group": ("1\u200a234 rows", ("1234 rows",), {}, []),
    "punctuation-space-group": ("1\u2008234 rows", ("1234 rows",), {}, []),
    "roman-numerals-out-of-scope": ("chapter IV and \u2163", (), {}, []),
}


def _rules(overrides: dict[str, Any]) -> GroundingSettings:
    return GroundingSettings(
        relative_tolerance=overrides.get("tol", 0.0),
        rounding_min_digits=overrides.get("min", 2),
        percent_matches_fraction=overrides.get("pct", False),
        **({"identifier_allowlist": frozenset(overrides["allow"])} if "allow" in overrides else {}),
    )


def test_case_table_is_large_enough() -> None:
    assert len(CASES) >= 25


@pytest.mark.parametrize("case_id", list(CASES))
def test_number_rules(case_id: str) -> None:
    message, evidence, overrides, expected = CASES[case_id]
    found = ungrounded_numbers(message, evidence, _rules(overrides), overrides.get("ids", ()))
    assert found == expected


# --- envelope, interceptor and broker behaviour ---------------------------------------


def _inspect(env: Envelope, *, issued: bool = True, **overrides: Any) -> list[Any]:
    ledger = InMemoryEvidenceLedger()
    if issued:
        issue(ledger, env)
    guard = NumberGrounding([FLAGGED, PLAIN], _rules(overrides), ledger)
    return list(guard.inspect(env))


def test_route_flag_defaults_to_false_and_forbids_extras() -> None:
    assert PLAIN.require_grounded_numbers is False
    with pytest.raises(ValidationError):
        Route(route_id="x", sender="a", recipient="b", kinds=frozenset(), grounded=True)  # type: ignore[call-arg]


def test_ungrounded_number_is_a_blocking_finding_that_names_it() -> None:
    findings = _inspect(make_env({"text": "p99 is 120 ms and 7 retries"}, "p99 is 120 ms"))
    assert len(findings) == 1
    assert findings[0].severity == "reject" and findings[0].interceptor == "number_grounding"
    assert "7" in findings[0].detail and "120" not in findings[0].detail


def test_every_ungrounded_number_is_named() -> None:
    (finding,) = _inspect(make_env({"text": "5 apples, 6 pears, 7 plums"}, "6 pears"))
    assert "5" in finding.detail and "7" in finding.detail and "6" not in finding.detail


def test_numbers_without_evidence_are_rejected() -> None:
    (finding,) = _inspect(make_env({"text": "we measured 42 samples"}))
    assert finding.severity == "reject" and "no issued evidence" in finding.detail
    assert "42" in finding.detail


def test_message_without_numbers_passes_with_or_without_evidence() -> None:
    assert _inspect(make_env({"text": "all good"})) == []
    assert _inspect(make_env({"text": "all good"}, "some 12 evidence")) == []


def test_grounded_message_passes() -> None:
    assert _inspect(make_env({"text": "latency 12 ms"}, "latency: 12 ms", "other 5")) == []


def test_unflagged_route_is_never_inspected() -> None:
    env = make_env(
        {"text": "we measured 42 samples"}, recipient="dispatcher", kind=MessageKind.CLARIFY
    )
    assert _inspect(env) == []


def test_unknown_edge_is_left_to_route_policy() -> None:
    env = make_env({"text": "42"}, recipient="director", kind=MessageKind.VERDICT)
    assert _inspect(env) == []


def test_flag_applies_per_kind_on_the_same_edge() -> None:
    env = make_env({"text": "42"}, kind=MessageKind.REVIEW)
    assert _inspect(env) == []


def test_flag_wins_when_two_routes_cover_one_edge() -> None:
    twin = FLAGGED.model_copy(update={"route_id": "twin", "require_grounded_numbers": False})
    guard = NumberGrounding([twin, FLAGGED])
    assert len(list(guard.inspect(make_env({"text": "42"})))) == 1


def test_evidence_id_does_not_ground_a_number() -> None:
    item = EvidenceItem.of("ev-42", "no figures in this text")
    env = seal(
        task_id="t",
        sender="builder",
        recipient="gatekeeper",
        kind=MessageKind.EDIT,
        payload={"text": "the answer is 42"},
        trust=Trust.UNTRUSTED,
        evidence=(item,),
    )
    assert len(_inspect(env)) == 1


def test_evidence_hash_must_match_text() -> None:
    good = EvidenceItem.of("ev_1", "12 ms")
    with pytest.raises(ValidationError, match="evidence hash"):
        EvidenceItem(evidence_id="ev_1", content_hash=good.content_hash, text="13 ms")


def test_evidence_item_forbids_extra_fields() -> None:
    data = EvidenceItem.of("ev_1", "x").model_dump()
    with pytest.raises(ValidationError):
        EvidenceItem.model_validate({**data, "note": "extra"})


def test_old_envelopes_without_evidence_stay_valid() -> None:
    env = seal(
        task_id="t",
        sender="builder",
        recipient="gatekeeper",
        kind=MessageKind.EDIT,
        payload={"a": 1},
        trust=Trust.UNTRUSTED,
    )
    data = env.model_dump()
    del data["evidence"]
    assert Envelope.model_validate(data).evidence == ()


def test_payload_numbers_and_keys_count_but_bools_do_not() -> None:
    payload = {"p99": 120, "ok": True, "nested": [{"score": 0.5}], "note": "n=3"}
    text = payload_text(payload)
    assert "120" in text and "0.5" in text and "True" not in text and "p99" in text
    findings = _inspect(make_env(payload, "120 0.5"))
    assert len(findings) == 1 and "3" in findings[0].detail


def test_numbers_hidden_in_dictionary_keys_are_read() -> None:
    (finding,) = _inspect(make_env({"95% accuracy": True, "tasks": {"18 passed": "yes"}}, "none"))
    assert "95%" in finding.detail and "18" in finding.detail


def test_float_payload_values_have_no_exponent_form() -> None:
    assert payload_text({"x": 0.00001}).splitlines()[-1] == "0.00001"


def test_settings_section_is_typed_and_validated() -> None:
    assert Settings().grounding == GroundingSettings()
    default = GroundingSettings()
    assert (default.relative_tolerance, default.rounding_min_digits) == (0.0, 2)
    assert default.percent_matches_fraction is False
    with pytest.raises(ValidationError):
        GroundingSettings(relative_tolerance=-0.1)
    with pytest.raises(ValidationError):
        GroundingSettings(rounding_min_digits=0)
    with pytest.raises(ValidationError):
        GroundingSettings(unknown=1)  # type: ignore[call-arg]


def test_seed_config_has_grounding_settings_and_unflagged_routes(registry: Registry) -> None:
    assert registry.settings.grounding == GroundingSettings()
    assert registry.routes
    assert not any(r.require_grounded_numbers for r in registry.routes.values())


def test_default_chain_inspects_only_flagged_routes(registry: Registry, tmp_path: Path) -> None:
    routes = [r for r in registry.routes.values() if r.route_id != "builder-gatekeeper"]
    routes.append(FLAGGED)
    ledger = InMemoryEvidenceLedger()
    env = make_env({"content": "x = 42"}, "limit is 42")
    issue(ledger, env)
    broker = Broker(default_chain(routes, ledger=ledger), AuditLog(tmp_path / "audit.jsonl"))
    broker.subscribe("gatekeeper", lambda env, findings: None)
    broker.subscribe("dispatcher", lambda env, findings: None)

    bad = broker.publish(make_env({"content": "x = 42"}))
    assert not bad.delivered and bad.findings[-1].interceptor == "number_grounding"
    assert broker.publish(env).delivered
    other = broker.publish(
        make_env({"text": "42"}, recipient="dispatcher", kind=MessageKind.CLARIFY)
    )
    assert other.delivered


def test_default_chain_without_flags_changes_nothing(registry: Registry, tmp_path: Path) -> None:
    broker = Broker(default_chain(registry.routes.values()), AuditLog(tmp_path / "a.jsonl"))
    broker.subscribe("gatekeeper", lambda env, findings: None)
    assert broker.publish(make_env({"content": "x = 42"})).delivered


def test_default_chain_takes_grounding_settings() -> None:
    env = make_env({"text": "score 0.7"}, "score 0.6712")
    for tolerance, blocked in ((0.0, True), (0.05, False)):
        ledger = InMemoryEvidenceLedger()
        issue(ledger, env)
        rules = GroundingSettings(relative_tolerance=tolerance)
        chain = default_chain([FLAGGED], grounding=rules, ledger=ledger)
        guard = next(i for i in chain if i.name == "number_grounding")
        assert bool(list(guard.inspect(env))) is blocked


def test_audit_records_evidence_hashes_only_when_present(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    broker = Broker(default_chain([FLAGGED]), audit)
    broker.subscribe("gatekeeper", lambda env, findings: None)
    with_ev = make_env({"text": "no figures"}, "12 ms")
    broker.publish(with_ev)
    broker.publish(make_env({"text": "none"}))
    first, second = audit.for_task("task_1")
    assert first["data"]["evidence_hashes"] == [with_ev.evidence[0].content_hash]
    assert "evidence_hashes" not in second["data"]
    assert audit.verify() == 2


def test_long_lists_of_missing_numbers_are_capped() -> None:
    body = " ".join(str(n) for n in range(100, 130))
    (finding,) = _inspect(make_env({"text": body}, "1"))
    assert finding.detail.endswith("and 10 more") and "100" in finding.detail
    assert "129" not in finding.detail


def test_non_finite_floats_and_unknown_types_are_ignored() -> None:
    assert "nan" not in payload_text({"a": float("nan"), "b": None}).lower().replace("\n", "")
    assert _inspect(make_env({"a": float("nan"), "b": None})) == []


def test_large_inputs_are_checked_without_a_quadratic_scan() -> None:
    body = " ".join(f"{n}.55" for n in range(20_000))
    evidence = " ".join(f"{n}.5512" for n in range(20_000))
    assert ungrounded_numbers(body, [evidence], GroundingSettings()) == []
