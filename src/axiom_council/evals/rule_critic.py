"""Deterministic rule critic: the baseline every model critic is compared with.

Each rule is a simple code check over the added and removed lines of one hunk. A rule
reports a fault class from the settings taxonomy and a new-file line range. The rules are
generic; none of them reads a case id, a title or an answer key.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from axiom_council.evals.critic_contracts import CriticCase, CriticKind, CriticOutput, Finding
from axiom_council.evals.unified_diff import DiffLine, Hunk, change_blocks, parse_diff

Severity = Literal["high", "medium", "low"]

_CONTAINMENT = re.compile(r"is_relative_to\(|commonpath\(")
_RAISE = re.compile(r"^\s*raise\b")
_BROAD_EXCEPT = re.compile(r"^\s*except\s*(?:\(?\s*(?:Exception|BaseException)\b[^:]*)?:\s*$")
_SQL = r"\b(?:SELECT|INSERT|UPDATE|DELETE|WHERE|ORDER BY|GROUP BY|FROM|VALUES)\b"
_SQL_FSTRING = re.compile(
    r"\bf([\"'])[^\"']*" + _SQL + r"[^\"']*\{|\bf([\"'])[^\"']*\{[^\"']*" + _SQL, re.I
)
_SQL_FORMAT = re.compile(_SQL + r".*[\"']\s*(?:\.format\(|%\s*[\w(])", re.I)
_SHELL = re.compile(r"shell\s*=\s*True|\bos\.system\(")
_TEST_PATH = re.compile(r"(?:^|/)tests?/|(?:^|/)test_[^/]*\.py$")
_STRICT_ASSERT = re.compile(r"^\s*assert\b.*(?:==|\bis\b|\bin\b)|pytest\.raises\(")
_OPEN_ASSIGN = re.compile(r"^\s*(\w+)\s*=\s*(?:[\w.]+\.)?open\(")
_UNIT_KW = re.compile(r"\b(milliseconds|seconds|minutes|hours)\s*=\s*([^,)]+)")
_UNIT_NAME = re.compile(r"\b\w+?_(ms|millis|s|sec|secs|seconds|min|mins|minutes|h|hours)\b")
_UNITS = {
    "ms": "milliseconds", "millis": "milliseconds", "s": "seconds", "sec": "seconds",
    "secs": "seconds", "seconds": "seconds", "min": "minutes", "mins": "minutes",
    "minutes": "minutes", "h": "hours", "hours": "hours",
}  # fmt: skip
_PLUS_MINUS_ONE = re.compile(r"[-+]\s*1\s*[\]\):]")
_COMPARISON = re.compile(r"[<>]=?")


@dataclass(frozen=True)
class Hit:
    fault_class: str
    severity: Severity
    start: int
    end: int
    claim: str
    evidence: str


Rule = Callable[[Hunk], list[Hit]]


def _span(block: tuple[DiffLine, ...]) -> tuple[int, int]:
    """New-file range of a change block; a pure removal spans the line before and after."""
    added = [line.new_line for line in block if line.op == "+"]
    if added:
        return min(added), max(added)
    anchor = block[0].new_line
    return max(anchor - 1, 1), anchor


def _hit(cls: str, sev: Severity, span: tuple[int, int], claim: str, text: str) -> Hit:
    return Hit(cls, sev, span[0], span[1], claim, text.strip()[:200])


def removed_containment(hunk: Hunk) -> list[Hit]:
    hits = []
    for block in change_blocks(hunk):
        removed = [ln for ln in block if ln.op == "-" and _CONTAINMENT.search(ln.text)]
        kept = any(ln.op == "+" and _CONTAINMENT.search(ln.text) for ln in hunk.lines)
        if removed and not kept:
            claim = "a path containment check is removed, so a name can escape its root"
            hits.append(_hit("path_traversal", "high", _span(block), claim, removed[0].text))
    return hits


def removed_raise(hunk: Hunk) -> list[Hit]:
    hits = []
    for block in change_blocks(hunk):
        if any(_CONTAINMENT.search(ln.text) for ln in block):
            continue  # removed_containment reports this block more precisely
        gone = [ln for ln in block if ln.op == "-" and _RAISE.match(ln.text)]
        added = [ln for ln in block if ln.op == "+" and _RAISE.match(ln.text)]
        if len(gone) > len(added):
            claim = "a guard that raised on bad input is removed"
            hits.append(_hit("missing_validation", "high", _span(block), claim, gone[0].text))
    return hits


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip())


def broad_except(hunk: Hunk) -> list[Hit]:
    new = [ln for ln in hunk.lines if ln.op != "-"]
    hits = []
    for index, line in enumerate(new):
        if line.op != "+" or not _BROAD_EXCEPT.match(line.text):
            continue
        body = []
        for follow in new[index + 1 :]:
            if follow.text.strip() and _indent(follow.text) <= _indent(line.text):
                break
            body.append(follow)
        if not any(_RAISE.match(b.text) for b in body):
            end = body[-1].new_line if body else line.new_line
            claim = "a broad except handler drops the error without re-raising"
            hits.append(_hit("swallowed_exception", "high", (line.new_line, end), claim, line.text))
    return hits


def interpolated_command(hunk: Hunk) -> list[Hit]:
    hits = []
    for line in hunk.lines:
        if line.op != "+":
            continue
        if _SQL_FSTRING.search(line.text) or _SQL_FORMAT.search(line.text):
            claim = "SQL text is built by string interpolation"
        elif _SHELL.search(line.text):
            claim = "a command runs through a shell"
        else:
            continue
        hits.append(_hit("injection", "high", (line.new_line, line.new_line), claim, line.text))
    return hits


def weakened_assertions(hunk: Hunk) -> list[Hit]:
    if not _TEST_PATH.search(hunk.path):
        return []
    hits = []
    for block in change_blocks(hunk):
        gone = sum(ln.op == "-" and bool(_STRICT_ASSERT.search(ln.text)) for ln in block)
        added = sum(ln.op == "+" and bool(_STRICT_ASSERT.search(ln.text)) for ln in block)
        if gone > added:
            claim = "exact assertions are replaced by weaker ones"
            hits.append(_hit("weakened_assertion", "medium", _span(block), claim, block[0].text))
    return hits


def open_without_with(hunk: Hunk) -> list[Hit]:
    hits = []
    new_text = "\n".join(ln.text for ln in hunk.lines if ln.op != "-")
    for line in hunk.lines:
        match = _OPEN_ASSIGN.match(line.text) if line.op == "+" else None
        if match and f"{match.group(1)}.close()" not in new_text:
            claim = "a file is opened outside a with block and never closed"
            span = (line.new_line, line.new_line)
            hits.append(_hit("resource_leak", "medium", span, claim, line.text))
    return hits


def _unit_mismatch(text: str) -> bool:
    for unit, expr in _UNIT_KW.findall(text):
        if any(_UNITS[suffix] != unit for suffix in _UNIT_NAME.findall(expr)):
            return True
    return False


def unit_mismatch(hunk: Hunk) -> list[Hit]:
    claim = "a value named in one time unit is passed as another unit"
    return [
        _hit("wrong_unit", "medium", (ln.new_line, ln.new_line), claim, ln.text)
        for ln in hunk.lines
        if ln.op == "+" and _unit_mismatch(ln.text)
    ]


def _bound_changed(old: str, new: str) -> bool:
    if _PLUS_MINUS_ONE.findall(new) != _PLUS_MINUS_ONE.findall(old):
        return len(_PLUS_MINUS_ONE.findall(new)) > len(_PLUS_MINUS_ONE.findall(old))
    same_shape = _COMPARISON.sub("<>", old) == _COMPARISON.sub("<>", new)
    return same_shape and _COMPARISON.findall(old) != _COMPARISON.findall(new)


def changed_bound(hunk: Hunk) -> list[Hit]:
    hits = []
    for block in change_blocks(hunk):
        removed = [ln.text for ln in block if ln.op == "-"]
        for line in (ln for ln in block if ln.op == "+"):
            ratios = [(SequenceMatcher(None, old, line.text).ratio(), old) for old in removed]
            best = max(ratios, default=(0.0, ""))
            if best[0] >= 0.6 and _bound_changed(best[1], line.text):
                claim = "a loop or slice bound moves by one"
                span = (line.new_line, line.new_line)
                hits.append(_hit("off_by_one", "medium", span, claim, line.text))
    return hits


_DEF = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(")


@dataclass(frozen=True)
class Signature:
    name: str
    params: dict[str, str | None]  # parameter name to default text; None means required
    start: int
    end: int
    touched: bool


def _close_paren(text: str) -> int | None:
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_top(text: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for char in text:
        depth += char in "([{"
        depth -= char in ")]}"
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    return [*parts, current]


def parameters(text: str) -> dict[str, str | None]:
    """Parameter names and default texts from a `def` line (or joined lines)."""
    opening = text.index("(")
    closing = _close_paren(text[opening:])
    assert closing is not None
    params: dict[str, str | None] = {}
    for part in _split_top(text[opening + 1 : opening + closing]):
        part = part.strip().lstrip("*")
        if not part or part == "/":
            continue
        head, equals, default = part.partition("=")
        params[head.split(":")[0].strip()] = default.strip() if equals else None
    return params


def signatures(lines: list[DiffLine]) -> dict[str, Signature]:
    """Function signatures in one side of a hunk; a signature may span several lines."""
    found: dict[str, Signature] = {}
    index = 0
    while index < len(lines):
        match = _DEF.match(lines[index].text)
        end, text = index, lines[index].text
        while match and _close_paren(text[text.index("(") :]) is None and end + 1 < len(lines):
            end += 1
            text += " " + lines[end].text.strip()
        if match and _close_paren(text[text.index("(") :]) is not None:
            span = lines[index : end + 1]
            found[match.group(1)] = Signature(
                match.group(1), parameters(text), span[0].new_line, span[-1].new_line,
                any(line.op != " " for line in span),
            )  # fmt: skip
        index = end + 1
    return found


def _public(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def _compare(before: Signature, after: Signature) -> list[Hit]:
    span, hits = (after.start, after.end), []
    for name, default in after.params.items():
        old = before.params.get(name)
        if old is not None and default is not None and old != default:
            claim = f"the default of {name} changes from {old} to {default}"
            hits.append(_hit("changed_default", "medium", span, claim, f"{name}={default}"))
    gone = before.params.keys() - after.params.keys() - {"self", "cls"}
    required = [p for p, d in after.params.items() if p not in before.params and d is None]
    if _public(after.name) and (gone or required):
        claim = f"public {after.name}() drops {sorted(gone)} or requires {required}"
        hits.append(_hit("api_break", "medium", span, claim, after.name))
    return hits


def signature_changes(hunk: Hunk) -> list[Hit]:
    old = signatures([ln for ln in hunk.lines if ln.op != "+"])
    new = signatures([ln for ln in hunk.lines if ln.op != "-"])
    hits = []
    for name, after in new.items():
        if name in old and (after.touched or old[name].touched):
            hits += _compare(old[name], after)
    for name, before in old.items():
        if name not in new and before.touched and _public(name):
            span = (max(before.start - 1, 1), before.start)
            hits.append(_hit("api_break", "medium", span, f"public {name}() is removed", name))
    return hits


RULES: tuple[Rule, ...] = (
    removed_containment, removed_raise, broad_except, interpolated_command,
    weakened_assertions, open_without_with, unit_mismatch, changed_bound, signature_changes,
)  # fmt: skip


class RuleCritic:
    """Runs every rule on every hunk. Deterministic: the same diff gives the same findings."""

    @property
    def critic_id(self) -> str:
        return "rules"

    @property
    def kind(self) -> CriticKind:
        return "rule"

    def review(self, case: CriticCase) -> CriticOutput:
        findings: list[Finding] = []
        for hunk in parse_diff(case.diff):
            for rule in RULES:
                for hit in rule(hunk):
                    findings.append(
                        Finding(
                            finding_id=f"r{len(findings) + 1}", severity=hit.severity,
                            fault_class=hit.fault_class, path=hunk.path, start_line=hit.start,
                            end_line=hit.end, claim=hit.claim, evidence=hit.evidence,
                        )
                    )  # fmt: skip
        return CriticOutput(findings=tuple(findings))
