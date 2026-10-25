from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom_council.audit import AuditLog, AuditTamperError


def _log(tmp_path: Path, n: int = 3) -> AuditLog:
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(n):
        log.append("event", f"t{i % 2}", "tester", i=i)
    return log


def test_chain_verifies_and_filters_by_task(tmp_path: Path) -> None:
    log = _log(tmp_path)
    assert log.verify() == 3
    assert [r["data"]["i"] for r in log.for_task("t0")] == [0, 2]


def test_empty_log_verifies(tmp_path: Path) -> None:
    assert AuditLog(tmp_path / "none.jsonl").verify() == 0


def _rewrite(path: Path, mutate: object) -> None:
    lines = path.read_text().splitlines()
    assert callable(mutate)
    path.write_text("\n".join(mutate(lines)) + "\n")


def test_edited_record_is_detected(tmp_path: Path) -> None:
    log = _log(tmp_path)

    def edit(lines: list[str]) -> list[str]:
        record = json.loads(lines[1])
        record["data"]["i"] = 99
        lines[1] = json.dumps(record, sort_keys=True)
        return lines

    _rewrite(log.path, edit)
    with pytest.raises(AuditTamperError, match="hash mismatch"):
        log.verify()


def test_deleted_record_is_detected(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _rewrite(log.path, lambda lines: [lines[0], lines[2]])
    with pytest.raises(AuditTamperError, match="broken chain"):
        log.verify()


def test_corrupt_line_is_detected(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _rewrite(log.path, lambda lines: [*lines, "{not json"])
    with pytest.raises(AuditTamperError, match="corrupt"):
        log.verify()
