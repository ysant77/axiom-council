"""Audit integrity beyond the chain itself (defect D-06): anchors, locking and crash damage."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from axiom_council import cli
from axiom_council.audit import AuditLog, AuditTamperError, open_audit

from .conftest import CONFIG_DIR


def _log(tmp_path: Path, n: int = 3, *, db: bool = False) -> AuditLog:
    log = AuditLog(tmp_path / "audit.jsonl", anchor_db=tmp_path / "tasks.sqlite3" if db else None)
    for i in range(n):
        log.append("event", f"t{i % 2}", "tester", i=i)
    return log


def _anchor_file(log: AuditLog) -> Path:
    return log.path.with_name(log.path.name + ".anchor")


def _drop_last(path: Path) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))


def test_truncated_log_is_detected(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _drop_last(log.path)
    with pytest.raises(AuditTamperError, match="shorter than"):
        log.verify()


def test_replaced_log_is_detected(tmp_path: Path) -> None:
    log = _log(tmp_path)
    other = _log(tmp_path / "other", 3)
    shutil.copyfile(other.path, log.path)
    with pytest.raises(AuditTamperError, match="head"):
        log.verify()


def test_longer_replacement_is_detected(tmp_path: Path) -> None:
    log = _log(tmp_path)
    other = _log(tmp_path / "other", 5)
    shutil.copyfile(other.path, log.path)
    with pytest.raises(AuditTamperError, match="diverges"):
        log.verify()


def test_missing_anchor_with_records_fails(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _anchor_file(log).unlink()
    with pytest.raises(AuditTamperError, match="anchor missing"):
        log.verify()


def test_unreadable_anchor_fails_closed(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _anchor_file(log).write_text("{broken", encoding="utf-8")
    with pytest.raises(AuditTamperError, match="unreadable"):
        log.verify()


def test_database_anchor_catches_a_rewritten_anchor_file(tmp_path: Path) -> None:
    log = _log(tmp_path, db=True)
    before = _anchor_file(log).read_bytes()
    log.append("event", "t9", "tester")
    _drop_last(log.path)
    _anchor_file(log).write_bytes(before)  # the file anchor now agrees with the cut log
    with pytest.raises(AuditTamperError, match="database"):
        log.verify()


def test_deleted_log_is_detected_by_anchor(tmp_path: Path) -> None:
    log = _log(tmp_path, db=True)
    log.path.unlink()
    with pytest.raises(AuditTamperError, match="shorter than"):
        log.verify()


def test_append_refuses_to_launder_a_truncated_log(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _drop_last(log.path)
    cut = log.path.read_bytes()
    with pytest.raises(AuditTamperError, match="shorter than"):
        log.append("event", "t1", "tester")
    assert log.path.read_bytes() == cut


def test_open_audit_uses_both_anchors(tmp_path: Path) -> None:
    log = open_audit(tmp_path)
    log.append("event", "t1", "tester")
    assert log.verify() == 1
    assert (tmp_path / "tasks.sqlite3").exists()
    assert json.loads(_anchor_file(log).read_text())["count"] == 1


_WRITER = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from axiom_council.audit import AuditLog
    home, worker, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    log = AuditLog(home / "audit.jsonl", anchor_db=home / "tasks.sqlite3")
    for i in range(n):
        log.append("event", f"w{worker}", "writer", i=i)
    """
)


def test_concurrent_processes_keep_a_gapless_chain(tmp_path: Path) -> None:
    workers, each = 4, 25
    procs = [
        subprocess.Popen([sys.executable, "-c", _WRITER, str(tmp_path), str(w), str(each)])
        for w in range(workers)
    ]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    log = AuditLog(tmp_path / "audit.jsonl", anchor_db=tmp_path / "tasks.sqlite3")
    assert log.verify() == workers * each
    seqs = [json.loads(line)["seq"] for line in log.path.read_text().splitlines()]
    assert seqs == list(range(workers * each))


def test_partial_last_line_is_reported(tmp_path: Path) -> None:
    log = _log(tmp_path)
    with log.path.open("ab") as fh:
        fh.write(b'{"seq": 3, "event": "cut')
    with pytest.raises(AuditTamperError, match="partial last record"):
        log.verify()
    with pytest.raises(AuditTamperError, match="partial last record"):
        log.append("event", "t1", "tester")


def test_adopt_migrates_a_legacy_log_explicitly(tmp_path: Path) -> None:
    log = _log(tmp_path, db=True)
    _anchor_file(log).unlink()
    (tmp_path / "tasks.sqlite3").unlink()
    with pytest.raises(AuditTamperError, match="anchor missing"):
        log.append("event", "t1", "tester")
    record = log.adopt("director", "migrate the pre-anchor log")
    assert record["event"] == "audit_adopted"
    assert record["data"]["records"] == 3
    assert record["data"]["previous"] == {"file": None, "database": None}
    assert log.verify() == 4


def test_adopt_accepts_a_tail_written_before_a_crash(tmp_path: Path) -> None:
    log = _log(tmp_path)
    stale = _anchor_file(log).read_bytes()
    log.append("event", "t1", "tester")
    _anchor_file(log).write_bytes(stale)  # crash after the log write, before the anchor write
    with pytest.raises(AuditTamperError, match="past the file anchor"):
        log.verify()
    log.adopt("director", "crash between log and anchor")
    assert log.verify() == 5


def test_adopt_quarantines_a_partial_line(tmp_path: Path) -> None:
    log = _log(tmp_path)
    with log.path.open("ab") as fh:
        fh.write(b'{"seq": 3, "ev')
    record = log.adopt("director", "crash during the write")
    quarantine = Path(record["data"]["quarantined"])
    assert quarantine.read_bytes() == b'{"seq": 3, "ev'
    assert record["data"]["dropped_bytes"] == 14
    assert log.verify() == 4


@pytest.mark.parametrize("damage", ["truncate", "replace"])
def test_adopt_refuses_truncation_and_replacement(tmp_path: Path, damage: str) -> None:
    log = _log(tmp_path)
    if damage == "truncate":
        _drop_last(log.path)
    else:
        shutil.copyfile(_log(tmp_path / "other", 4).path, log.path)
    with pytest.raises(AuditTamperError, match="cannot adopt"):
        log.adopt("director", "please trust this")


def test_adopt_needs_a_reason_and_something_to_adopt(tmp_path: Path) -> None:
    log = _log(tmp_path)
    with pytest.raises(ValueError, match="reason"):
        log.adopt("director", "no")
    with pytest.raises(ValueError, match="nothing to adopt"):
        log.adopt("director", "all is well here")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("AXIOM_COUNCIL_HOME", str(home))
    monkeypatch.setenv("AXIOM_COUNCIL_CONFIG", str(CONFIG_DIR))
    return home


def test_cli_audit_reports_damage_and_adopts(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = open_audit(home)
    log.append("event", "t1", "tester")
    _anchor_file(log).unlink()
    assert cli.main(["audit"]) == 1
    assert "anchor missing" in capsys.readouterr().out
    assert cli.main(["audit", "adopt"]) == 2
    assert cli.main(["audit", "adopt", "--reason", "migrate legacy log"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["records"] == 2
    assert cli.main(["audit"]) == 0
    assert cli.main(["audit", "adopt", "--reason", "nothing changed here"]) == 2
    assert "nothing to adopt" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("last", "match"),
    [(b"[1, 2]\n", "corrupt last audit line"), (b"not json\n", "corrupt last audit line")],
)
def test_append_refuses_a_corrupt_last_line(tmp_path: Path, last: bytes, match: str) -> None:
    log = _log(tmp_path)
    with log.path.open("ab") as fh:
        fh.write(last)
    with pytest.raises(AuditTamperError, match=match):
        log.append("event", "t1", "tester")
    with pytest.raises(AuditTamperError, match="corrupt audit line 4"):
        log.verify()


def test_append_refuses_an_edited_last_record(tmp_path: Path) -> None:
    log = _log(tmp_path)
    lines = log.path.read_text().splitlines()
    record = json.loads(lines[-1])
    record["actor"] = "someone else"
    lines[-1] = json.dumps(record, sort_keys=True)
    log.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(AuditTamperError, match="hash mismatch at the last record"):
        log.append("event", "t1", "tester")


def test_reordered_sequence_is_detected(tmp_path: Path) -> None:
    from axiom_council.contracts import content_hash

    body = {"seq": 5, "ts": "x", "event": "e", "task_id": "t", "actor": "a", "data": {}}
    body["prev"] = "0" * 64
    (tmp_path / "audit.jsonl").write_text(json.dumps({**body, "hash": content_hash(body)}) + "\n")
    with pytest.raises(AuditTamperError, match="out-of-order"):
        AuditLog(tmp_path / "audit.jsonl").verify()


def test_long_records_are_read_from_the_tail(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(3):
        log.append("event", "t1", "tester", blob="x" * 10_000, i=i)
    assert log.append("event", "t1", "tester")["seq"] == 3
    assert log.verify() == 4
