from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from axiom_council.contracts import Clarification, utcnow
from axiom_council.inbox import Inbox

HASH = "c" * 64


def test_approval_flow_issues_bound_token(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path)
    item = inbox.request_approval(task_id="t1", action="execute", artifact_hash=HASH, summary="s")
    assert [i.item_id for i in inbox.pending()] == [item.item_id]
    token = inbox.approve(item.item_id, "director", timedelta(hours=1), "looks safe")
    assert token.authorises("t1", "execute", HASH, utcnow())
    assert inbox.pending() == []
    assert inbox.get(item.item_id).status == "approved"
    with pytest.raises(ValueError, match="not a pending approval"):
        inbox.approve(item.item_id, "director", timedelta(hours=1), "again please")


def test_decisions_need_a_reason(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path)
    item = inbox.request_approval(task_id="t1", action="execute", artifact_hash=HASH, summary="s")
    with pytest.raises(ValueError, match="reason"):
        inbox.approve(item.item_id, "director", timedelta(hours=1), " ok ")
    with pytest.raises(ValueError, match="reason"):
        inbox.reject(item.item_id, "")
    inbox.reject(item.item_id, "out of scope")
    assert inbox.get(item.item_id).status == "rejected"
    with pytest.raises(ValueError, match="not pending"):
        inbox.reject(item.item_id, "out of scope")


def test_clarification_flow(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path)
    clar = Clarification(task_id="t2", asked_by="worker", question="Which paper?")
    item = inbox.ask(clar)
    assert item.kind == "clarification"
    inbox.answer(item.item_id, "Attention Is All You Need")
    assert inbox.get(item.item_id).response == "Attention Is All You Need"
    with pytest.raises(ValueError, match="not a pending clarification"):
        inbox.answer(item.item_id, "again")


def test_path_traversal_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid inbox id"):
        Inbox(tmp_path).get("../../etc/passwd")
