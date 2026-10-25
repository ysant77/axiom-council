from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from datetime import timedelta

import pytest
from pydantic import ValidationError

from axiom_council.contracts import (
    ApprovalToken,
    Capability,
    InferenceUsage,
    TaskClass,
    TaskSpec,
    content_hash,
    utcnow,
)


def test_unknown_fields_are_rejected(make_task: Callable[..., TaskSpec]) -> None:
    with pytest.raises(ValidationError):
        make_task(surprise="field")


def test_capability_cannot_be_both_allowed_and_forbidden(
    make_task: Callable[..., TaskSpec],
) -> None:
    with pytest.raises(ValidationError, match="both allowed and forbidden"):
        make_task(forbidden_capabilities=frozenset({Capability.LLM_INFERENCE}))


def test_network_capability_requires_network_flag(make_task: Callable[..., TaskSpec]) -> None:
    caps = frozenset({Capability.LLM_INFERENCE, Capability.NETWORK_EGRESS})
    with pytest.raises(ValidationError, match="network_allowed"):
        make_task(allowed_capabilities=caps)
    assert make_task(allowed_capabilities=caps, network_allowed=True).network_allowed


def test_code_patch_requires_pinned_commit(make_task: Callable[..., TaskSpec]) -> None:
    with pytest.raises(ValidationError, match="pin repo_ref"):
        make_task(task_class=TaskClass.CODE_PATCH)
    task = make_task(task_class=TaskClass.CODE_PATCH, repo_ref="main", base_commit_sha="abc1234")
    assert task.base_commit_sha == "abc1234"


def test_digest_is_stable_and_content_sensitive(make_task: Callable[..., TaskSpec]) -> None:
    task = make_task()
    assert task.digest() == task.digest()
    assert task.digest() != make_task(objective="Something else entirely").digest()
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_approval_token_is_bound_to_task_action_artifact_and_time() -> None:
    now = utcnow()
    token = ApprovalToken(
        task_id="t1",
        action="execute",
        artifact_hash="a" * 64,
        approver="director",
        expires_at=now + timedelta(minutes=5),
    )
    assert token.authorises("t1", "execute", "a" * 64, now)
    assert not token.authorises("t2", "execute", "a" * 64, now)
    assert not token.authorises("t1", "merge", "a" * 64, now)
    assert not token.authorises("t1", "execute", "b" * 64, now)
    assert not token.authorises("t1", "execute", "a" * 64, now + timedelta(minutes=6))


def test_usage_total() -> None:
    assert InferenceUsage(prompt_tokens=3, completion_tokens=4, wall_seconds=0.1).total_tokens == 7


_DIGEST_SCRIPT = """
import json, sys
from axiom_council.contracts import TaskSpec
spec = TaskSpec.model_validate_json(sys.stdin.read())
again = TaskSpec.model_validate(json.loads(spec.model_dump_json()))  # store round-trip
print(spec.digest(), again.digest())
"""


def test_digest_does_not_depend_on_hash_seed(make_task: Callable[..., TaskSpec]) -> None:
    """Regression: capability sets serialised in hash-seed order, so a multi-capability task
    could fail its own tamper check after being stored and reloaded."""
    task = make_task(
        workspace_id="council",
        allowed_capabilities=frozenset(
            {Capability.LLM_INFERENCE, Capability.PROPOSE_PATCH, Capability.RUN_SANDBOX}
        ),
        forbidden_capabilities=frozenset({Capability.MERGE, Capability.SPEND_MONEY}),
    )
    digests = set()
    for seed in range(8):
        out = subprocess.run(
            [sys.executable, "-c", _DIGEST_SCRIPT],
            input=task.model_dump_json(),
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        ).stdout.split()
        digests.update(out)
    assert digests == {task.digest()}
