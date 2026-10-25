"""Code fixtures: build the refactor prompt, then verify the answer in the sandbox.

Verification copies a pristine snapshot, splices the model's functions into the module,
applies it through the sandbox path policy and runs the allowlisted checks with no network.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from axiom_council.bus.spotlight import fence
from axiom_council.evals.fixtures import CodeFixture
from axiom_council.evals.scoring import Score
from axiom_council.evals.splice import SpliceError, extract_code, splice
from axiom_council.registry import PromptTemplate
from axiom_council.sandbox import DockerSandbox, EditRejected, FileEdit, Snapshot, apply_edits

CODE_CHECKS = ("lint", "tests")


def code_messages(
    prompt: PromptTemplate, fixture: CodeFixture, snapshot: Snapshot
) -> list[dict[str, str]]:
    source = (snapshot.path / fixture.module).read_text(encoding="utf-8")
    return prompt.render(
        module=fence(source, fixture.module),
        function=fixture.function,
        max_lines=str(fixture.max_lines),
    )


def verify_code(
    fixture: CodeFixture,
    output: str,
    base: Snapshot,
    sandbox: DockerSandbox,
    work_root: Path,
    run_id: str,
) -> Score:
    source = (base.path / fixture.module).read_text(encoding="utf-8")
    try:
        new_source = splice(source, fixture.function, extract_code(output), fixture.max_lines)
    except SpliceError as exc:
        return Score(False, {"splice": False}, str(exc))

    copy = Snapshot(work_root / run_id, base.commit)
    if copy.path.exists():
        shutil.rmtree(copy.path)
    shutil.copytree(base.path, copy.path, symlinks=True)
    try:
        edit = FileEdit(path=fixture.module, action="replace", content=new_source)
        apply_edits(copy, [edit], sandbox.profile)
    except EditRejected as exc:
        return Score(False, {"splice": True, "edit_policy": False}, str(exc))

    checks: dict[str, bool] = {"splice": True, "edit_policy": True}
    note = ""
    for name in CODE_CHECKS:
        result = sandbox.run_check(copy, name)
        checks[name] = result.passed
        if not result.passed and not note:
            note = result.output_tail[-600:]
    shutil.rmtree(copy.path, ignore_errors=True)
    return Score(all(checks.values()), checks, note)
