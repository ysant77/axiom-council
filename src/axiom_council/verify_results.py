"""Offline checks for bake-off results files: each row must carry provenance that matches config.

A row passes when its provenance block has well-formed config and harness hashes, a stored config
version has that config hash, and the model card in that version pins the digest the row names.
A row without a provenance block predates provenance; it is reported as such, never as a pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from axiom_council.provenance import harness_hash
from axiom_council.verify_checks import Check, ConfigHistory, Status, failed, passed

NAME = "provenance"
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _against_config(
    subject: str, prov: dict[str, Any], row: dict[str, Any], configs: ConfigHistory
) -> Check:
    digest = prov["config_hash"]
    registry = configs.by_config_hash(digest)
    if registry is None:
        reason = f"no config version in the store has config hash {digest[:12]}"
        return failed(NAME, subject, reason, Status.CANNOT_RESOLVE)
    if harness_hash(registry) != prov["harness_hash"]:
        return failed(NAME, subject, f"harness hash differs from config {digest[:12]}")
    model_id = str(prov.get("model_id") or row.get("model_id"))
    card = registry.models.get(model_id)
    if card is None:
        reason = f"model {model_id} has no card in config {digest[:12]}"
        return failed(NAME, subject, reason, Status.CANNOT_RESOLVE)
    if card.digest is None:
        reason = f"the model card for {model_id} pins no digest in config {digest[:12]}"
        return failed(NAME, subject, reason, Status.CANNOT_RESOLVE)
    if prov.get("model_digest") != card.digest:
        reason = (
            f"row digest {str(prov.get('model_digest'))[:12]} differs from the digest"
            f" {card.digest[:12]} pinned on the card for {model_id}"
        )
        return failed(NAME, subject, reason)
    return passed(
        NAME, subject, f"{model_id} digest {card.digest[:12]} matches config {digest[:12]}"
    )


def _row_check(subject: str, line: str, configs: ConfigHistory, allow_legacy: bool) -> Check:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return failed(NAME, subject, "row is not valid JSON")
    if not isinstance(row, dict):
        return failed(NAME, subject, "row is not a JSON object")
    prov = row.get("provenance")
    if prov is None:
        reason = "row has no provenance block (written before provenance was recorded)"
        return Check(NAME, subject, Status.NO_PROVENANCE, reason, blocking=not allow_legacy)
    if not isinstance(prov, dict):
        return failed(NAME, subject, "provenance block is not an object")
    bad = [k for k in ("config_hash", "harness_hash") if not _HASH.match(str(prov.get(k)))]
    if bad:
        return failed(NAME, subject, f"malformed {', '.join(bad)} (want 64 lowercase hex)")
    return _against_config(subject, prov, row, configs)


def results_checks(path: Path, configs: ConfigHistory, *, allow_legacy: bool) -> list[Check]:
    """One check per row of one results file."""
    if not path.is_file():
        return [failed(NAME, str(path), "results file not found")]
    lines = path.read_text(encoding="utf-8").splitlines()
    checks = [
        _row_check(f"{path.name}:{n}", line, configs, allow_legacy)
        for n, line in enumerate(lines, start=1)
        if line.strip()
    ]
    return checks or [failed(NAME, str(path), "results file has no rows")]
