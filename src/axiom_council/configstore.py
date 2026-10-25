"""Versioned configuration store (SQLite) for models, nodes, roles, workspaces and prompts.

Every write that changes content creates a new version and deactivates the previous one, so
the exact config behind any past run can be recovered by hash. YAML files are only seeds.
The schema is plain SQL so it can move to Postgres without changing callers.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from axiom_council.contracts import content_hash, utcnow

KINDS: dict[str, tuple[str, str]] = {
    # kind: (seed file, key field)
    "model": ("models.yaml", "model_id"),
    "node": ("nodes.yaml", "node_id"),
    "role": ("roles.yaml", "name"),
    "workspace": ("workspaces.yaml", "workspace_id"),
    "prompt": ("prompts.yaml", "prompt_id"),
    "route": ("routes.yaml", "route_id"),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS config_entries (
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    version INTEGER NOT NULL,
    body TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0, 1)),
    PRIMARY KEY (kind, key, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_config_active
    ON config_entries (kind, key) WHERE active = 1;
"""


@dataclass(frozen=True)
class ConfigEntry:
    kind: str
    key: str
    version: int
    body: dict[str, Any]
    sha256: str


class ConfigStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    @staticmethod
    def _check_kind(kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown config kind {kind!r}; expected one of {sorted(KINDS)}")

    def put(self, kind: str, key: str, body: dict[str, Any]) -> ConfigEntry:
        """Store body as the active version; a no-op if the content is unchanged."""
        self._check_kind(kind)
        current = self.get(kind, key)
        digest = content_hash(body)
        if current and current.sha256 == digest:
            return current
        version = (current.version + 1) if current else 1
        with self._db:
            self._db.execute(
                "UPDATE config_entries SET active = 0 WHERE kind = ? AND key = ?", (kind, key)
            )
            self._db.execute(
                "INSERT INTO config_entries VALUES (?, ?, ?, ?, ?, ?, 1)",
                (
                    kind,
                    key,
                    version,
                    json.dumps(body, sort_keys=True),
                    digest,
                    utcnow().isoformat(),
                ),
            )
        return ConfigEntry(kind, key, version, body, digest)

    def get(self, kind: str, key: str) -> ConfigEntry | None:
        self._check_kind(kind)
        row = self._db.execute(
            "SELECT version, body, sha256 FROM config_entries"
            " WHERE kind = ? AND key = ? AND active = 1",
            (kind, key),
        ).fetchone()
        return ConfigEntry(kind, key, row[0], json.loads(row[1]), row[2]) if row else None

    def active(self, kind: str) -> dict[str, dict[str, Any]]:
        self._check_kind(kind)
        rows = self._db.execute(
            "SELECT key, body FROM config_entries WHERE kind = ? AND active = 1 ORDER BY key",
            (kind,),
        ).fetchall()
        return {key: json.loads(body) for key, body in rows}

    def history(self, kind: str, key: str) -> list[ConfigEntry]:
        self._check_kind(kind)
        rows = self._db.execute(
            "SELECT version, body, sha256 FROM config_entries"
            " WHERE kind = ? AND key = ? ORDER BY version",
            (kind, key),
        ).fetchall()
        return [ConfigEntry(kind, key, v, json.loads(b), h) for v, b, h in rows]

    def is_empty(self) -> bool:
        count: int = self._db.execute("SELECT COUNT(*) FROM config_entries").fetchone()[0]
        return count == 0

    def seed(self, config_dir: Path) -> dict[str, int]:
        """Import every seed YAML present in config_dir; returns entries written per kind."""
        written: dict[str, int] = {}
        for kind, (filename, key_field) in KINDS.items():
            path = config_dir / filename
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                items = (yaml.safe_load(fh) or {}).get(f"{kind}s", [])
            for item in items:
                self.put(kind, str(item[key_field]), item)
            written[kind] = len(items)
        return written
