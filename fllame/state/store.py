"""Tracks servers fllame started in the background (`fllame serve --detach`)
so `fllame status` and `fllame stop` can find them again. Ephemeral machine
state, not configuration - deliberately not a file an operator would
hand-edit or git-track (see CLAUDE.md, "Architecture").
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by someone else - still alive.
        return True
    return True


@dataclass(frozen=True)
class RunningServer:
    handle: str
    pid: int
    port: int
    started_at: str
    argv: list[str]

    def is_alive(self) -> bool:
        return _pid_alive(self.pid)


class StateStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS servers (
                    handle TEXT PRIMARY KEY,
                    pid INTEGER NOT NULL,
                    port INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    argv TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def record_started(self, handle: str, pid: int, port: int, argv: list[str]) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO servers (handle, pid, port, started_at, argv) "
                "VALUES (?, ?, ?, ?, ?)",
                (handle, pid, port, datetime.now(UTC).isoformat(), json.dumps(argv)),
            )
            conn.commit()

    def get(self, handle: str) -> RunningServer | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT handle, pid, port, started_at, argv FROM servers WHERE handle = ?",
                (handle,),
            ).fetchone()
        if row is None:
            return None
        return RunningServer(row[0], row[1], row[2], row[3], json.loads(row[4]))

    def list_all(self) -> list[RunningServer]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT handle, pid, port, started_at, argv FROM servers"
            ).fetchall()
        return [RunningServer(r[0], r[1], r[2], r[3], json.loads(r[4])) for r in rows]

    def remove(self, handle: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM servers WHERE handle = ?", (handle,))
            conn.commit()
