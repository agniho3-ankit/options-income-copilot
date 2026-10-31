"""SQLite persistence at data/copilot.db: agent runs, masked traces, briefs, and order tickets.

Trace rows hold only masked content (see mask.py). Briefs and tickets hold the raw trade
numbers, which are synthetic in this build and never leave data/.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, ticker TEXT, started_at TEXT NOT NULL,
    finished_at TEXT, model TEXT, outcome TEXT, steps INTEGER DEFAULT 0, tool_calls INTEGER DEFAULT 0,
    bad_calls INTEGER DEFAULT 0, eval_tokens INTEGER DEFAULT 0, playbook_sha256 TEXT
);
CREATE TABLE IF NOT EXISTS trace (
    run_id TEXT NOT NULL REFERENCES runs(run_id), seq INTEGER NOT NULL, role TEXT NOT NULL,
    name TEXT, content TEXT NOT NULL, duration_ms INTEGER, eval_tokens INTEGER, created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS briefs (
    brief_id TEXT PRIMARY KEY, run_id TEXT, kind TEXT NOT NULL, ticker TEXT NOT NULL,
    status TEXT NOT NULL, verdict TEXT, created_at TEXT NOT NULL, decided_at TEXT,
    decision_reason TEXT, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY, brief_id TEXT NOT NULL REFERENCES briefs(brief_id),
    created_at TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%d}-{secrets.token_hex(3)}"


class Database:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # runs and traces
    def start_run(self, kind: str, ticker: str, model: str, playbook_sha256: str) -> str:
        run_id = new_id("run")
        with self.conn:
            self.conn.execute("INSERT INTO runs (run_id, kind, ticker, started_at, model, playbook_sha256) "
                              "VALUES (?, ?, ?, ?, ?, ?)", (run_id, kind, ticker, now(), model, playbook_sha256))
        return run_id

    def finish_run(self, run_id: str, outcome: str, **stats: int) -> None:
        cols = ["finished_at = ?", "outcome = ?"] + [f"{k} = ?" for k in stats]
        with self.conn:
            self.conn.execute(f"UPDATE runs SET {', '.join(cols)} WHERE run_id = ?",
                              (now(), outcome, *stats.values(), run_id))

    def add_trace(self, run_id: str, role: str, content: Any, name: str | None = None,
                  duration_ms: int | None = None, eval_tokens: int | None = None) -> None:
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        with self.conn:
            seq = self.conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM trace WHERE run_id = ?",
                                    (run_id,)).fetchone()[0]
            self.conn.execute("INSERT INTO trace VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                              (run_id, seq, role, name, text, duration_ms, eval_tokens, now()))

    def run(self, run_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def trace(self, run_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM trace WHERE run_id = ? ORDER BY seq", (run_id,))
        return [dict(r) for r in rows]

    # briefs and tickets
    def save_brief(self, brief_id: str, run_id: str | None, kind: str, ticker: str, status: str,
                   verdict: str | None, payload: dict) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO briefs (brief_id, run_id, kind, ticker, status, verdict, created_at, "
                              "payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                              (brief_id, run_id, kind, ticker, status, verdict, now(),
                               json.dumps(payload, default=str)))

    def brief(self, brief_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM briefs WHERE brief_id = ?", (brief_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["payload"] = json.loads(out["payload"])
        return out

    def briefs(self, status: str | None = None) -> list[dict]:
        q, args = "SELECT brief_id, kind, ticker, status, verdict, created_at, decided_at FROM briefs", ()
        if status:
            q, args = q + " WHERE status = ?", (status,)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY created_at DESC", args)]

    def decide(self, brief_id: str, status: str, reason: str) -> None:
        with self.conn:
            cur = self.conn.execute("UPDATE briefs SET status = ?, decided_at = ?, decision_reason = ? "
                                    "WHERE brief_id = ? AND status = 'pending'", (status, now(), reason, brief_id))
            if cur.rowcount != 1:
                raise ValueError(f"brief {brief_id} is not pending")

    def add_ticket(self, brief_id: str, payload: dict) -> str:
        ticket_id = new_id("tkt")
        with self.conn:
            self.conn.execute("INSERT INTO tickets VALUES (?, ?, ?, ?, ?)",
                              (ticket_id, brief_id, now(), "approved", json.dumps(payload, default=str)))
        return ticket_id

    def tickets(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM tickets ORDER BY created_at DESC")
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]
