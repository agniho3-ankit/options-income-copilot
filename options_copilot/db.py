"""SQLite persistence at data/copilot.db: agent runs, masked traces, Tree-of-Thought nodes,
briefs, order tickets, synthetic paper positions, the trade journal, and IV snapshots.

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

from .guardrails import forbid_in_sandbox

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
CREATE TABLE IF NOT EXISTS tot_nodes (
    run_id TEXT NOT NULL REFERENCES runs(run_id), node_id TEXT NOT NULL, parent_id TEXT,
    depth INTEGER NOT NULL, status TEXT NOT NULL, score REAL, prune_reason TEXT, source TEXT,
    node_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY (run_id, node_id)
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
CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY, ticket_id TEXT, ticker TEXT NOT NULL, status TEXT NOT NULL,
    opened_on TEXT NOT NULL, closed_on TEXT, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS journal (
    position_id TEXT PRIMARY KEY REFERENCES positions(position_id), date_opened TEXT NOT NULL,
    ticker TEXT NOT NULL, strategy TEXT NOT NULL, short_put REAL, short_call REAL, long_put REAL,
    long_call REAL, dte INTEGER, entry_credit REAL, entry_max_loss REAL, earnings_note TEXT,
    date_closed TEXT, close_cost REAL, pnl REAL, pnl_pct REAL, return_on_risk_pct REAL,
    days_held INTEGER, annualized_pct REAL, exit_rule TEXT, notes TEXT, synthetic INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS spans (
    run_id TEXT NOT NULL REFERENCES runs(run_id), span_id TEXT NOT NULL, parent_id TEXT, agent TEXT NOT NULL,
    pattern TEXT, started_at TEXT NOT NULL, ended_at TEXT, duration_ms INTEGER, llm_calls INTEGER DEFAULT 0,
    input_summary TEXT, output_summary TEXT, error TEXT, PRIMARY KEY (run_id, span_id)
);
CREATE TABLE IF NOT EXISTS guardrail_events (
    run_id TEXT NOT NULL REFERENCES runs(run_id), seq INTEGER NOT NULL, span_id TEXT, guardrail TEXT NOT NULL,
    stage TEXT NOT NULL, kind TEXT NOT NULL, decision TEXT NOT NULL, signal TEXT, reason TEXT, data TEXT,
    created_at TEXT NOT NULL, PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS evals (
    eval_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, mode TEXT NOT NULL, summary TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS iv_history (
    ticker TEXT NOT NULL, as_of TEXT NOT NULL, atm_iv REAL NOT NULL, expiry TEXT, dte INTEGER,
    created_at TEXT NOT NULL, PRIMARY KEY (ticker, as_of)
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
        # check_same_thread=False: the in-process MCP bridge runs tools on its event-loop thread;
        # calls are strictly sequential (the caller waits on each one), so access never overlaps.
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()

    # columns added after a table first shipped (ALTER TABLE keeps existing rows)
    MIGRATIONS = {
        "spans": {"tokens": "INTEGER DEFAULT 0", "fallbacks": "INTEGER DEFAULT 0", "decision": "TEXT",
                  "confidence": "REAL", "ticker": "TEXT", "as_of": "TEXT", "citations": "TEXT"},
        "briefs": {"decision_code": "TEXT", "deferred_at": "TEXT"},
    }

    def _migrate(self) -> None:
        with self.conn:
            for table, cols in self.MIGRATIONS.items():
                have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
                for col, decl in cols.items():
                    if col not in have:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

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

    # Tree-of-Thought nodes (one row per node; node_json is the full serialized node)
    def save_tot_nodes(self, run_id: str, nodes: list[dict]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM tot_nodes WHERE run_id = ?", (run_id,))
            self.conn.executemany(
                "INSERT INTO tot_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(run_id, n["id"], n["parent_id"], n["depth"], n["status"], n["score"], n["prune_reason"],
                  n["source"], json.dumps(n, default=str), now()) for n in nodes])

    def tot_nodes(self, run_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT node_json FROM tot_nodes WHERE run_id = ? ORDER BY depth, rowid", (run_id,))
        return [json.loads(r[0]) for r in rows]

    # briefs and tickets
    def save_brief(self, brief_id: str, run_id: str | None, kind: str, ticker: str, status: str,
                   verdict: str | None, payload: dict) -> None:
        forbid_in_sandbox("writing a brief or ticket")
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
        q, args = ("SELECT brief_id, run_id, kind, ticker, status, verdict, created_at, decided_at, decision_code, "
                   "deferred_at FROM briefs"), ()
        if status:
            q, args = q + " WHERE status = ?", (status,)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY created_at DESC", args)]

    def decide(self, brief_id: str, status: str, reason: str, code: str | None = None) -> None:
        with self.conn:
            cur = self.conn.execute("UPDATE briefs SET status = ?, decided_at = ?, decision_reason = ?, "
                                    "decision_code = ? WHERE brief_id = ? AND status = 'pending'",
                                    (status, now(), reason, code, brief_id))
            if cur.rowcount != 1:
                raise ValueError(f"brief {brief_id} is not pending")

    def defer(self, brief_id: str) -> None:
        """Keep a brief pending but move it to the end of the review queue (no decision is made)."""
        with self.conn:
            cur = self.conn.execute("UPDATE briefs SET deferred_at = ? WHERE brief_id = ? AND status = 'pending'",
                                    (now(), brief_id))
            if cur.rowcount != 1:
                raise ValueError(f"brief {brief_id} is not pending")

    def add_ticket(self, brief_id: str, payload: dict) -> str:
        forbid_in_sandbox("writing a brief or ticket")
        ticket_id = new_id("tkt")
        with self.conn:
            self.conn.execute("INSERT INTO tickets VALUES (?, ?, ?, ?, ?)",
                              (ticket_id, brief_id, now(), "approved", json.dumps(payload, default=str)))
        return ticket_id

    def tickets(self, status: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM tickets", ()
        if status:
            q, args = q + " WHERE status = ?", (status,)
        rows = self.conn.execute(q + " ORDER BY created_at, rowid", args)
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def ticket(self, ticket_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        return {**dict(row), "payload": json.loads(row["payload"])} if row else None

    def update_ticket(self, ticket_id: str, status: str, **updates: Any) -> None:
        t = self.ticket(ticket_id)
        if t is None:
            raise ValueError(f"no ticket {ticket_id}")
        with self.conn:
            self.conn.execute("UPDATE tickets SET status = ?, payload = ? WHERE ticket_id = ?",
                              (status, json.dumps({**t["payload"], **updates}, default=str), ticket_id))

    # synthetic paper positions and the trade journal (playbook Skill #7)
    def save_position(self, position_id: str, ticket_id: str | None, ticker: str, status: str, opened_on: str,
                      payload: dict, closed_on: str | None = None) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (position_id, ticket_id, ticker, status, opened_on, closed_on,
                               json.dumps(payload, default=str)))

    def positions(self, status: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM positions", ()
        if status:
            q, args = q + " WHERE status = ?", (status,)
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in self.conn.execute(q + " ORDER BY opened_on", args)]

    def position(self, position_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM positions WHERE position_id = ?", (position_id,)).fetchone()
        return {**dict(row), "payload": json.loads(row["payload"])} if row else None

    JOURNAL_COLS = ("position_id", "date_opened", "ticker", "strategy", "short_put", "short_call", "long_put",
                    "long_call", "dte", "entry_credit", "entry_max_loss", "earnings_note", "date_closed",
                    "close_cost", "pnl", "pnl_pct", "return_on_risk_pct", "days_held", "annualized_pct",
                    "exit_rule", "notes", "synthetic")

    def save_journal(self, row: dict) -> None:
        cols = [c for c in self.JOURNAL_COLS if c in row]
        with self.conn:
            self.conn.execute(f"INSERT OR REPLACE INTO journal ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                              [row[c] for c in cols])

    def journal(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM journal ORDER BY date_opened, position_id")]

    # spans: one per node execution of a multi-agent run (summaries are masked by the caller)
    def add_span(self, run_id: str, span_id: str, parent_id: str | None, agent: str, pattern: str | None,
                 started_at: str, ended_at: str | None, duration_ms: int | None, llm_calls: int, input_summary: Any,
                 output_summary: Any, error: str | None = None, *, tokens: int = 0, fallbacks: int = 0,
                 decision: str | None = None, confidence: float | None = None, ticker: str | None = None,
                 as_of: str | None = None, citations: Any = None) -> None:
        """Operational (duration, LLM calls, tokens, fallbacks, error), cognitive (decision,
        confidence; guardrail events link by span_id) and contextual (ticker, as_of, masked
        summaries, citations) fields of one span."""
        enc = lambda v: v if v is None or isinstance(v, str) else json.dumps(v, default=str)  # noqa: E731
        with self.conn:
            self.conn.execute(
                "INSERT INTO spans (run_id, span_id, parent_id, agent, pattern, started_at, ended_at, duration_ms, "
                "llm_calls, input_summary, output_summary, error, tokens, fallbacks, decision, confidence, ticker, "
                "as_of, citations) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id, span_id) DO UPDATE SET parent_id = excluded.parent_id, agent = excluded.agent, "
                "pattern = excluded.pattern, started_at = excluded.started_at, ended_at = excluded.ended_at, "
                "duration_ms = excluded.duration_ms, llm_calls = excluded.llm_calls, "
                "input_summary = excluded.input_summary, output_summary = excluded.output_summary, "
                "error = excluded.error, tokens = excluded.tokens, fallbacks = excluded.fallbacks, "
                "decision = excluded.decision, confidence = excluded.confidence, ticker = excluded.ticker, "
                "as_of = excluded.as_of, citations = excluded.citations",
                (run_id, span_id, parent_id, agent, pattern, started_at, ended_at, duration_ms, llm_calls,
                 enc(input_summary), enc(output_summary), error, tokens, fallbacks, decision, confidence,
                 ticker, as_of, enc(citations)))

    # guardrail decisions (one row per decision, linked to the span that was running)
    def add_guardrail_event(self, run_id: str, seq: int, span_id: str | None, guardrail: str, stage: str, kind: str,
                            decision: str, signal: str | None, reason: str, data: Any = None) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO guardrail_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                              (run_id, seq, span_id, guardrail, stage, kind, decision, signal, reason,
                               None if data is None else json.dumps(data, default=str), now()))

    def guardrail_events(self, run_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM guardrail_events WHERE run_id = ? ORDER BY seq", (run_id,))
        return [dict(r) for r in rows]

    # evaluation summaries (the eval harness writes one row per eval or ablation run)
    def add_eval(self, mode: str, summary: dict) -> str:
        eval_id = new_id("eval")
        with self.conn:
            self.conn.execute("INSERT INTO evals VALUES (?, ?, ?, ?)",
                              (eval_id, now(), mode, json.dumps(summary, default=str)))
        return eval_id

    def latest_eval(self, mode: str = "eval") -> dict | None:
        row = self.conn.execute("SELECT * FROM evals WHERE mode = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                                (mode,)).fetchone()
        return {**dict(row), "summary": json.loads(row["summary"])} if row else None

    def spans(self, run_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM spans WHERE run_id = ? ORDER BY started_at, rowid", (run_id,))
        return [dict(r) for r in rows]

    # daily IV snapshots (one per ticker per day; a re-run the same day replaces it)
    def add_iv_snapshot(self, ticker: str, as_of: str, atm_iv: float, expiry: str | None, dte: int | None) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO iv_history VALUES (?, ?, ?, ?, ?, ?)",
                              (ticker, as_of, atm_iv, expiry, dte, now()))

    def iv_history(self, ticker: str, before: str | None = None) -> list[tuple[str, float]]:
        """(as_of, atm_iv) oldest first, optionally only days strictly before `before`."""
        q, args = "SELECT as_of, atm_iv FROM iv_history WHERE ticker = ?", [ticker]
        if before:
            q += " AND as_of < ?"
            args.append(before)
        return [(r[0], r[1]) for r in self.conn.execute(q + " ORDER BY as_of", args)]
