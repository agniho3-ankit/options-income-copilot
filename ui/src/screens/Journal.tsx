// Journal: the Skill #7 trade log and its monthly summary. Every number comes from journal.py
// (the stored rows and journal.metrics); this screen only formats them.
import { useEffect, useState } from "react";
import { get, type Journal as JournalData, type JournalSummary } from "../lib/api";
import * as f from "../lib/format";

export function Journal() {
  const [data, setData] = useState<JournalData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { get<JournalData>("/api/journal").then(setData).catch((e: Error) => setError(e.message)); }, []);
  if (error) return <p className="error" role="alert">Could not load the journal: {error}</p>;
  if (!data) return <p className="muted" role="status">Loading the journal…</p>;
  return (
    <div className="journal">
      <h1 className="visually-hidden">Journal</h1>
      <p className="muted">Synthetic paper positions only. P&amp;L percent is measured against the entry credit; return on
        risk is P&amp;L against the max loss. Both come from journal.py.</p>
      <section className="panel" aria-labelledby="summary-title">
        <h2 id="summary-title">Summary</h2>
        <div className="summaries">
          <SummaryCard s={data.all} title="All time" />
          {[...data.months].reverse().map((m) => <SummaryCard key={m.month} s={m} title={m.month} />)}
        </div>
      </section>
      <section className="panel" aria-labelledby="log-title">
        <h2 id="log-title">Trade log</h2>
        {data.rows.length === 0 ? <p className="muted">No trades yet.</p> : (
          <div className="table-scroll">
            <table className="log">
              <thead>
                <tr>
                  <th>Opened</th><th>Ticker</th><th>Strategy</th><th>Strikes</th><th>DTE</th><th>Credit</th>
                  <th>Max loss</th><th>Earnings</th><th>Closed</th><th>Close</th><th>P&amp;L</th><th>P&amp;L %</th>
                  <th>Return on risk</th><th>Days</th><th>Exit rule</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((r) => (
                  <tr key={r.position_id}>
                    <td className="num">{r.date_opened}</td>
                    <td>{r.ticker}</td>
                    <td>{f.words(r.strategy)}</td>
                    <td className="num">{[r.long_put, r.short_put, r.short_call, r.long_call].filter((x) => x !== null).join(" / ")}</td>
                    <td className="num">{r.dte ?? "–"}</td>
                    <td className="num">{f.money(r.entry_credit)}</td>
                    <td className="num">{f.money(r.entry_max_loss)}</td>
                    <td>{r.earnings_note ?? "–"}</td>
                    <td className="num">{r.date_closed ?? "open"}</td>
                    <td className="num">{f.money(r.close_cost)}</td>
                    <td className="num">{f.money(r.pnl)}</td>
                    <td className="num">{f.pct(r.pnl_pct)}</td>
                    <td className="num">{f.pct(r.return_on_risk_pct)}</td>
                    <td className="num">{r.days_held ?? "–"}</td>
                    <td>{r.exit_rule ? f.words(r.exit_rule) : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function SummaryCard({ s, title }: { s: JournalSummary; title: string }) {
  const rules = Object.entries(s.exit_rules ?? {});
  return (
    <article className="summary-card" aria-label={`Summary ${title}`}>
      <h3 className="num">{title}</h3>
      <dl>
        <dt>P&amp;L</dt><dd className="num">{f.money(s.total_pnl)}</dd>
        <dt>Trades</dt><dd className="num">{s.closed_trades} closed · {s.active_trades} open</dd>
        <dt>Win rate</dt><dd className="num">{f.pct(s.win_rate_pct)}</dd>
        <dt>Return on risk</dt><dd className="num">{f.pct(s.avg_return_on_risk_pct, 2)}</dd>
        <dt>Avg return</dt><dd className="num">{f.pct(s.avg_return_pct, 2)} of credit</dd>
        <dt>Avg days held</dt><dd className="num">{f.num(s.avg_days_held, 1)}</dd>
        <dt>Exit rules</dt>
        <dd>{rules.length === 0 ? "–" : rules.map(([k, n]) => <span key={k} className="rule-chip">{f.words(k)} <span className="num">{n}</span></span>)}</dd>
      </dl>
    </article>
  );
}
