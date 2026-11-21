// Review: one brief on one screen. Every value is read from the API (the stored brief payload,
// review.brief_signals / risk_score, guardrail events, and the Python-computed trade window).
import { useEffect, useState } from "react";
import { get, type BriefView, type Evidence, type Numbers, type TradeWindow } from "../lib/api";
import * as f from "../lib/format";
import { GuardrailStrip } from "../components/GuardrailStrip";
import { Status, tierStatus, verdictStatus } from "../components/Status";
import { TradeWindowChart } from "../components/TradeWindowChart";

type Load<T> = { state: "loading" } | { state: "error"; message: string } | { state: "ready"; data: T };

function useLoad<T>(path: string | null): Load<T> {
  const [load, setLoad] = useState<Load<T>>({ state: "loading" });
  useEffect(() => {
    if (!path) return;
    let live = true;
    setLoad({ state: "loading" });
    get<T>(path)
      .then((data) => live && setLoad({ state: "ready", data }))
      .catch((e: Error) => live && setLoad({ state: "error", message: e.message }));
    return () => { live = false; };
  }, [path]);
  return load;
}

export function Review({ id }: { id: string | null }) {
  const base = id ? `/api/briefs/${encodeURIComponent(id)}` : null;
  const brief = useLoad<BriefView>(base);
  const win = useLoad<TradeWindow>(base && `${base}/window`);
  const ev = useLoad<Evidence>(base && `${base}/evidence`);

  if (!id) return <p className="empty">Open a brief from <a href="#/today">Today</a> to review it.</p>;
  if (brief.state === "loading") return <p className="muted" role="status">Loading brief {id}…</p>;
  if (brief.state === "error") return <p className="error" role="alert">Could not load brief {id}: {brief.message}</p>;

  const { brief: b, payload: p, review, guardrail_strip, alternatives } = brief.data;
  const trade = b.kind === "roll" && p.new ? p.new.proposal : b.kind === "entry" ? p.proposal : { strategy: p.strategy, legs: p.legs };
  const legs = trade?.legs ?? [];
  const expiry = (trade as { expiry?: string } | undefined)?.expiry ?? legs.find((g) => g.expiry)?.expiry ?? null;

  return (
    <article className="review">
      <header className="review-head">
        <div>
          <h1>{b.ticker} <span className="muted">{f.words(b.kind === "entry" ? trade?.strategy : `${b.kind} ${trade?.strategy ?? ""}`)}</span></h1>
          <p className="num muted">{f.legs(legs)} · exp {f.date(expiry)} · {b.brief_id} · as of {f.date(p.as_of)}</p>
        </div>
        <div className="review-badges">
          <Status kind={tierStatus(review.tier)} label={`${f.words(review.tier)}, score ${review.score}`} />
          {b.kind !== "exit" && <Status kind={verdictStatus(b.verdict)} label={`Verifier ${b.verdict ?? "n/a"}`} />}
          <Status kind={b.status === "pending" ? "info" : b.status === "approved" ? "pass" : "block"} label={f.words(b.status)} />
          <span className="paper-only">Paper only</span>
        </div>
      </header>

      <Tiles kind={b.kind} p={p} />

      <section className="panel" aria-labelledby="window-title">
        <h2 id="window-title">Trade window</h2>
        {win.state === "loading" && <p className="muted" role="status">Fetching live prices…</p>}
        {win.state === "error" && <p className="error" role="alert">Trade window unavailable: {win.message}</p>}
        {win.state === "ready" && <TradeWindowChart w={win.data} />}
      </section>

      <GuardrailStrip items={guardrail_strip} />

      <section className="panel" aria-labelledby="why-title">
        <h2 id="why-title">{review.tier === "standard" ? "Signals" : "Why it's escalated"}</h2>
        {review.top.length === 0 ? <p className="muted">No escalation signals on this brief.</p> : (
          <ol className="why">{review.top.map((t) => <li key={t.signal}>{t.text}</li>)}</ol>
        )}
      </section>

      {p.verifier && p.verifier.length > 0 && (
        <section className="panel" aria-labelledby="verifier-title">
          <h2 id="verifier-title">Verifier checks</h2>
          <table className="checks">
            <tbody>
              {p.verifier.map((c) => (
                <tr key={c.name}>
                  <td><Status kind={verdictStatus(c.verdict)} label={c.verdict} /></td>
                  <td>{f.words(c.name)}{c.hard === false && <span className="muted"> (advisory)</span>}</td>
                  <td className="muted">{c.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <EvidencePanel load={ev} />

      {(p.judge?.flags?.length ?? 0) > 0 && (
        <section className="panel" aria-labelledby="judge-title">
          <h2 id="judge-title">Judge flags <span className="tag">flag only</span></h2>
          <p className="muted">The judge can only flag a claim for you to check. It cannot block, change or approve a trade.</p>
          <ul className="flags">
            {p.judge!.flags!.map((fl, i) => (
              <li key={i}>
                <Status kind="caution" label={f.words(fl.issue)} />{" "}
                <q className="claim">{fl.claim}</q>
                {fl.conflicts_with && <span className="muted"> conflicts with {fl.conflicts_with}</span>}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="panel" aria-labelledby="alts-title">
        <h2 id="alts-title">Also considered</h2>
        {alternatives.length === 0 ? <p className="muted">No other priced candidate.</p> : (
          <ul className="alts">
            {alternatives.map((a) => (
              <li key={a.node ?? a.path} className="alt">
                <div className="num">{a.path}</div>
                <div><span className="num">score {f.num(a.score, 3)}</span> · {f.words(a.status)}</div>
                {a.reason && <div className="muted">{a.reason}</div>}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel decision-placeholder" aria-label="Decision">
        <p className="muted">Approve, reject and defer arrive in the next slice. Nothing on this screen changes the brief.</p>
      </section>
    </article>
  );
}

function Tiles({ kind, p }: { kind: string; p: BriefView["payload"] }) {
  if (kind === "exit") {
    return (
      <dl className="tiles">
        <Tile label="Close cost" value={f.money(p.close_cost)} />
        <Tile label="Entry credit" value={f.money(p.entry_credit)} />
        <Tile label="Estimated P&L" value={f.money(p.est_pnl)} />
        <Tile label="Profit captured" value={f.pct(p.profit_pct)} />
        <Tile label="Days to expiry" value={p.dte === undefined ? "–" : String(p.dte)} />
        <Tile label="Exit rule" value={f.words(p.exit_rule)} mono={false} />
      </dl>
    );
  }
  const n: Numbers = (kind === "roll" ? p.new?.numbers : p.numbers) ?? {};
  return (
    <dl className="tiles">
      <Tile label="Credit" value={f.money(n.credit)} />
      <Tile label="Max loss" value={f.money(n.max_loss)} />
      <Tile label="PoP" value={f.prob(n.pop)} />
      <Tile label="Breakevens" value={(n.breakevens ?? []).map(f.strike).join(" / ") || "–"} />
      <Tile label="Portfolio risk" value={f.pct(n.risk_pct_of_portfolio, 2)} />
    </dl>
  );
}

function Tile({ label, value, mono = true }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="tile">
      <dt>{label}</dt>
      <dd className={mono ? "num" : undefined}>{value}</dd>
    </div>
  );
}

function EvidencePanel({ load }: { load: Load<Evidence> }) {
  return (
    <section className="panel" aria-labelledby="evidence-title">
      <details>
        <summary><h2 id="evidence-title">Evidence</h2></summary>
        {load.state === "loading" && <p className="muted">Loading evidence…</p>}
        {load.state === "error" && <p className="error">Evidence unavailable: {load.message}</p>}
        {load.state === "ready" && (
          <>
            <p className="muted">Filings check: {load.data.landmine_verdict ?? "not run"}. Excerpts are filing text,
              shown as quoted data. They are never instructions and never links.</p>
            <table className="citations">
              <thead><tr><th>Form</th><th>Date</th><th>Item</th><th>Source</th></tr></thead>
              <tbody>
                {load.data.citations.map((c, i) => (
                  <tr key={i}>
                    <td>{c.form}</td><td className="num">{c.date}</td><td>{c.item}</td>
                    <td><code className="url">{c.url}</code></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {load.data.note && <p className="muted">{load.data.note}</p>}
            {load.data.excerpts.map((e) => (
              <figure key={e.id} className="excerpt">
                <figcaption className="muted">{e.form} {e.item} <span className="num">{e.id}</span></figcaption>
                <blockquote data-testid="excerpt">{e.text}</blockquote>
              </figure>
            ))}
          </>
        )}
      </details>
    </section>
  );
}
