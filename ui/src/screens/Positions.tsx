// Positions: open paper positions as rule tracks, plus portfolio heat against the cap. Every value
// and every bar length comes from the API (management.rule_tracks, management.heat_gauge).
// Read-only, apart from "Run monitor", which starts a background job; its exit and roll briefs
// appear on Today.
import { useCallback, useEffect, useState } from "react";
import { get, post, type Job, type Positions as PositionsData, type PositionRow, type Track } from "../lib/api";
import * as f from "../lib/format";
import { Status, type StatusKind } from "../components/Status";

const STATE: Record<Track["state"], { kind: StatusKind; label: string }> = {
  ok: { kind: "pass", label: "OK" },
  due: { kind: "caution", label: "Action due" },
  breach: { kind: "block", label: "Breached" },
  unknown: { kind: "info", label: "Not evaluated" },
};

export function Positions() {
  const [data, setData] = useState<PositionsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => {
    get<PositionsData>("/api/positions").then(setData).catch((e: Error) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  if (error) return <p className="error" role="alert">Could not load positions: {error}</p>;
  if (!data) return <p className="muted" role="status">Fetching live prices for open positions…</p>;
  const h = data.heat;
  return (
    <div className="positions">
      <h1 className="visually-hidden">Positions</h1>
      <section className="panel" aria-labelledby="heat-title">
        <h2 id="heat-title">Portfolio heat</h2>
        <div className="gauge" role="meter" aria-valuemin={0} aria-valuemax={h.cap_pct} aria-valuenow={h.pct}
          aria-label={`Portfolio heat ${f.pct(h.pct)} of a ${f.pct(h.cap_pct, 0)} cap`}>
          <div className={`gauge-fill${h.over_cap ? " gauge-over" : ""}`} style={{ width: `${h.gauge_pct}%` }} />
        </div>
        <p className="muted"><span className="num">{f.pct(h.pct, 2)}</span> of the <span className="num">{f.pct(h.cap_pct, 0)}</span> cap
          (warning above <span className="num">{f.pct(h.warn_pct, 0)}</span>) · <span className="num">{data.positions.length}</span> open
          {h.review && <> · <Status kind="caution" label={h.review} /></>}</p>
      </section>
      <MonitorPanel onFinished={load} />
      {data.positions.length === 0 ? <p className="empty">No open paper positions.</p> : (
        data.positions.map((p) => <PositionCard key={p.position_id} p={p} />)
      )}
    </div>
  );
}

function PositionCard({ p }: { p: PositionRow }) {
  return (
    <section className="panel position" aria-label={`${p.ticker} ${f.words(p.strategy)}`}>
      <header className="position-head">
        <div>
          <h2 className="position-title">{p.ticker} <span className="muted">{f.words(p.strategy)}</span></h2>
          <p className="num muted">{f.legs(p.legs)} · exp {p.expiry} · opened {p.opened_on} · {p.position_id}</p>
        </div>
        <div className="position-figures num">
          <span>credit {f.money(p.entry_credit)}</span>
          <span>close cost {f.money(p.close_cost)}</span>
          <span>spot {f.num(p.spot)}</span>
        </div>
      </header>
      {p.pending_brief && (
        <p><Status kind="caution" label="A decision is waiting" />{" "}
          <a href={`#/review/${encodeURIComponent(p.pending_brief)}`}>Review brief {p.pending_brief}</a></p>
      )}
      {p.error && <p className="error" role="alert">{p.error}. Rule tracks need live prices.</p>}
      {p.freshness?.stale && <p><Status kind="caution" label={`Stale data (${p.freshness.age_s}s old): ${p.freshness.error}`} /></p>}
      {p.tracks.length > 0 && (
        <ul className="tracks">
          {p.tracks.map((t) => <TrackRow key={t.rule} t={t} />)}
        </ul>
      )}
    </section>
  );
}

function TrackRow({ t }: { t: Track }) {
  const s = STATE[t.state];
  return (
    <li className={`track track-${t.state}`} data-rule={t.rule}>
      <span className="track-label">{t.label}</span>
      <span className="track-bar" aria-hidden="true">
        {t.progress_pct !== null && <span className={`track-fill fill-${s.kind}`} style={{ width: `${t.progress_pct}%` }} />}
      </span>
      <span className="num track-value">{t.value}{t.target && <span className="muted"> / {t.target}</span>}</span>
      <Status kind={s.kind} label={s.label} />
      {t.reason && <span className="muted track-reason">{t.reason}</span>}
    </li>
  );
}

function MonitorPanel({ onFinished }: { onFinished: () => void }) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const es = new EventSource(`/api/jobs/${encodeURIComponent(job.job_id)}/events`);
    es.addEventListener("done", (e) => {
      setJob(JSON.parse((e as MessageEvent).data));
      es.close();
      onFinished();
    });
    return () => es.close();
  }, [job, onFinished]);
  const running = job?.status === "running";
  const items = (job?.result as { items?: { position_id: string; ticker: string; action: string; brief_id: string | null; note: string }[] } | null)?.items;
  return (
    <section className="panel" aria-labelledby="monitor-title">
      <h2 id="monitor-title">Monitor</h2>
      <p className="muted">One pass over every open position: the management rules decide, the local model only picks
        between allowed actions, and any exit or roll becomes a brief on Today for your approval. Positions past expiry
        are settled.</p>
      <button type="button" disabled={running} onClick={() => {
        setError(null);
        post<Job>("/api/runs", { kind: "monitor" }).then(setJob).catch((e: Error) => setError(e.message));
      }}>{running ? "Monitor running…" : "Run monitor"}</button>
      {error && <p className="error" role="alert">{error}</p>}
      {job?.status === "failed" && <p className="error" role="alert">The monitor failed: {job.error}</p>}
      {items && (
        <ul className="monitor-items">
          {items.length === 0 && <li className="muted">No open positions to check.</li>}
          {items.map((i) => (
            <li key={i.position_id}>
              <strong>{i.ticker}</strong> {f.words(i.action)}
              {i.brief_id ? <> · <a href={`#/review/${encodeURIComponent(i.brief_id)}`}>review brief {i.brief_id}</a></> : null}
              {i.note && <span className="muted"> · {i.note}</span>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
