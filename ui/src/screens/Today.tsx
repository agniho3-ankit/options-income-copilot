// Today: the triage inbox. Lanes come straight from the order of review.queue; this screen only
// groups the items by the tier the API sent and formats the values.
import { useEffect, useState } from "react";
import { get, type InfoItem, type Queue, type QueueTrade } from "../lib/api";
import * as f from "../lib/format";
import { Status, tierStatus, verdictStatus } from "../components/Status";
import { moveFocus, useShortcuts } from "../lib/keys";

const LANES: { tier: QueueTrade["tier"]; title: string }[] = [
  { tier: "escalated", title: "Escalated" },
  { tier: "look_closely", title: "Look closely" },
  { tier: "standard", title: "Standard" },
];

type Load = { state: "loading" } | { state: "error"; message: string } | { state: "ready"; queue: Queue };

export function Today({ filter = null }: { filter?: string | null }) {
  const [load, setLoad] = useState<Load>({ state: "loading" });
  useShortcuts((key) => {
    if (key === "j" || key === "J") moveFocus("a.card[data-brief]", 1);
    if (key === "k" || key === "K") moveFocus("a.card[data-brief]", -1);
  });
  useEffect(() => {
    let live = true;
    get<Queue>("/api/queue")
      .then((queue) => live && setLoad({ state: "ready", queue }))
      .catch((e: Error) => live && setLoad({ state: "error", message: e.message }));
    return () => { live = false; };
  }, []);

  if (load.state === "loading") return <p className="muted" role="status">Loading the review queue…</p>;
  if (load.state === "error") {
    return <p className="error" role="alert">Could not load the review queue: {load.message}</p>;
  }
  const { heat } = load.queue;
  const match = (t: string) => !filter || t.toUpperCase().startsWith(filter);
  const trades = load.queue.trades.filter((t) => match(t.ticker));
  const info = load.queue.info.filter((i) => match(i.ticker));
  const active = trades.filter((t) => !t.deferred);
  const deferred = trades.filter((t) => t.deferred);

  return (
    <>
      <h1 className="visually-hidden">Today</h1>
      {filter && (
        <p className="filter-note">Showing <span className="num">{filter}</span> only · <a href="#/today">show all</a></p>
      )}
      <p className="summary" data-testid="summary">
        <strong className="num">{active.length}</strong> to review · <strong className="num">{info.length}</strong> info ·
        portfolio heat <strong className="num">{f.pct(heat.pct)}</strong> of <span className="num">{f.pct(heat.cap_pct, 0)}</span>
      </p>
      {active.length === 0 && info.length === 0 && deferred.length === 0 && (
        <p className="empty">{filter ? `No pending brief or info item for ${filter}.`
          : "No pending briefs. Start a run from Explore, or run the monitor from Positions."}</p>
      )}
      {LANES.map((lane) => {
        const items = active.filter((t) => t.tier === lane.tier);
        return items.length ? <Lane key={lane.tier} title={lane.title} items={items} /> : null;
      })}
      {info.length > 0 && (
        <section className="lane" aria-labelledby="lane-info">
          <h2 id="lane-info">Info, no trade <span className="lane-count num">{info.length}</span></h2>
          <ul className="cards">{info.map((i) => <InfoCard key={i.run_id} item={i} />)}</ul>
        </section>
      )}
      {deferred.length > 0 && <Lane title="Deferred" items={deferred} />}
    </>
  );
}

function Lane({ title, items }: { title: string; items: QueueTrade[] }) {
  const id = `lane-${title.toLowerCase().replace(/\s+/g, "-")}`;
  return (
    <section className="lane" aria-labelledby={id}>
      <h2 id={id}>{title} <span className="lane-count num">{items.length}</span></h2>
      <ul className="cards">{items.map((t) => <li key={t.brief_id}><TradeCard item={t} /></li>)}</ul>
    </section>
  );
}

export function TradeCard({ item }: { item: QueueTrade }) {
  const t = item.trade;
  const what = item.kind === "entry" ? f.words(t.strategy) : `${f.words(item.kind)}: ${f.words(t.strategy)}`;
  return (
    <a className="card" href={`#/review/${encodeURIComponent(item.brief_id)}`} data-brief={item.brief_id}>
      <div>
        <div className="card-ticker">{item.ticker}</div>
        <div className="num muted card-id">{item.brief_id}</div>
      </div>
      <div className="card-trade">
        <div>{what}{t.exit_rule ? ` · ${f.words(t.exit_rule)}` : ""}</div>
        <div className="num">{f.legs(t.legs)} · exp {f.date(t.expiry)}</div>
      </div>
      <div className="card-score">
        <Status kind={tierStatus(item.tier)} label={f.words(item.tier)} />
        <div className="num">score {item.score}</div>
        {item.kind === "exit" ? <div className="muted">close only</div>
          : <Status kind={verdictStatus(item.verdict)} label={`verifier ${item.verdict ?? "n/a"}`} />}
      </div>
      {item.top.length > 0 && (
        <ul className="card-signals" aria-label="Top signals">
          {item.top.map((s) => <li key={s.signal}>{s.text}</li>)}
        </ul>
      )}
    </a>
  );
}

function InfoCard({ item }: { item: InfoItem }) {
  return (
    <li className="card" data-run={item.run_id}>
      <div>
        <div className="card-ticker">{item.ticker}</div>
        <div className="num muted card-id">{item.run_id}</div>
      </div>
      <div className="card-trade">
        <div>{item.why}</div>
        <div className="muted">Next: {item.next}</div>
      </div>
      <div className="card-score">
        <Status kind={verdictStatus(item.outcome)} label={f.words(item.outcome.toLowerCase())} />
        <div className="num muted">{f.date(item.when)}</div>
      </div>
    </li>
  );
}
