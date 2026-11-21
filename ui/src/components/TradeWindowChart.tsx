// The trade-window chart: time x price. Every value drawn here comes from GET /briefs/{id}/window,
// which computes it in Python; this component only maps values to pixels (lib/scale).
import type { TradeWindow } from "../lib/api";
import * as f from "../lib/format";
import { bound, linear, monthTicks, niceTicks, path, spreadLabels, timeScale } from "../lib/scale";
import { Status } from "./Status";

const W = 760;
const H = 340;
const M = { top: 16, right: 132, bottom: 44, left: 56 };

export function TradeWindowChart({ w }: { w: TradeWindow }) {
  const x = timeScale(w.x_domain, [M.left, W - M.right]);
  const y = linear(w.y_domain, [H - M.bottom, M.top]);
  const [lo, hi] = w.y_domain;
  const x0 = x(w.today);
  const x1 = x(w.expiry);
  const yTicks = niceTicks(lo, hi, 5);
  const xTicks = monthTicks(w.x_domain[0], w.x_domain[1]);
  const cone = w.cone.length
    ? `${path(w.cone.map((c) => [x(c.date), y(c.high)]))} ${path([...w.cone].reverse().map((c) => [x(c.date), y(c.low)])).replace(/^M/, "L")} Z`
    : "";
  const levels = [
    ...w.strikes.filter((s) => s.strike !== null).map((s) => ({
      v: s.strike!, cls: s.side === "short" ? "strike-short" : "strike-long",
      label: `${s.side} ${s.option_type} ${f.strike(s.strike)}` })),
    ...w.breakevens.map((b) => ({ v: b, cls: "breakeven", label: `breakeven ${f.strike(b)}` })),
  ];
  const labelYs = spreadLabels(levels.map((l) => y(l.v)), 13);
  const axisY = H - M.bottom;

  return (
    <figure className="chart" aria-labelledby="window-caption">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Trade window for ${w.ticker}`} className="chart-svg">
        {w.zones.map((z, i) => (
          <rect key={`z${i}`} className={`zone zone-${z.kind}`} data-zone={z.kind}
            x={x0} width={x1 - x0} y={y(bound(z.high, hi))} height={y(bound(z.low, lo)) - y(bound(z.high, hi))} />
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`} className="tick">
            <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.left - 8} y={y(t)} dy="0.32em" textAnchor="end" className="num">{t}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} x={x(t)} y={axisY + 16} textAnchor="middle" className="num tick">{t.slice(0, 7)}</text>
        ))}
        {cone && <path d={cone} className="cone" data-testid="cone" />}
        <path d={path(w.history.map((p) => [x(p.date), y(p.close)]))} className="price" data-testid="price" />
        <line x1={x0} x2={x0} y1={M.top} y2={axisY} className="marker" />
        <text x={x0} y={M.top} dy="-4" textAnchor="middle" className="marker-label">Today</text>
        <line x1={x1} x2={x1} y1={M.top} y2={axisY} className="marker marker-expiry" />
        <text x={x1} y={M.top} dy="-4" textAnchor="middle" className="marker-label">Expiry</text>
        {levels.map((l, i) => (
          <Level key={`${l.cls}${l.label}`} y={y(l.v)} labelY={labelYs[i]} cls={l.cls} label={l.label} />
        ))}
        <line x1={M.left} x2={W - M.right} y1={axisY} y2={axisY} className="axis" />
        {w.events.map((e) => (
          <g key={`${e.date}${e.label}`} className={e.in_window ? "pin pin-in" : "pin"} data-event={e.kind}>
            <title>{`${e.label} ${e.date} (${e.source})${e.in_window ? ", inside the trade window" : ""}`}</title>
            <line x1={x(e.date)} x2={x(e.date)} y1={axisY} y2={axisY - 8} />
            <text x={x(e.date)} y={axisY + 32} textAnchor="middle">{e.in_window ? "▲" : "●"}</text>
          </g>
        ))}
      </svg>
      <figcaption id="window-caption" className="chart-caption">
        <span>Spot <span className="num">{f.num(w.spot)}</span> on <span className="num">{w.today}</span></span>
        {w.sigma.at_expiry && (
          <span>±1σ at expiry <span className="num">{f.num(w.sigma.at_expiry.low)}–{f.num(w.sigma.at_expiry.high)}</span>
            {" "}(IV <span className="num">{f.prob(w.sigma.iv)}</span>, {w.sigma.source})</span>
        )}
        {!w.sigma.iv && <Status kind="caution" label="No implied volatility: the cone is not drawn" />}
        {w.freshness.stale && <Status kind="caution" label={`Stale market data (${w.freshness.age_s}s old): ${w.freshness.error}`} />}
      </figcaption>
      <ul className="chart-legend" aria-label="Legend">
        <li><span className="swatch swatch-price" />Price</li>
        <li><span className="swatch swatch-cone" />±1σ expected move</li>
        <li><span className="swatch swatch-profit" />Profit at expiry</li>
        <li><span className="swatch swatch-loss" />Loss at expiry</li>
        <li><span className="swatch swatch-be" />Breakeven</li>
      </ul>
      {w.events.length > 0 && (
        <table className="events">
          <caption>Events on the time axis</caption>
          <thead><tr><th>Date</th><th>Event</th><th>Source</th><th>Trade window</th></tr></thead>
          <tbody>
            {w.events.map((e) => (
              <tr key={`${e.date}${e.label}`}>
                <td className="num">{e.date}</td><td>{e.label}</td><td>{e.source}</td>
                <td>{e.in_window ? <Status kind="caution" label="Inside" /> : <Status kind="info" label="Outside" />}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </figure>
  );
}

function Level({ y, labelY, cls, label }: { y: number; labelY: number; cls: string; label: string }) {
  return (
    <g className={`level ${cls}`}>
      <line x1={M.left} x2={W - M.right} y1={y} y2={y} />
      <text x={W - M.right + 6} y={labelY} dy="0.32em" className="num">{label}</text>
    </g>
  );
}
