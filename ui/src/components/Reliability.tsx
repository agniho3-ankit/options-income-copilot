// Reliability diagram: for each confidence bin, the observed accuracy (bar) against the bin's mean
// stated confidence (dot), with the count of runs in the bin. Values come from evaluation.reliability.
import type { Calibration } from "../lib/api";
import * as f from "../lib/format";
import { linear } from "../lib/scale";

const W = 340;
const H = 240;
const M = { top: 12, right: 12, bottom: 44, left: 40 };

function edges(bin: string): [number, number] {
  const [lo, hi] = bin.split("-").map(Number);
  return [lo, hi];
}

export function Reliability({ title, c }: { title: string; c: Calibration }) {
  const x = linear([0, 1], [M.left, W - M.right]);
  const y = linear([0, 1], [H - M.bottom, M.top]);
  return (
    <figure className="reliability">
      <figcaption>
        <strong>{title}</strong> <span className="muted">n <span className="num">{c.n}</span> · ECE{" "}
          <span className="num">{f.num(c.ece, 3)}</span> · Brier <span className="num">{f.num(c.brier, 3)}</span></span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Reliability diagram, ${title}`} className="rel-svg">
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.left - 6} y={y(t)} dy="0.32em" textAnchor="end" className="num">{t}</text>
          </g>
        ))}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className="diagonal" />
        {c.reliability.map((b) => {
          const [lo, hi] = edges(b.bin);
          return (
            <g key={b.bin} data-bin={b.bin}>
              {b.accuracy !== null && (
                <rect x={x(lo) + 3} width={x(hi) - x(lo) - 6} y={y(b.accuracy)} height={y(0) - y(b.accuracy)} className="rel-bar">
                  <title>{`${b.bin}: accuracy ${b.accuracy}, mean confidence ${b.mean_conf}, n ${b.n}`}</title>
                </rect>
              )}
              {b.mean_conf !== null && <circle cx={x(b.mean_conf)} cy={y(b.mean_conf)} r={4} className="rel-dot" />}
              <text x={x(lo) + 4} y={y(0)} dy={16} className="num">{b.bin}</text>
              <text x={x(lo) + 4} y={y(0)} dy={30} className="num rel-n">n {b.n}</text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
