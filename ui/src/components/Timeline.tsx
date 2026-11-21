// Live supervisor timeline: one lane per agent, a bar per span (an open span grows until it ends),
// guardrail decisions as shape markers on the lane of the span that was running, and each span's
// LLM-call count as recorded. Values come from the spans and guardrail_events tables.
import type { GuardEvent, Span } from "../lib/api";
import * as f from "../lib/format";
import { bar, instant, instantScale, rowTop, spanWindow } from "../lib/scale";
import { SHAPES } from "./Status";
import { decisionStatus } from "./GuardrailStrip";

const ORDER = ["supervisor", "researcher", "strategist", "risk_critic", "verifier_gate", "brief_writer", "learner"];
const W = 760;
const LANE = 34;
const LEFT = 116;
const TOP = 8;

export function Timeline({ spans, events, now }: { spans: Span[]; events: GuardEvent[]; now: number }) {
  const shown = spans.filter((s) => s.agent !== "session" && instant(s.started_at) !== null);
  const lanes = [...ORDER.filter((a) => shown.some((s) => s.agent === a)),
    ...[...new Set(shown.map((s) => s.agent))].filter((a) => !ORDER.includes(a))];
  if (shown.length === 0) return <p className="muted">No agent has started yet.</p>;
  const x = instantScale(spanWindow(shown, now), [LEFT, W - 12]);
  const laneOf = new Map(lanes.map((a, i) => [a, i]));
  const spanById = new Map(shown.map((s) => [s.span_id, s]));
  const height = rowTop(lanes.length, LANE, TOP);

  return (
    <figure className="timeline">
      <svg viewBox={`0 0 ${W} ${height}`} className="timeline-svg" role="img"
        aria-label={`Supervisor timeline: ${shown.length} spans across ${lanes.length} agents`}>
        {lanes.map((a, i) => (
          <g key={a}>
            <line x1={LEFT} x2={W - 12} y1={rowTop(i + 1, LANE, TOP)} y2={rowTop(i + 1, LANE, TOP)} className="grid" />
            <text x={LEFT - 10} y={rowTop(i, LANE, TOP)} dy={LANE - 14} textAnchor="end">{f.words(a)}</text>
          </g>
        ))}
        {shown.map((s) => {
          const lane = laneOf.get(s.agent) ?? 0;
          const open = s.ended_at === null;
          const [bx, bw] = bar(x(s.started_at), x(instant(s.ended_at) ?? now));
          return (
            <g key={`${s.run_id}${s.span_id}`} className={`span-bar${open ? " span-open" : ""}${s.error ? " span-error" : ""}`}
              data-span={s.span_id} data-open={open || undefined}>
              <title>{`${f.words(s.agent)}: ${open ? "running" : `${s.duration_ms ?? 0} ms`}, ${s.llm_calls} LLM call(s)${s.decision ? `, ${s.decision}` : ""}${s.error ? `, error ${s.error}` : ""}`}</title>
              <rect x={bx} y={rowTop(lane, LANE, TOP + 6)} width={bw} height={LANE - 14} rx={3} />
              {s.llm_calls > 0 && (
                <text x={bx + 4} y={rowTop(lane, LANE, TOP)} dy={LANE - 14} className="num span-label">{s.llm_calls}</text>
              )}
            </g>
          );
        })}
        {events.map((e) => {
          const s = e.span_id ? spanById.get(e.span_id) : undefined;
          const lane = s ? laneOf.get(s.agent) ?? 0 : 0;
          const at = instant(e.created_at) ?? instant(s?.started_at) ?? now;
          const kind = decisionStatus(e.decision);
          return (
            <text key={`${e.run_id}${e.seq}`} x={x(at)} y={rowTop(lane, LANE, TOP)} dy={5}
              className={`marker-shape status-${kind}`} textAnchor="middle" data-guardrail={e.guardrail}>
              <title>{`${f.words(e.guardrail)} (${e.stage}): ${e.decision}. ${e.reason ?? ""}`}</title>
              {SHAPES[kind]}
            </text>
          );
        })}
      </svg>
      <figcaption className="muted">Bars are agent spans; numbers on a bar are its LLM calls; shapes above a lane are
        guardrail decisions (◆ pass, ▲ flag, ■ block).</figcaption>
    </figure>
  );
}
