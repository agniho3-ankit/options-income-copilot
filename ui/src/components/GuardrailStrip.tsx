// The 12 guardrails in pipeline order, grouped pre / during / post. Each decision comes from
// guardrails.pipeline_summary over the run's guardrail_events; clicking one shows the reasons.
import { useState } from "react";
import type { StripItem } from "../lib/api";
import * as f from "../lib/format";
import { Status, type StatusKind } from "./Status";

const STAGES = [["pre", "Before"], ["during", "During"], ["post", "After"]] as const;

export function decisionStatus(d: string | null): StatusKind {
  return d === "pass" ? "pass" : d === "flag" ? "caution" : d === "block" ? "block" : "info";
}

export function GuardrailStrip({ items }: { items: StripItem[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const chosen = items.find((g) => g.name === open);
  return (
    <section className="panel" aria-labelledby="strip-title">
      <h2 id="strip-title">Guardrail pipeline</h2>
      <div className="strip">
        {STAGES.map(([stage, title]) => (
          <div key={stage} className="strip-stage">
            <div className="strip-stage-title">{title}</div>
            <div className="strip-items">
              {items.filter((g) => g.stage === stage).map((g) => (
                <button key={g.name} type="button" className="strip-item" aria-expanded={open === g.name}
                  aria-controls="strip-detail" onClick={() => setOpen(open === g.name ? null : g.name)}>
                  <Status kind={decisionStatus(g.decision)} label={f.words(g.name)} />
                  <span className="visually-hidden">: {g.decision ?? "not run"}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      {chosen && (
        <div id="strip-detail" className="strip-detail" role="region" aria-label={`${f.words(chosen.name)} details`}>
          <p><strong>{f.words(chosen.name)}</strong> <span className="muted">({chosen.kind})</span>: {chosen.what}</p>
          {chosen.events.length === 0 ? <p className="muted">No event was recorded for this guardrail on this run.</p> : (
            <ul>
              {chosen.events.map((e) => (
                <li key={e.seq}>
                  <Status kind={decisionStatus(e.decision)} label={e.decision} />{" "}
                  {e.signal && <span className="muted">[{f.words(e.signal)}] </span>}{e.reason}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
