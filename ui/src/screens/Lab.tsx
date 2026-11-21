// Lab: the evidence for the submission, read from saved results (lab.build and observability.metrics).
// Nothing is run from here: starting an eval shows the long-run warning and the command to run.
import { useEffect, useState } from "react";
import { get, type Aggregate, type Lab as LabData, type Metrics } from "../lib/api";
import { barLengths } from "../lib/scale";
import * as f from "../lib/format";
import { Reliability } from "../components/Reliability";
import { SHAPES, Status } from "../components/Status";

const SCORE: { field: keyof Aggregate; label: string; kind: "count" | "fraction" | "seconds" | "number" }[] = [
  { field: "false_blocks", label: "False blocks", kind: "count" },
  { field: "outcome_accuracy", label: "Outcome accuracy", kind: "fraction" },
  { field: "verdict_accuracy", label: "Verdict accuracy", kind: "fraction" },
  { field: "paraphrase_agreement", label: "Paraphrase agreement", kind: "fraction" },
  { field: "outcome_consistency", label: "Outcome consistency", kind: "fraction" },
  { field: "latency_p50_s", label: "Latency p50", kind: "seconds" },
  { field: "latency_p95_s", label: "Latency p95", kind: "seconds" },
  { field: "llm_calls_mean", label: "LLM calls per run", kind: "number" },
];

function show(v: number | undefined | null, kind: string): string {
  if (v === undefined || v === null) return "–";
  if (kind === "fraction") return f.prob(v, 0);
  if (kind === "seconds") return `${f.num(v, 1)} s`;
  if (kind === "number") return f.num(v, 1);
  return String(v);
}

export function Lab() {
  const [lab, setLab] = useState<LabData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { get<LabData>("/api/lab").then(setLab).catch((e: Error) => setError(e.message)); }, []);
  if (error) return <p className="error" role="alert">Could not load the lab results: {error}</p>;
  if (!lab) return <p className="muted" role="status">Loading saved results…</p>;
  const a = lab.after_floor;
  const b = lab.before_floor;
  return (
    <div className="lab">
      <h1 className="visually-hidden">Lab</h1>
      <section className="panel" aria-labelledby="score-title">
        <h2 id="score-title">Evaluation scoreboard</h2>
        {!a ? <p className="muted">No saved evaluation yet. See "Run an evaluation" below.</p> : (
          <>
            <div className="hero" data-testid="unsafe">
              <span className="hero-number num">{a.unsafe_passed ?? "–"}</span>
              <span className="hero-label">
                <Status kind={a.unsafe_passed === 0 ? "pass" : "block"} label="Unsafe trades passed" />
                <span className="muted">over <span className="num">{a.runs ?? "–"}</span> runs of{" "}
                  <span className="num">{a.cases ?? "–"}</span> labelled cases{lab.repeats ? `, ${lab.repeats} repeats each` : ""}</span>
              </span>
            </div>
            <table className="scores">
              <thead><tr><th>Metric</th><th>Before the liquidity floor</th><th>After</th></tr></thead>
              <tbody>
                <tr><td>Unsafe trades passed</td><td className="num">{show(b?.unsafe_passed, "count")}</td>
                  <td className="num">{show(a.unsafe_passed, "count")}</td></tr>
                {SCORE.map((m) => (
                  <tr key={m.field}><td>{m.label}</td><td className="num">{show(b?.[m.field], m.kind)}</td>
                    <td className="num">{show(a[m.field], m.kind)}</td></tr>
                ))}
              </tbody>
            </table>
            {lab.before_note && <p className="muted">Before-floor results were rescored: {lab.before_note}.</p>}
            <p className="muted">Sources: <span className="num">{lab.sources.eval ?? "evals table"}</span>
              {lab.sources.before_floor && <>, <span className="num">{lab.sources.before_floor}</span></>}.</p>
          </>
        )}
      </section>

      {lab.ablation.rows.length > 0 && (
        <section className="panel" aria-labelledby="abl-title">
          <h2 id="abl-title">Ablation: one guardrail off at a time</h2>
          <div className="table-scroll">
            <table className="heat">
              <thead><tr><th>Configuration</th>{lab.ablation.columns.map((c) => <th key={c.key}>{c.label}</th>)}</tr></thead>
              <tbody>
                {lab.ablation.rows.map((r) => (
                  <tr key={r.config}>
                    <th scope="row" className="num">{r.config}</th>
                    {lab.ablation.columns.map((c) => {
                      const cell = r.cells[c.key];
                      return (
                        <td key={c.key} className={`num heat-${cell.direction} heat-l${cell.level}`} data-direction={cell.direction}>
                          {cell.value ?? "–"}
                          {cell.direction === "worse" && <span className="heat-mark" aria-label="worse than full"> {SHAPES.caution}</span>}
                          {cell.direction === "better" && <span className="heat-mark" aria-label="better than full"> {SHAPES.pass}</span>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted">Compared with the full configuration: {SHAPES.pass} better, {SHAPES.caution} worse; darker
            means a larger change. The ablation ran once per case before the liquidity floor.</p>
          {lab.ablation.note && <p><Status kind="caution" label="Note" /> {lab.ablation.note}</p>}
        </section>
      )}

      {lab.calibration && (
        <section className="panel" aria-labelledby="cal-title">
          <h2 id="cal-title">Calibration of the filings verdict</h2>
          <div className="rel-grid">
            {lab.calibration.confidence && <Reliability title="Verbal confidence" c={lab.calibration.confidence} />}
            {lab.calibration.p_true && <Reliability title="P(True) from token log-probabilities" c={lab.calibration.p_true} />}
          </div>
          <p className="muted">Bars show the accuracy in each confidence bin, dots the mean stated confidence; the diagonal is
            perfect calibration. Each bin shows its count.</p>
        </section>
      )}

      {lab.cases.length > 0 && (
        <section className="panel" aria-labelledby="cases-title">
          <h2 id="cases-title">Per case</h2>
          <table>
            <thead><tr><th>Case</th><th>Runs</th><th>Verdicts</th><th>Outcomes</th><th>Unsafe</th><th>False blocks</th></tr></thead>
            <tbody>
              {lab.cases.map((c) => (
                <tr key={c.case}>
                  <td className="num">{c.case}</td><td className="num">{c.runs ?? "–"}</td>
                  <td className="num">{Object.entries(c.verdicts ?? {}).map(([k, n]) => `${k} ${n}`).join(", ")}</td>
                  <td className="num">{Object.entries(c.outcomes ?? {}).map(([k, n]) => `${k} ${n}`).join(", ")}</td>
                  <td className="num">{c.unsafe_passed ?? "–"}</td><td className="num">{c.false_blocks ?? "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <MetricsPanel />

      <section className="panel" aria-labelledby="run-eval-title">
        <h2 id="run-eval-title">Run an evaluation</h2>
        <p><Status kind="caution" label="Long run" /> {lab.run.warning}</p>
        <p className="muted">Nothing starts from this page. Copy a command into a terminal in the project folder:</p>
        <CommandLine cmd={lab.run.eval_command} />
        <CommandLine cmd={lab.run.ablation_command} />
      </section>
    </div>
  );
}

function CommandLine({ cmd }: { cmd: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="cmd">
      <code>{cmd}</code>
      <button type="button" onClick={() => {
        navigator.clipboard?.writeText(cmd).then(() => setCopied(true)).catch(() => setCopied(false));
      }}>{copied ? "Copied" : "Copy"}</button>
    </div>
  );
}

function MetricsPanel() {
  const [db, setDb] = useState<"main" | "eval">("main");
  const [m, setM] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    setM(null);
    setError(null);
    get<Metrics>(`/api/metrics?db=${db}`).then((x) => live && setM(x)).catch((e: Error) => live && setError(e.message));
    return () => { live = false; };
  }, [db]);
  const codes = Object.entries(m?.reject_codes ?? {});
  const lengths = barLengths(codes.map(([, n]) => n));
  return (
    <section className="panel" aria-labelledby="metrics-title">
      <h2 id="metrics-title">Observability</h2>
      <div className="segmented" role="radiogroup" aria-label="Database">
        {(["main", "eval"] as const).map((d) => (
          <label key={d} className={db === d ? "seg-on" : undefined}>
            <input type="radio" name="metrics-db" checked={db === d} onChange={() => setDb(d)} />
            {d === "main" ? "Your runs (copilot.db)" : "Evaluation sandbox (eval.db)"}
          </label>
        ))}
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {!m && !error && <p className="muted">Loading…</p>}
      {m && (
        <div className="metrics-grid">
          <div>
            <h3>Guardrail decisions</h3>
            <table className="guard-table">
              <tbody>
                {Object.entries(m.guardrails).map(([g, d]) => (
                  <tr key={g}>
                    <td>{f.words(g)}</td>
                    <td className="stack-cell">
                      <span className="stack" aria-label={`pass ${d.pass ?? 0}, flag ${d.flag ?? 0}, block ${d.block ?? 0}`}>
                        {(["pass", "flag", "block"] as const).map((k) => (d[k] ? (
                          <span key={k} className={`stack-${k}`} style={{ flexGrow: d[k] }} title={`${k} ${d[k]}`} />) : null))}
                      </span>
                    </td>
                    <td className="num">{SHAPES.pass} {d.pass ?? 0} · {SHAPES.caution} {d.flag ?? 0} · {SHAPES.block} {d.block ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <h3>Latency per agent</h3>
            <table>
              <thead><tr><th>Agent</th><th>Spans</th><th>p50</th><th>p95</th></tr></thead>
              <tbody>
                {Object.entries(m.latency_per_agent).map(([a, v]) => (
                  <tr key={a}><td>{f.words(a)}</td><td className="num">{v.n}</td>
                    <td className="num">{show(v.p50_s, "seconds")}</td><td className="num">{show(v.p95_s, "seconds")}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="muted num">LLM calls per run: mean {show(m.llm_calls_per_run.mean, "number")}, p95{" "}
              {show(m.llm_calls_per_run.p95, "number")} over {m.llm_calls_per_run.runs} runs · cloud fallbacks {m.claude_fallbacks}</p>
          </div>
          <div>
            <h3>Reject codes</h3>
            {codes.length === 0 ? <p className="muted">No rejections recorded.</p> : (
              <ul className="code-bars">
                {codes.map(([k, n], i) => (
                  <li key={k} data-code={k}><span>{f.words(k)}</span>
                    <span className="code-track"><span className="code-fill" style={{ width: `${lengths[i]}%` }} /></span>
                    <span className="num">{n}</span></li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
