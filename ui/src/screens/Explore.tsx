// Explore: start a run and watch the glass box. A run is a background job on the server; progress
// streams over Server-Sent Events from the run's spans and guardrail_events. When it ends, the
// screen opens the stored run: its timeline, its Tree-of-Thought, and a link to the brief.
import { useEffect, useRef, useState } from "react";
import { get, post, type GuardEvent, type Job, type JobResult, type RunRow, type RunView, type Span } from "../lib/api";
import * as f from "../lib/format";
import { elapsed, instant } from "../lib/scale";
import type { TotNode } from "../lib/tree";
import { Status } from "../components/Status";
import { Timeline } from "../components/Timeline";
import { TotTree } from "../components/TotTree";

const INFO = ["INSUFFICIENT_EVIDENCE", "BLOCK"];

export function Explore({ runId }: { runId: string | null }) {
  const [job, setJob] = useState<Job | null>(null);
  useEffect(() => {
    let live = true;
    get<{ active: Job | null }>("/api/jobs").then((r) => live && r.active && setJob(r.active)).catch(() => {});
    return () => { live = false; };
  }, []);
  return (
    <div className="explore">
      <h1 className="visually-hidden">Explore</h1>
      <StartForm busy={job?.status === "running"} onStarted={setJob} />
      {job && <LiveRun key={job.job_id} job={job} />}
      {runId ? <StoredRun key={runId} runId={runId} /> : <RecentRuns />}
    </div>
  );
}

function StartForm({ busy, onStarted }: { busy: boolean; onStarted: (j: Job) => void }) {
  const [ticker, setTicker] = useState("");
  const [kind, setKind] = useState<"multi_agent" | "tot">("multi_agent");
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  return (
    <section className="panel" aria-labelledby="start-title">
      <h2 id="start-title">Start a run</h2>
      <form className="start-form" onSubmit={(e) => {
        e.preventDefault();
        setSending(true);
        setError(null);
        post<Job>("/api/runs", { ticker: ticker.trim().toUpperCase(), kind })
          .then(onStarted).catch((x: Error) => setError(x.message)).finally(() => setSending(false));
      }}>
        <label className="field">
          <span>Ticker</span>
          <input value={ticker} onChange={(e) => setTicker(e.target.value)} maxLength={8} required
            autoComplete="off" spellCheck={false} className="num ticker-input" />
        </label>
        <fieldset className="kind">
          <legend className="visually-hidden">Run type</legend>
          <label><input type="radio" name="kind" checked={kind === "multi_agent"} onChange={() => setKind("multi_agent")} />
            Multi-agent (writes a brief if a trade passes)</label>
          <label><input type="radio" name="kind" checked={kind === "tot"} onChange={() => setKind("tot")} />
            Tree-of-Thought only (no brief)</label>
        </fieldset>
        <button type="submit" className="primary" disabled={busy || sending || ticker.trim() === ""}>Start run</button>
      </form>
      <p className="muted">A run takes 1–5 minutes on the local model and uses live market data. One run at a time;
        while it runs, rejections write their lesson from the template.</p>
      {busy && <p className="muted">A run is in progress below.</p>}
      {error && <p className="error" role="alert">{error}</p>}
    </section>
  );
}

function LiveRun({ job: initial }: { job: Job }) {
  const [job, setJob] = useState(initial);
  const [spans, setSpans] = useState<Span[]>([]);
  const [events, setEvents] = useState<GuardEvent[]>([]);
  const [now, setNow] = useState(() => Date.now());
  const [lost, setLost] = useState(false);
  const source = useRef<EventSource | null>(null);

  useEffect(() => {
    const es = new EventSource(`/api/jobs/${encodeURIComponent(initial.job_id)}/events`);
    source.current = es;
    const data = (e: Event) => JSON.parse((e as MessageEvent).data);
    es.addEventListener("job", (e) => setJob(data(e)));
    es.addEventListener("span", (e) => {
      const s: Span = data(e);
      setSpans((prev) => {
        const i = prev.findIndex((p) => p.run_id === s.run_id && p.span_id === s.span_id);
        return i < 0 ? [...prev, s] : prev.map((p, k) => (k === i ? s : p));
      });
    });
    es.addEventListener("guardrail", (e) => setEvents((prev) => [...prev, data(e)]));
    es.addEventListener("done", (e) => { setJob(data(e)); es.close(); });
    es.onerror = () => { if (es.readyState === EventSource.CLOSED) setLost(true); };
    return () => es.close();
  }, [initial.job_id]);

  const running = job.status === "running";
  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [running]);

  const started = instant(job.created_at) ?? now;
  const end = instant(job.finished_at) ?? now;
  return (
    <section className="panel" aria-labelledby="live-title" aria-busy={running}>
      <h2 id="live-title">{running ? "Running" : "Finished"}: {job.ticker} {f.words(job.kind)}</h2>
      <p className="muted" role="status">
        {running ? <Status kind="info" label="In progress" /> : job.status === "done"
          ? <Status kind="pass" label="Done" /> : <Status kind="block" label="Failed" />}
        {" "}elapsed <span className="num">{elapsed(started, end)}</span> · job <span className="num">{job.job_id}</span>
      </p>
      {lost && running && <p className="error" role="alert">The progress stream closed. Reload the page to reattach.</p>}
      <Timeline spans={spans} events={events} now={end} />
      {job.error && <p className="error" role="alert">The run failed: {job.error}</p>}
      {job.result && <Outcome result={job.result} />}
    </section>
  );
}

function Outcome({ result: r, runId }: { result: JobResult; runId?: string }) {
  const id = r.run_id ?? runId;
  return (
    <div className="outcome">
      <p>
        Outcome <strong className="num">{r.outcome ?? "–"}</strong>
        {r.llm_calls !== undefined && <> · <span className="num">{r.llm_calls}</span> LLM calls</>}
        {r.latency_ms !== undefined && <> · <span className="num">{f.num(r.latency_ms, 0)}</span> ms</>}
      </p>
      {r.brief_id ? (
        <p><a href={`#/review/${encodeURIComponent(r.brief_id)}`}>Open brief {r.brief_id} in Review</a></p>
      ) : INFO.includes(r.outcome ?? "") ? (
        <p>No trade. This run is an info item on <a href="#/today">Today</a> with the reason and the next step.</p>
      ) : (
        <p className="muted">No brief was written.</p>
      )}
      {(r.reasons?.length ?? 0) > 0 && <ul className="muted">{r.reasons!.map((x, i) => <li key={i}>{x}</li>)}</ul>}
      {id && <p><a href={`#/explore/${encodeURIComponent(id)}`}>Open the stored run</a></p>}
    </div>
  );
}

function StoredRun({ runId }: { runId: string }) {
  const [view, setView] = useState<RunView | null>(null);
  const [nodes, setNodes] = useState<TotNode[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    get<RunView>(`/api/runs/${encodeURIComponent(runId)}`)
      .then((v) => {
        if (!live) return;
        setView(v);
        if (v.tot_run_id) {
          get<{ nodes: TotNode[] }>(`/api/runs/${encodeURIComponent(v.tot_run_id)}/tot`)
            .then((t) => live && setNodes(t.nodes)).catch(() => live && setNodes([]));
        } else setNodes([]);
      })
      .catch((e: Error) => live && setError(e.message));
    return () => { live = false; };
  }, [runId]);
  if (error) return <p className="error" role="alert">Could not load run {runId}: {error}</p>;
  if (!view) return <p className="muted" role="status">Loading run {runId}…</p>;
  const r = view.run;
  const end = instant(r.finished_at) ?? instant(r.started_at) ?? 0;
  return (
    <>
      <section className="panel" aria-labelledby="run-title">
        <h2 id="run-title">Run {r.ticker} {f.words(r.kind)}</h2>
        <p className="muted"><span className="num">{r.run_id}</span> · started <span className="num">{r.started_at}</span>
          {" "}· <a href="#/explore">all runs</a></p>
        <Outcome runId={r.run_id} result={{ outcome: r.outcome ?? undefined, brief_id: view.brief_id, reasons: view.reasons }} />
        {view.spans.length > 0 && <Timeline spans={view.spans} events={view.guardrail_events} now={end} />}
      </section>
      <section className="panel" aria-labelledby="tree-title">
        <h2 id="tree-title">Tree-of-Thought</h2>
        {nodes === null ? <p className="muted">Loading the tree…</p> : <TotTree nodes={nodes} />}
      </section>
    </>
  );
}

function RecentRuns() {
  const [runs, setRuns] = useState<RunRow[] | null>(null);
  useEffect(() => {
    let live = true;
    get<{ runs: RunRow[] }>("/api/runs?limit=40")
      .then((r) => live && setRuns(r.runs.filter((x) => x.kind === "multi_agent" || x.kind === "tot")))
      .catch(() => live && setRuns([]));
    return () => { live = false; };
  }, []);
  return (
    <section className="panel" aria-labelledby="recent-title">
      <h2 id="recent-title">Recent runs</h2>
      {runs === null ? <p className="muted">Loading…</p> : runs.length === 0 ? <p className="muted">No runs yet.</p> : (
        <table>
          <thead><tr><th>Started</th><th>Ticker</th><th>Kind</th><th>Outcome</th><th>Run</th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id}>
                <td className="num">{r.started_at.slice(0, 16).replace("T", " ")}</td><td>{r.ticker}</td>
                <td>{f.words(r.kind)}</td><td className="num">{r.outcome ?? "unfinished"}</td>
                <td><a className="num" href={`#/explore/${encodeURIComponent(r.run_id)}`}>{r.run_id}</a></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
