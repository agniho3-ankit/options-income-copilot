// Home: what this project is, what each tab does, how a trade reaches you, the rules that keep it
// safe, and a live "at a glance" line. Thresholds are read from the loaded playbook, never typed in.
import { useEffect, useState } from "react";
import { get, type Job, type Lab, type PlaybookView, type Queue, type Session } from "../lib/api";
import * as f from "../lib/format";
import { SHORTCUTS } from "../lib/keys";
import { Status } from "../components/Status";

const TABS: { href: string; name: string; what: string; use: string }[] = [
  { href: "#/today", name: "Today", what: "The review queue. Every pending brief, ranked by a deterministic risk score into escalated, look closely and standard lanes, plus info items for runs that ended in BLOCK or INSUFFICIENT_EVIDENCE.",
    use: "Start here each session. Escalated briefs need the closest look." },
  { href: "#/review", name: "Review", what: "One brief on one screen: the numbers, the trade-window chart (price, ±1σ cone, strikes, breakevens, event dates), the 12-guardrail pipeline, why it was escalated, the cited filings, the judge's flags and the alternatives the search considered.",
    use: "Decide here: approve as a paper ticket, reject with a reason code, or defer." },
  { href: "#/explore", name: "Explore", what: "Start a multi-agent or Tree-of-Thought run for one ticker and watch it live: a lane per agent, guardrail decisions as they happen, and the search tree with every pruned branch and its reason.",
    use: "Use it to see how a brief was produced, or to research a ticker." },
  { href: "#/positions", name: "Positions", what: "Open paper positions as rule tracks: profit toward the target, days toward the roll point, loss toward the stop, the 1σ move, earnings and more, plus portfolio heat against the cap.",
    use: "Run the monitor here; any exit or roll it proposes lands on Today for approval." },
  { href: "#/journal", name: "Journal", what: "The trade log (playbook Skill #7) and monthly summaries: P&L, win rate, return on risk and the exit rule that closed each trade.",
    use: "Review results and which rules are doing the work." },
  { href: "#/lab", name: "Lab", what: "The evidence: unsafe trades passed, false blocks, results before and after the liquidity floor, the guardrail ablation, calibration diagrams and observability for your runs or the evaluation sandbox.",
    use: "Check that the safety machinery works, with numbers." },
  { href: "#/playbook", name: "Playbook", what: "The rules the agent follows, read-only: version, fingerprint, key thresholds and every section of playbook.yaml.",
    use: "Look up a threshold. Only the owner changes it, in the file." },
];

const PIPELINE: { step: string; who: string; what: string }[] = [
  { step: "Research", who: "Researcher agent", what: "Retrieves recent SEC filings for the ticker and checks for landmines (deals, tender offers, financing). Too little evidence means INSUFFICIENT_EVIDENCE, never CLEAR." },
  { step: "Explore", who: "Strategist + Tree-of-Thought", what: "Searches strategy, then expiry, then delta. Python prices every candidate; thin, illiquid expiries are withheld." },
  { step: "Critique", who: "Risk critic", what: "Can keep, lower or veto a candidate, citing a check or a lesson. It can never raise a score." },
  { step: "Verify", who: "Deterministic verifier", what: "Hard rules from the playbook: PoP, risk, heat, earnings, event dates, open interest. Final and cannot be overridden." },
  { step: "Write", who: "Brief writer + judge", what: "Writes the brief from tool numbers only; the judge flags any claim the facts do not support." },
  { step: "Decide", who: "You", what: "The brief waits on Today. Approve writes a paper ticket; nothing is ever sent to a broker." },
  { step: "Manage", who: "Monitor", what: "Applies the management rules to open paper positions and proposes exits and rolls, which come back to you." },
];

const TECHNIQUES: { name: string; where: string }[] = [
  { name: "ReAct tool use", where: "Agents plan tool calls and re-plan on the verifier's observations" },
  { name: "Retrieval-augmented generation", where: "Filings landmine check with coverage guard and grader (Review → Evidence)" },
  { name: "Tree-of-Thought", where: "Strategy → expiry → delta beam search (Explore → tree)" },
  { name: "Multi-agent supervisor", where: "LangGraph supervisor with CrewAI roles (Explore → timeline)" },
  { name: "Reflexion", where: "Every rejection becomes a lesson for future runs (Review → Reject)" },
  { name: "Guardrails and LLM-as-judge", where: "12 guardrails across pre, during and post (Review → pipeline)" },
  { name: "Evaluation and ablation", where: "Labelled cases, calibration, one-guardrail-off study (Lab)" },
  { name: "Human in the loop", where: "Every entry, exit and roll needs your approval (Today, Review)" },
];

const GLOSSARY: { term: string; meaning: string }[] = [
  { term: "Credit spread", meaning: "Sell one option and buy a further one for protection; you collect a credit and your loss is capped." },
  { term: "Iron condor", meaning: "A put spread and a call spread together; it profits if the price stays between them." },
  { term: "Cash-secured put", meaning: "Sell a put with cash set aside to buy the shares if assigned." },
  { term: "PoP", meaning: "Probability of profit: the chance the price finishes past the breakeven(s), from implied volatility." },
  { term: "DTE", meaning: "Days to expiry." },
  { term: "Breakeven", meaning: "The price at expiry where the trade neither makes nor loses money." },
  { term: "±1σ cone", meaning: "The expected move: spot × implied volatility × √(days ÷ 365), drawn from today to expiry." },
  { term: "Portfolio heat", meaning: "The sum of max losses of open positions as a share of the (synthetic) portfolio." },
  { term: "Escalation tier", meaning: "How closely to look: a deterministic score from signals such as judge flags or verifier cautions. Every tier still needs your approval." },
  { term: "Paper ticket", meaning: "A record of an approved trade. The paper broker simulates the fill; no real order exists." },
];

function threshold(pb: PlaybookView | null, path: string, fmt: (v: number) => string): string {
  const v = pb?.key_thresholds.find((k) => k.path === path)?.value;
  return typeof v === "number" ? fmt(v) : "…";
}

export function Home() {
  const [session, setSession] = useState<Session | null>(null);
  const [queue, setQueue] = useState<Queue | null>(null);
  const [lab, setLab] = useState<Lab | null>(null);
  const [pb, setPb] = useState<PlaybookView | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  useEffect(() => {
    let live = true;
    const keep = <T,>(set: (v: T) => void) => (v: T) => { if (live) set(v); };
    get<Session>("/api/session").then(keep(setSession)).catch(() => {});
    get<Queue>("/api/queue").then(keep(setQueue)).catch(() => {});
    get<Lab>("/api/lab").then(keep(setLab)).catch(() => {});
    get<PlaybookView>("/api/playbook").then(keep(setPb)).catch(() => {});
    get<{ active: Job | null }>("/api/jobs").then((r) => keep(setJob)(r.active)).catch(() => {});
    return () => { live = false; };
  }, []);

  const pending = queue ? queue.trades.filter((t) => !t.deferred) : null;
  const escalated = pending ? pending.filter((t) => t.tier === "escalated") : null;
  const unsafe = lab?.after_floor?.unsafe_passed;

  return (
    <div className="home">
      <section className="home-hero" aria-labelledby="home-title">
        <h1 id="home-title">Options Income Copilot</h1>
        <p className="lede">An agent that runs one person's written, rules-based options-income playbook: it researches
          large-cap tickers, builds credit spreads, iron condors and cash-secured puts, checks every proposal against
          hard rules, and hands you a brief to decide on. Python does every calculation; the local language model
          plans, reads filings and explains. You approve every trade, and nothing places a real order.</p>
        <p className="disclaimer" role="note"><strong>Educational capstone project.</strong> Not financial advice. Paper
          trading only, with a synthetic portfolio and public data. Built for the Carnegie Mellon University Agentic AI
          course.</p>
        <div className="home-actions">
          <a className="button primary" href="#/today">Open the review queue</a>
          <a className="button" href="#/explore">Start a run</a>
          <a className="button" href="#/lab">See the evidence</a>
        </div>
      </section>

      <section className="panel" aria-labelledby="glance-title">
        <h2 id="glance-title">At a glance</h2>
        <dl className="glance">
          <div><dt>To review</dt><dd className="num">{pending ? pending.length : "…"}</dd>
            <dd className="muted">{escalated ? `${escalated.length} escalated` : ""}</dd></div>
          <div><dt>Info items</dt><dd className="num">{queue ? queue.info.length : "…"}</dd>
            <dd className="muted">runs with no trade</dd></div>
          <div><dt>Portfolio heat</dt><dd className="num">{queue ? f.pct(queue.heat.pct) : "…"}</dd>
            <dd className="muted">{queue ? `cap ${f.pct(queue.heat.cap_pct, 0)}` : ""}</dd></div>
          <div><dt>Unsafe trades passed</dt><dd className="num">{unsafe ?? "–"}</dd>
            <dd className="muted">{unsafe === undefined ? "no saved evaluation" : "latest saved evaluation"}</dd></div>
          <div><dt>Playbook</dt><dd className="num">{session ? `v${session.playbook.version}` : "…"}</dd>
            <dd className="muted num">{session ? session.playbook.sha256.slice(0, 12) : ""}</dd></div>
          <div><dt>Run in progress</dt><dd>{job ? <a href="#/explore">{job.ticker} {f.words(job.kind)}</a> : "none"}</dd>
            <dd className="muted">one LLM job at a time</dd></div>
        </dl>
      </section>

      <section className="panel" aria-labelledby="tabs-title">
        <h2 id="tabs-title">The tabs</h2>
        <div className="tab-cards">
          {TABS.map((t) => (
            <a key={t.name} className="tab-card" href={t.href}>
              <h3>{t.name}</h3>
              <p>{t.what}</p>
              <p className="muted">{t.use}</p>
            </a>
          ))}
        </div>
      </section>

      <section className="panel" aria-labelledby="flow-title">
        <h2 id="flow-title">How a trade reaches you</h2>
        <ol className="flow">
          {PIPELINE.map((p) => (
            <li key={p.step}><strong>{p.step}</strong> <span className="muted">· {p.who}</span><p>{p.what}</p></li>
          ))}
        </ol>
      </section>

      <section className="panel" aria-labelledby="rules-title">
        <h2 id="rules-title">Rules that cannot be bent</h2>
        <ul className="rules">
          <li><Status kind="pass" label="Python does the maths." /> Strikes, credit, max loss, breakevens, PoP, risk and
            heat come from deterministic code. The language model never calculates.</li>
          <li><Status kind="pass" label="The verifier is final." /> A proposal must clear PoP of at least{" "}
            <span className="num">{threshold(pb, "entry.pop_min", (v) => f.prob(v, 0))}</span>, position risk under{" "}
            <span className="num">{threshold(pb, "entry.position_risk_pct.block", (v) => f.pct(v, 0))}</span>, portfolio
            heat under <span className="num">{threshold(pb, "entry.portfolio_heat_pct.block", (v) => f.pct(v, 0))}</span>{" "}
            and no earnings within <span className="num">{threshold(pb, "entry.earnings.block_within_days", String)}</span> days.
            The model may add caution, never remove it.</li>
          <li><Status kind="pass" label="Missing evidence is never safe." /> Too little filing evidence gives
            INSUFFICIENT_EVIDENCE and an info item, not a trade.</li>
          <li><Status kind="pass" label="Retrieved text is data." /> Filing excerpts are shown as quoted plain text and
            never act as instructions or links.</li>
          <li><Status kind="pass" label="You approve every trade." /> Approve always opens a confirm step that names the
            ticker, strategy and credit. There is no bulk or automatic approval.</li>
          <li><Status kind="pass" label="Paper only, local only." /> Nothing is sent to a broker. The app runs on
            127.0.0.1 with the local model, no telemetry and no outside assets.</li>
        </ul>
      </section>

      <div className="home-grid">
        <section className="panel" aria-labelledby="tech-title">
          <h2 id="tech-title">Agentic techniques in this app</h2>
          <table className="tech">
            <tbody>{TECHNIQUES.map((t) => <tr key={t.name}><th scope="row">{t.name}</th><td className="muted">{t.where}</td></tr>)}</tbody>
          </table>
        </section>
        <section className="panel" aria-labelledby="keys-title">
          <h2 id="keys-title">Keyboard</h2>
          <dl className="shortcut-list">
            {SHORTCUTS.map((s) => <div key={s.keys}><dt><kbd>{s.keys}</kbd></dt><dd>{s.what}</dd></div>)}
          </dl>
        </section>
      </div>

      <section className="panel" aria-labelledby="gloss-title">
        <h2 id="gloss-title">Glossary</h2>
        <dl className="glossary">
          {GLOSSARY.map((g) => <div key={g.term}><dt>{g.term}</dt><dd>{g.meaning}</dd></div>)}
        </dl>
      </section>

      <section className="panel" aria-labelledby="data-title">
        <h2 id="data-title">Data and privacy</h2>
        <ul className="plain-list">
          <li>Market data from yfinance and filings from SEC EDGAR, both public and cached locally.</li>
          <li>Positions, tickets and the journal are synthetic paper records in a local SQLite file.</li>
          <li>Prompts and traces pass through a mask layer: account values become a share of the portfolio.</li>
          <li>Evaluation runs live in their own sandbox database and can never write a brief or a ticket.</li>
          {session && !session.databases.eval && <li className="muted">No evaluation database is present yet.</li>}
        </ul>
      </section>
    </div>
  );
}
