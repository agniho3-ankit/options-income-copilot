// The decision bar: approve (with a confirm step), reject (code + reason), defer (with a confirm step).
// Approval is a deliberate human act: nothing is sent until the owner presses the confirm button,
// and the request echoes the brief ID and ticker shown in the dialog.
import { useCallback, useState } from "react";
import { post, type BriefView } from "../lib/api";
import * as f from "../lib/format";
import { useShortcuts } from "../lib/keys";
import { Dialog } from "./Dialog";

type Mode = null | "approve" | "reject" | "defer";
type Result = { kind: "ok" | "error"; text: string };

export function confirmText(v: BriefView): { what: string; amount: string } {
  const { brief: b, payload: p } = v;
  if (b.kind === "exit") {
    return { what: `close ${b.ticker} ${f.words(p.strategy).toLowerCase()} (${f.words(p.exit_rule)})`,
      amount: `close cost ${f.money(p.close_cost)}` };
  }
  const prop = b.kind === "roll" ? p.new?.proposal : p.proposal;
  const n = b.kind === "roll" ? p.new?.numbers : p.numbers;
  return { what: `${b.kind === "roll" ? "roll " : ""}${b.ticker} ${f.words(prop?.strategy).toLowerCase()}`,
    amount: `credit ${f.money(n?.credit)}` };
}

export function DecisionBar({ view, onDecided, initial = null }: {
  view: BriefView; onDecided: (message: string) => void; initial?: Mode;
}) {
  const [mode, setMode] = useState<Mode>(initial);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [code, setCode] = useState<string>("");
  const [reason, setReason] = useState("");
  const { brief: b } = view;
  const close = useCallback(() => { if (!busy) setMode(null); }, [busy]);
  // A, R, D open the same confirm steps as the buttons; they never send anything themselves
  useShortcuts((key, e) => {
    const k = key.toLowerCase();
    if (k === "a" || k === "r" || k === "d") {
      e.preventDefault();
      setMode(k === "a" ? "approve" : k === "r" ? "reject" : "defer");
    }
  }, !busy);
  const c = confirmText(view);

  async function send(path: string, body: unknown, done: (r: Record<string, string>) => string) {
    setBusy(true);
    setResult(null);
    try {
      const r = await post<Record<string, string>>(`/api/briefs/${encodeURIComponent(b.brief_id)}/${path}`, body);
      setMode(null);
      onDecided(done(r));
    } catch (e) {
      setResult({ kind: "error", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  const canReject = code !== "" && reason.trim() !== "";
  const alert = result && <p className={result.kind === "error" ? "error" : "muted"} role="alert">{result.text}</p>;
  const inDialog = mode === "approve" || mode === "defer";

  return (
    <section className="panel decision" aria-label="Decision">
      <div className="decision-buttons">
        <button type="button" className="primary" onClick={() => setMode("approve")}>Approve paper trade</button>
        <button type="button" onClick={() => setMode(mode === "reject" ? null : "reject")}
          aria-expanded={mode === "reject"} aria-controls="reject-form">Reject with code</button>
        <button type="button" onClick={() => setMode("defer")}>Defer</button>
      </div>
      {!inDialog && alert}

      {mode === "reject" && (
        <form id="reject-form" className="reject-form" onSubmit={(e) => {
          e.preventDefault();
          if (canReject) void send("reject", { code, reason: reason.trim() },
            (r) => `Rejected with code ${r.code}. Lesson (${r.lesson_by}): ${r.lesson}`);
        }}>
          <fieldset>
            <legend>Reason code</legend>
            <div className="chips" role="radiogroup" aria-label="Reason code">
              {view.reject_codes.map((rc) => (
                <label key={rc} className={`chip${code === rc ? " chip-on" : ""}`}>
                  <input type="radio" name="reject-code" value={rc} checked={code === rc} onChange={() => setCode(rc)} />
                  {f.words(rc)}
                </label>
              ))}
            </div>
          </fieldset>
          <label className="field">
            <span>Reason (it becomes a lesson for future runs)</span>
            <textarea value={reason} onChange={(e) => setReason(e.target.value)} rows={3} maxLength={1000} required />
          </label>
          <div className="decision-buttons">
            <button type="submit" className="danger" disabled={!canReject || busy}>Reject brief</button>
            <button type="button" onClick={close}>Cancel</button>
          </div>
          {busy && <p className="muted" role="status">Recording the rejection and writing the lesson…</p>}
        </form>
      )}

      {mode === "approve" && (
        <Dialog title="Approve paper trade" onClose={close}>
          <p>Write a <strong>paper ticket</strong> to {c.what}, <span className="num">{c.amount}</span>?</p>
          <p className="muted">Brief <span className="num">{b.brief_id}</span>. Nothing is sent to a broker: the
            paper broker simulates the fill.</p>
          {alert}
          <div className="decision-buttons">
            <button type="button" data-autofocus onClick={close} disabled={busy}>Cancel</button>
            <button type="button" className="primary" disabled={busy}
              onClick={() => void send("approve", { brief_id: b.brief_id, ticker: b.ticker },
                (r) => `Approved. Paper ticket ${r.ticket_id} written; nothing was sent to any broker.`)}>
              Write paper ticket
            </button>
          </div>
        </Dialog>
      )}

      {mode === "defer" && (
        <Dialog title="Defer this brief" onClose={close}>
          <p>Defer {b.ticker} {b.brief_id}? It stays pending and moves to the end of the queue. Nothing is approved.</p>
          {alert}
          <div className="decision-buttons">
            <button type="button" data-autofocus onClick={close} disabled={busy}>Cancel</button>
            <button type="button" disabled={busy}
              onClick={() => void send("defer", { brief_id: b.brief_id }, () => "Deferred. It stays pending at the end of the queue.")}>
              Defer brief
            </button>
          </div>
        </Dialog>
      )}
    </section>
  );
}
