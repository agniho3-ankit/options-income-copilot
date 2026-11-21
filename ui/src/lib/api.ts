// The only network code in the UI: same-origin calls to the local API (/api, proxied in dev).
// State-changing calls carry the per-session CSRF token from GET /api/session.

export type Leg = { side: string | null; option_type: string | null; strike: number | null; expiry: string | null };
export type TradeSummary = {
  strategy: string | null; legs: Leg[]; expiry: string | null;
  exit_rule?: string | null; action?: string | null; dte?: number | null;
  new?: { strategy: string | null; legs: Leg[]; expiry: string | null };
};
export type Tier = "escalated" | "look_closely" | "standard";
export type QueueTrade = {
  brief_id: string; ticker: string; kind: string; verdict: string | null; created_at: string;
  deferred: boolean; score: number; tier: Tier; top: { signal: string; text: string }[];
  signals: Record<string, number>; trade: TradeSummary;
};
export type InfoItem = { run_id: string; ticker: string; kind: string; outcome: string; when: string; why: string; next: string };
export type Queue = { trades: QueueTrade[]; info: InfoItem[]; heat: { pct: number; cap_pct: number; warn_pct: number } };
export type Session = {
  csrf_token: string; paper_only: true; databases: { main: boolean; eval: boolean };
  playbook: { version: number; sha256: string };
};

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

let session: Promise<Session> | null = null;

export function getSession(): Promise<Session> {
  session ??= get<Session>("/api/session");
  return session;
}

export function resetSessionForTests(): void {
  session = null;
}

async function read<T>(r: Response): Promise<T> {
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail = typeof body?.detail === "string" ? body.detail : `request failed (${r.status})`;
    throw new ApiError(r.status, detail);
  }
  return body as T;
}

export async function get<T>(path: string): Promise<T> {
  return read<T>(await fetch(path, { credentials: "same-origin", headers: { Accept: "application/json" } }));
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  const { csrf_token } = await getSession();
  return read<T>(await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "application/json", "X-CSRF-Token": csrf_token },
    body: JSON.stringify(body),
  }));
}

// --- Review ------------------------------------------------------------------------------

export type Check = { name: string; verdict: string; reason: string; hard?: boolean };
export type Citation = { form?: string; date?: string; item?: string; url?: string };
export type Alternative = { node?: string; path?: string; status?: string; score: number; reason?: string };
export type StripEvent = { seq: number; decision: string; signal: string | null; reason: string | null };
export type StripItem = { name: string; stage: string; kind: string; what: string; decision: string | null; events: StripEvent[] };
export type Numbers = { credit?: number; max_loss?: number; pop?: number; breakevens?: number[]; risk_pct_of_portfolio?: number };
export type JudgeFlag = { claim: string; issue: string; conflicts_with?: string };

export type BriefView = {
  brief: { brief_id: string; run_id: string | null; kind: string; ticker: string; status: string; verdict: string | null;
    created_at: string; decided_at: string | null; decision_reason: string | null; decision_code: string | null;
    deferred_at: string | null };
  payload: Record<string, unknown> & {
    as_of?: string; numbers?: Numbers; proposal?: { strategy?: string; legs?: Leg[]; expiry?: string };
    verifier?: Check[]; risks?: string[]; judge?: { flags?: JudgeFlag[]; note?: string }; summary?: string;
    landmine_verdict?: string; strategy?: string; legs?: Leg[]; exit_rule?: string; close_cost?: number;
    est_pnl?: number; profit_pct?: number; dte?: number; entry_credit?: number;
    new?: { proposal?: { strategy?: string; legs?: Leg[]; expiry?: string }; numbers?: Numbers };
  };
  review: { score: number; tier: Tier; signals: Record<string, number>; top: { signal: string; points: number; text: string }[] };
  guardrail_strip: StripItem[];
  alternatives: Alternative[];
  reject_codes: string[];
};

export type Evidence = {
  brief_id: string; landmine_verdict: string | null; citations: Citation[];
  excerpts: { id: string; form: string | null; item: string | null; text: string }[]; note: string | null;
};

export type WindowEvent = { date: string; kind: string; label: string; source: string; in_window: boolean };
export type TradeWindow = {
  ticker: string; kind: string; strategy: string | null; brief_as_of: string; today: string; expiry: string; spot: number;
  sigma: { iv: number | null; source: string; convention: string; at_expiry: { date: string; low: number; high: number } | null };
  history: { date: string; close: number }[];
  cone: { date: string; low: number; high: number }[];
  strikes: { side: string | null; option_type: string | null; strike: number | null }[];
  breakevens: number[];
  zones: { kind: "profit" | "loss"; low: number | null; high: number | null }[];
  events: WindowEvent[];
  x_domain: [string, string]; y_domain: [number, number];
  freshness: { stale: boolean; age_s: number; error?: string };
};

// --- Explore -------------------------------------------------------------------------------

export type Span = {
  run_id: string; span_id: string; parent_id: string | null; agent: string; pattern: string | null;
  started_at: string; ended_at: string | null; duration_ms: number | null; llm_calls: number;
  decision?: string | null; error?: string | null;
};
export type GuardEvent = {
  run_id: string; seq: number; span_id: string | null; guardrail: string; stage: string; decision: string;
  signal: string | null; reason: string | null; created_at: string;
};
export type JobResult = { run_id?: string; outcome?: string; brief_id?: string | null; tot_run_id?: string | null;
  reasons?: string[]; llm_calls?: number; latency_ms?: number; summary?: string };
export type Job = {
  job_id: string; kind: "multi_agent" | "tot"; ticker: string; as_of: string; status: "running" | "done" | "failed";
  created_at: string; finished_at: string | null; runs: Record<string, string>; result: JobResult | null; error: string | null;
};
export type RunRow = { run_id: string; kind: string; ticker: string; started_at: string; finished_at: string | null; outcome: string | null };
export type RunView = {
  run: RunRow & Record<string, unknown>; spans: Span[]; guardrail_events: GuardEvent[];
  tot_run_id: string | null; brief_id: string | null; reasons?: string[];
};
