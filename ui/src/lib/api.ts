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
