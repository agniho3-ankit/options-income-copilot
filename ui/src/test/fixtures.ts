import type { Queue, QueueTrade } from "../lib/api";
import { vi } from "vitest";

export function trade(over: Partial<QueueTrade>): QueueTrade {
  return {
    brief_id: "brf-x", ticker: "MSFT", kind: "entry", verdict: "CAUTION", created_at: "2026-11-19T19:07:19+00:00",
    deferred: false, score: 9, tier: "escalated", signals: {},
    top: [{ signal: "judge_flag", text: "the judge found a summary claim the facts do not support" }],
    trade: {
      strategy: "call_credit_spread", expiry: "2027-01-15",
      legs: [{ side: "short", option_type: "call", strike: 545, expiry: "2027-01-15" },
        { side: "long", option_type: "call", strike: 560, expiry: "2027-01-15" }],
    },
    ...over,
  };
}

export const QUEUE: Queue = {
  trades: [
    trade({ brief_id: "brf-esc", ticker: "META", score: 11, tier: "escalated" }),
    trade({ brief_id: "brf-look", ticker: "BAC", score: 5, tier: "look_closely", verdict: "CLEAR" }),
    trade({ brief_id: "brf-std", ticker: "MSFT", score: 0, tier: "standard", verdict: "CLEAR", top: [] }),
    trade({ brief_id: "brf-def", ticker: "AAPL", score: 7, tier: "escalated", deferred: true }),
  ],
  info: [{ run_id: "run-ie", ticker: "AAPL", kind: "multi_agent", outcome: "INSUFFICIENT_EVIDENCE",
    when: "2026-11-16T16:00:00+00:00", why: "0 of 20 chunks passed",
    next: "no trade; run `ingest TICKER` for recent filings, then re-run" }],
  heat: { pct: 1.267, cap_pct: 30, warn_pct: 20 },
};

export function mockFetch(routes: Record<string, unknown>) {
  const fn = vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (!(path in routes)) return new Response(JSON.stringify({ detail: "no route" }), { status: 404 });
    return new Response(JSON.stringify(routes[path]), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}
