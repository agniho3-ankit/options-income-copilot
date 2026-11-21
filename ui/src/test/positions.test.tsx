import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetSessionForTests, type Journal as JournalData, type Positions as PositionsData } from "../lib/api";
import { Journal } from "../screens/Journal";
import { Positions } from "../screens/Positions";

const POSITIONS: PositionsData = {
  as_of: "2026-11-20",
  heat: { pct: 5.28, cap_pct: 30, warn_pct: 20, gauge_pct: 17.6, over_cap: false, review: null },
  positions: [
    { position_id: "pos-1", ticker: "MSFT", strategy: "put_credit_spread", opened_on: "2026-11-01", expiry: "2026-12-18",
      contracts: 1, entry_credit: 233, entry_max_loss: 1267, pending_brief: "brf-exit",
      legs: [{ side: "short", option_type: "put", strike: 480 }, { side: "long", option_type: "put", strike: 465 }],
      spot: 484.52, close_cost: 487.5, profit_pct: -109.2, freshness: { stale: false, age_s: 0 },
      tracks: [
        { rule: "r1_profit_target", label: "Profit taken", value: "-109.2%", target: "50%", progress_pct: 0, state: "ok", action: null, reason: "" },
        { rule: "r2_roll", label: "Days to expiry", value: "28 DTE", target: "roll below 21", progress_pct: 73.1, state: "ok", action: null, reason: "" },
        { rule: "r3_stop_loss", label: "Loss vs credit", value: "1.09x", target: "stop at 1x", progress_pct: 100, state: "breach",
          action: "close", reason: "unrealized loss is 1.09x the credit" },
        { rule: "r6_iv_crush", label: "IV crush", value: "–", target: "", progress_pct: null, state: "unknown", action: null, reason: "" },
      ] },
    { position_id: "pos-2", ticker: "BAC", strategy: "iron_condor", opened_on: "2026-11-05", expiry: "2026-12-26",
      contracts: 1, entry_credit: 90, entry_max_loss: 410, pending_brief: null, legs: [], tracks: [],
      error: "live data unavailable (ConnectionError)" },
  ],
};

class FakeEventSource {
  static last: FakeEventSource | null = null;
  listeners = new Map<string, (e: MessageEvent) => void>();
  constructor(public url: string) { FakeEventSource.last = this; }
  addEventListener(k: string, fn: (e: MessageEvent) => void) { this.listeners.set(k, fn); }
  close() {}
}

let posts: { path: string; body: unknown; headers: Record<string, string> }[];
let positionLoads: number;

beforeEach(() => {
  resetSessionForTests();
  posts = [];
  positionLoads = 0;
  FakeEventSource.last = null;
  vi.stubGlobal("EventSource", FakeEventSource);
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    const reply = (b: unknown) => new Response(JSON.stringify(b));
    if (init?.method === "POST") {
      posts.push({ path, body: JSON.parse(String(init.body)), headers: init.headers as Record<string, string> });
      return reply({ job_id: "job-m", kind: "monitor", ticker: "ALL", status: "running", runs: {}, result: null, error: null,
        created_at: "", finished_at: null, as_of: "2026-11-20" });
    }
    if (path === "/api/session") return reply({ csrf_token: "tok" });
    if (path === "/api/positions") { positionLoads += 1; return reply(POSITIONS); }
    return new Response("{}", { status: 404 });
  }));
});
afterEach(() => vi.unstubAllGlobals());

describe("Positions", () => {
  it("draws each rule track from the API's progress and state", async () => {
    render(<Positions />);
    const stop = await waitFor(() => {
      const el = document.querySelector("[data-rule=r3_stop_loss]");
      expect(el).not.toBeNull();
      return el!;
    });
    expect(stop.querySelector(".track-fill")).toHaveStyle({ width: "100%" });
    expect(within(stop as HTMLElement).getByText("Breached")).toBeInTheDocument();
    expect(stop.querySelector("[data-status]")).toHaveAttribute("data-status", "block");
    expect(stop).toHaveTextContent("1.09x / stop at 1x");
    const roll = document.querySelector("[data-rule=r2_roll]")!;
    expect(roll.querySelector(".track-fill")).toHaveStyle({ width: "73.1%" });
    const iv = document.querySelector("[data-rule=r6_iv_crush]")!;
    expect(iv.querySelector(".track-fill")).toBeNull();                 // unknown: no bar, never a pass
    expect(iv.querySelector("[data-status]")).toHaveAttribute("data-status", "info");
  });

  it("shows heat against the cap from the API's gauge value", async () => {
    render(<Positions />);
    const meter = await screen.findByRole("meter");
    expect(meter).toHaveAccessibleName("Portfolio heat 5.3% of a 30% cap");
    expect(meter.querySelector(".gauge-fill")).toHaveStyle({ width: "17.6%" });
  });

  it("links a pending exit brief and shows a position whose data failed", async () => {
    render(<Positions />);
    expect(await screen.findByRole("link", { name: "Review brief brf-exit" })).toHaveAttribute("href", "#/review/brf-exit");
    expect(screen.getByRole("alert")).toHaveTextContent("live data unavailable (ConnectionError)");
  });

  it("Run monitor starts a background job and reloads positions when it finishes", async () => {
    render(<Positions />);
    fireEvent.click(await screen.findByRole("button", { name: "Run monitor" }));
    await waitFor(() => expect(FakeEventSource.last).not.toBeNull());
    expect(posts).toEqual([{ path: "/api/runs", body: { kind: "monitor" }, headers: expect.objectContaining({ "X-CSRF-Token": "tok" }) }]);
    expect(screen.getByRole("button", { name: "Monitor running…" })).toBeDisabled();
    const before = positionLoads;
    act(() => FakeEventSource.last!.listeners.get("done")!(new MessageEvent("done", { data: JSON.stringify({
      job_id: "job-m", kind: "monitor", status: "done", result: { items: [
        { position_id: "pos-1", ticker: "MSFT", action: "close", brief_id: "brf-new", note: "" }] } }) })));
    expect(await screen.findByRole("link", { name: "review brief brf-new" })).toBeInTheDocument();
    await waitFor(() => expect(positionLoads).toBe(before + 1));
  });
});

const JOURNAL: JournalData = {
  rows: [{ position_id: "p1", date_opened: "2026-10-01", ticker: "MSFT", strategy: "put_credit_spread", short_put: 450,
    short_call: null, long_put: 440, long_call: null, dte: 45, entry_credit: 200, entry_max_loss: 800, earnings_note: "No (2027-01-27)",
    date_closed: "2026-10-20", close_cost: 100, pnl: 100, pnl_pct: 50, return_on_risk_pct: 12.5, days_held: 19,
    annualized_pct: 960.5, exit_rule: "r1_profit_target", notes: "" }],
  all: { month: "all", total_trades: 1, closed_trades: 1, active_trades: 0, total_pnl: 100, win_rate_pct: 100,
    avg_days_held: 19, avg_return_pct: 50, annualized_return_pct: 960.5, avg_return_on_risk_pct: 12.5,
    exit_rules: { r1_profit_target: 1 } },
  months: [{ month: "2026-10", total_trades: 1, closed_trades: 1, active_trades: 0, total_pnl: 100, win_rate_pct: 100,
    avg_days_held: 19, avg_return_pct: 50, annualized_return_pct: 960.5, avg_return_on_risk_pct: 12.5,
    exit_rules: { r1_profit_target: 1 } }],
};

describe("Journal", () => {
  it("shows journal.py's summary values and the trade log, formatted only", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(JOURNAL))));
    render(<Journal />);
    const month = await screen.findByRole("article", { name: "Summary 2026-10" });
    const m = within(month);
    expect(m.getByText("$100.00")).toBeInTheDocument();
    expect(m.getByText("100.0%")).toBeInTheDocument();
    expect(m.getByText("12.50%")).toBeInTheDocument();
    expect(m.getByText("R1 profit target")).toBeInTheDocument();
    const row = screen.getByText("No (2027-01-27)").closest("tr")!;
    expect(row).toHaveTextContent("440 / 450");
    expect(row).toHaveTextContent("12.5%");
  });
});
