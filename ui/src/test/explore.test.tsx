import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Timeline } from "../components/Timeline";
import { TotTree } from "../components/TotTree";
import { resetSessionForTests, type Job, type Span } from "../lib/api";
import type { TotNode } from "../lib/tree";
import { Explore } from "../screens/Explore";

class FakeEventSource {
  static last: FakeEventSource | null = null;
  static CLOSED = 2;
  readyState = 1;
  onerror: (() => void) | null = null;
  listeners = new Map<string, ((e: MessageEvent) => void)[]>();
  constructor(public url: string) { FakeEventSource.last = this; }
  addEventListener(kind: string, fn: (e: MessageEvent) => void) {
    this.listeners.set(kind, [...(this.listeners.get(kind) ?? []), fn]);
  }
  close() { this.readyState = 2; }
  emit(kind: string, data: unknown) {
    act(() => { for (const fn of this.listeners.get(kind) ?? []) fn(new MessageEvent(kind, { data: JSON.stringify(data) })); });
  }
}

const JOB: Job = { job_id: "job-1", kind: "multi_agent", ticker: "MSFT", as_of: "2026-11-20", status: "running",
  created_at: "2026-11-20T10:00:00Z", finished_at: null, runs: {}, result: null, error: null };
const span = (id: string, agent: string, start: string, end: string | null, calls = 0): Span => ({
  run_id: "run-1", span_id: id, parent_id: null, agent, pattern: "one-way", started_at: start, ended_at: end,
  duration_ms: end ? 1000 : null, llm_calls: calls });

let posts: { path: string; body: unknown; headers: Record<string, string> }[];

beforeEach(() => {
  resetSessionForTests();
  posts = [];
  FakeEventSource.last = null;
  vi.stubGlobal("EventSource", FakeEventSource);
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    const reply = (b: unknown, status = 200) => new Response(JSON.stringify(b), { status });
    if (init?.method === "POST") {
      posts.push({ path, body: JSON.parse(String(init.body)), headers: init.headers as Record<string, string> });
      return reply(JOB);
    }
    if (path === "/api/session") return reply({ csrf_token: "tok" });
    if (path === "/api/jobs") return reply({ active: null, recent: [] });
    if (path.startsWith("/api/runs?")) return reply({ runs: [] });
    return reply({ detail: "no route" }, 404);
  }));
});
afterEach(() => vi.unstubAllGlobals());

describe("Explore live run", () => {
  it("starts a run with the CSRF token and streams spans, guardrails and the outcome", async () => {
    render(<Explore runId={null} />);
    fireEvent.change(screen.getByLabelText("Ticker"), { target: { value: "msft" } });
    fireEvent.click(screen.getByRole("button", { name: "Start run" }));
    await waitFor(() => expect(FakeEventSource.last).not.toBeNull());
    expect(posts[0]).toMatchObject({ path: "/api/runs", body: { ticker: "MSFT", kind: "multi_agent" } });
    expect(posts[0].headers["X-CSRF-Token"]).toBe("tok");
    expect(FakeEventSource.last!.url).toBe("/api/jobs/job-1/events");
    expect(screen.getByRole("button", { name: "Start run" })).toBeDisabled();   // one run at a time

    const es = FakeEventSource.last!;
    es.emit("span", span("s1", "researcher", "2026-11-20T10:00:01Z", null));
    expect(document.querySelector("[data-span=s1]")).toHaveAttribute("data-open", "true");
    es.emit("span", span("s1", "researcher", "2026-11-20T10:00:01Z", "2026-11-20T10:00:09Z", 3));
    expect(document.querySelector("[data-span=s1]")).not.toHaveAttribute("data-open");
    expect(document.querySelectorAll("[data-span]")).toHaveLength(1);           // updated, not duplicated
    es.emit("guardrail", { run_id: "run-1", seq: 1, span_id: "s1", guardrail: "grader_rewrite", stage: "pre",
      decision: "flag", signal: "grader_zero_relevant", reason: "no relevant passage", created_at: "2026-11-20T10:00:05Z" });
    expect(document.querySelector("[data-guardrail=grader_rewrite]")).toHaveTextContent("▲");
    es.emit("done", { ...JOB, status: "done", finished_at: "2026-11-20T10:02:00Z",
      result: { run_id: "run-1", outcome: "SELECTED", brief_id: "brf-9", llm_calls: 11 } });
    expect(screen.getByRole("link", { name: "Open brief brf-9 in Review" })).toHaveAttribute("href", "#/review/brf-9");
    expect(es.readyState).toBe(2);
  });

  it("a no-evidence outcome points to the info item on Today", async () => {
    render(<Explore runId={null} />);
    fireEvent.change(screen.getByLabelText("Ticker"), { target: { value: "AAPL" } });
    fireEvent.click(screen.getByRole("button", { name: "Start run" }));
    await waitFor(() => expect(FakeEventSource.last).not.toBeNull());
    FakeEventSource.last!.emit("done", { ...JOB, status: "done",
      result: { run_id: "run-2", outcome: "INSUFFICIENT_EVIDENCE", brief_id: null, reasons: ["0 of 20 chunks passed"] } });
    expect(screen.getByText(/info item on/)).toBeInTheDocument();
    expect(screen.getByText("0 of 20 chunks passed")).toBeInTheDocument();
  });
});

describe("Timeline", () => {
  it("draws one lane per agent and a shape for each guardrail decision", () => {
    render(<Timeline now={Date.parse("2026-11-20T10:01:00Z")} spans={[
      span("a", "supervisor", "2026-11-20T10:00:00Z", "2026-11-20T10:00:00Z"),
      span("b", "researcher", "2026-11-20T10:00:00Z", "2026-11-20T10:00:30Z", 3),
      span("c", "strategist", "2026-11-20T10:00:30Z", null, 2),
    ]} events={[{ run_id: "run-1", seq: 1, span_id: "c", guardrail: "verifier", stage: "post", decision: "block",
      signal: null, reason: "PoP 60%", created_at: "2026-11-20T10:00:40Z" }]} />);
    const svg = screen.getByRole("img");
    expect(svg).toHaveAccessibleName("Supervisor timeline: 3 spans across 3 agents");
    expect(within(svg).getByText("Researcher")).toBeInTheDocument();
    expect(document.querySelector("[data-span=c]")).toHaveAttribute("data-open", "true");
    expect(document.querySelector("[data-guardrail=verifier]")).toHaveTextContent("■");
  });
});

const NODES: TotNode[] = [
  { id: "n0", parent_id: null, depth: 0, status: "open", score: null, prune_reason: "", source: "root",
    spec: { strategy: null, expiry: null, target_delta: null }, detail: {} },
  { id: "n1", parent_id: "n0", depth: 1, status: "open", score: null, prune_reason: "", source: "llm",
    spec: { strategy: "call_credit_spread", expiry: null, target_delta: null },
    detail: { thin_expiries: ["2026-12-31: in-band median open interest 43 below 500"] } },
  { id: "n2", parent_id: "n1", depth: 2, status: "open", score: 0.72, prune_reason: "", source: "llm",
    spec: { strategy: "call_credit_spread", expiry: "2027-01-15", target_delta: null }, detail: {} },
  { id: "n3", parent_id: "n2", depth: 3, status: "selected", score: 0.402, prune_reason: "", source: "llm",
    spec: { strategy: "call_credit_spread", expiry: "2027-01-15", target_delta: 0.28 },
    detail: { numbers: { credit: 347.5, pop: 0.7829, max_loss: 1152.5 }, verdict: "CAUTION" } },
  { id: "n4", parent_id: "n2", depth: 3, status: "pruned", score: 0.31, prune_reason: "open_interest: open interest 120 below 500",
    source: "llm", spec: { strategy: "call_credit_spread", expiry: "2027-01-15", target_delta: 0.2 },
    detail: { critic: { action: "lower_major", reason: "ex-dividend inside the window" } } },
];

describe("ToT tree", () => {
  it("greys pruned nodes with their reason, marks critic lowers and withholds, highlights the selected leaf", () => {
    render(<TotTree nodes={NODES} />);
    const pruned = document.querySelector("[data-node=n4]")!;
    expect(pruned).toHaveAttribute("data-status", "pruned");
    expect(pruned.querySelector("title")!.textContent).toContain("open interest 120 below 500");
    expect(pruned).toHaveTextContent("▲ critic lower major");
    expect(document.querySelector("[data-node=n1]")).toHaveTextContent("▲ 1 thin expiry withheld");
    expect(document.querySelector("[data-node=n3]")).toHaveClass("tot-selected");
    expect(screen.getByText(/^Selected/, { selector: "p" })).toHaveTextContent("call credit spread / 2027-01-15 / Δ 0.28");
    expect(screen.getByRole("region", { name: "Node n3 details" })).toHaveTextContent("credit $347.50");
  });

  it("clicking a node shows its stored details", () => {
    render(<TotTree nodes={NODES} />);
    fireEvent.click(document.querySelector("[data-node=n4]")!);
    const d = screen.getByRole("region", { name: "Node n4 details" });
    expect(d).toHaveTextContent("Pruned: open_interest: open interest 120 below 500");
    expect(d).toHaveTextContent("ex-dividend inside the window");
  });
});
