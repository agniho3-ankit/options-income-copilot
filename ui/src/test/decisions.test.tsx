// Approval stays a deliberate human act: the confirm step blocks approve, reject needs a code and a
// reason, defer confirms, and every POST carries the session's CSRF token.
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DecisionBar, confirmText } from "../components/DecisionBar";
import { resetSessionForTests, type BriefView } from "../lib/api";

const VIEW: BriefView = {
  brief: { brief_id: "brf-1", run_id: "run-1", kind: "entry", ticker: "MSFT", status: "pending", verdict: "CAUTION",
    created_at: "2026-11-19", decided_at: null, decision_reason: null, decision_code: null, deferred_at: null },
  payload: { proposal: { strategy: "call_credit_spread", legs: [] }, numbers: { credit: 347.5 } },
  review: { score: 9, tier: "escalated", signals: {}, top: [] },
  guardrail_strip: [], alternatives: [],
  reject_codes: ["event_risk", "too_close_to_resistance", "low_credit", "bad_citation", "other"],
};

type Call = { path: string; method: string; headers: Record<string, string>; body: unknown };
let calls: Call[];

beforeEach(() => {
  resetSessionForTests();
  calls = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    const method = init?.method ?? "GET";
    calls.push({ path, method, headers: (init?.headers ?? {}) as Record<string, string>,
      body: init?.body ? JSON.parse(String(init.body)) : undefined });
    const reply = (b: unknown, status = 200) => new Response(JSON.stringify(b), { status });
    if (path === "/api/session") return reply({ csrf_token: "tok-123", paper_only: true });
    if (path.endsWith("/approve")) return reply({ status: "approved", ticket_id: "tkt-9", paper_only: true });
    if (path.endsWith("/reject")) return reply({ status: "rejected", code: "event_risk", lesson: "L", lesson_by: "local LLM" });
    if (path.endsWith("/defer")) return reply({ status: "pending", deferred: true });
    return reply({ detail: "no route" }, 404);
  }));
});
afterEach(() => vi.unstubAllGlobals());

const posts = () => calls.filter((c) => c.method === "POST");

describe("approve", () => {
  it("opening approve sends nothing; the dialog names ticker, strategy and credit", () => {
    render(<DecisionBar view={VIEW} onDecided={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Approve paper trade" }));
    const dialog = screen.getByRole("dialog", { name: "Approve paper trade" });
    expect(dialog).toHaveTextContent("paper ticket");
    expect(dialog).toHaveTextContent("MSFT call credit spread");
    expect(dialog).toHaveTextContent("credit $347.50");
    expect(posts()).toHaveLength(0);
  });

  it("cancel and Escape close the dialog without sending anything", () => {
    render(<DecisionBar view={VIEW} onDecided={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Approve paper trade" }));
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Approve paper trade" }));
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(posts()).toHaveLength(0);
  });

  it("the dialog's default focus is Cancel, so Enter alone cannot approve", () => {
    render(<DecisionBar view={VIEW} onDecided={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Approve paper trade" }));
    expect(document.activeElement).toHaveTextContent("Cancel");
  });

  it("only the confirm button posts, with the CSRF token and the echoed brief", async () => {
    const done = vi.fn();
    render(<DecisionBar view={VIEW} onDecided={done} />);
    fireEvent.click(screen.getByRole("button", { name: "Approve paper trade" }));
    fireEvent.click(screen.getByRole("button", { name: "Write paper ticket" }));
    await waitFor(() => expect(done).toHaveBeenCalled());
    expect(posts()).toHaveLength(1);
    const [p] = posts();
    expect(p.path).toBe("/api/briefs/brf-1/approve");
    expect(p.headers["X-CSRF-Token"]).toBe("tok-123");
    expect(p.body).toEqual({ brief_id: "brf-1", ticker: "MSFT" });
    expect(done.mock.calls[0][0]).toContain("Paper ticket tkt-9");
  });

  it("exit briefs name the close and the close cost", () => {
    const exit = { ...VIEW, brief: { ...VIEW.brief, kind: "exit", verdict: "EXIT" },
      payload: { strategy: "put_credit_spread", exit_rule: "r3_stop_loss", close_cost: 487.5 } };
    expect(confirmText(exit)).toEqual({ what: "close MSFT put credit spread (R3 stop loss)", amount: "close cost $487.50" });
  });
});

describe("reject", () => {
  it("needs a code and a reason before it can send", async () => {
    const done = vi.fn();
    render(<DecisionBar view={VIEW} onDecided={done} />);
    fireEvent.click(screen.getByRole("button", { name: "Reject with code" }));
    const submit = screen.getByRole("button", { name: "Reject brief" });
    expect(submit).toBeDisabled();
    fireEvent.click(screen.getByRole("radio", { name: "Event risk" }));
    expect(submit).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "   " } });
    expect(submit).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "8-K deal closes before expiry" } });
    expect(submit).toBeEnabled();
    expect(posts()).toHaveLength(0);
    fireEvent.click(submit);
    await waitFor(() => expect(done).toHaveBeenCalled());
    expect(posts()[0].body).toEqual({ code: "event_risk", reason: "8-K deal closes before expiry" });
    expect(posts()[0].headers["X-CSRF-Token"]).toBe("tok-123");
  });

  it("offers exactly the playbook's reject codes", () => {
    render(<DecisionBar view={VIEW} onDecided={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Reject with code" }));
    expect(screen.getAllByRole("radio").map((r) => (r as HTMLInputElement).value)).toEqual(VIEW.reject_codes);
  });
});

describe("defer", () => {
  it("confirms, then posts and keeps the brief pending", async () => {
    const done = vi.fn();
    render(<DecisionBar view={VIEW} onDecided={done} />);
    fireEvent.click(screen.getByRole("button", { name: "Defer" }));
    expect(screen.getByRole("dialog")).toHaveTextContent("stays pending");
    expect(posts()).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Defer brief" }));
    await waitFor(() => expect(done).toHaveBeenCalled());
    expect(posts().map((p) => [p.path, p.body])).toEqual([["/api/briefs/brf-1/defer", { brief_id: "brf-1" }]]);
  });
});

describe("errors", () => {
  it("a refused approve shows the server's reason and keeps the brief open", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => String(input) === "/api/session"
      ? new Response(JSON.stringify({ csrf_token: "t" }))
      : new Response(JSON.stringify({ detail: "brief brf-1 is approved, not pending" }), { status: 409 })));
    const done = vi.fn();
    render(<DecisionBar view={VIEW} onDecided={done} />);
    fireEvent.click(screen.getByRole("button", { name: "Approve paper trade" }));
    fireEvent.click(screen.getByRole("button", { name: "Write paper ticket" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("is approved, not pending");
    expect(done).not.toHaveBeenCalled();
  });
});
