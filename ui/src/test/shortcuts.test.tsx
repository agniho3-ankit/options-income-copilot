// Keyboard shortcuts open the same confirm steps as the buttons and never skip them; plus the
// accessibility basics: skip link, named controls, reduced motion.
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { DecisionBar } from "../components/DecisionBar";
import { resetSessionForTests, type BriefView } from "../lib/api";
import { Today } from "../screens/Today";
import { QUEUE } from "./fixtures";

const VIEW: BriefView = {
  brief: { brief_id: "brf-1", run_id: null, kind: "entry", ticker: "MSFT", status: "pending", verdict: "CLEAR",
    created_at: "", decided_at: null, decision_reason: null, decision_code: null, deferred_at: null },
  payload: { proposal: { strategy: "call_credit_spread", legs: [] }, numbers: { credit: 347.5 } },
  review: { score: 0, tier: "standard", signals: {}, top: [] }, guardrail_strip: [], alternatives: [],
  reject_codes: ["event_risk", "other"],
};

let posts: string[];
beforeEach(() => {
  resetSessionForTests();
  posts = [];
  window.location.hash = "";
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    if (init?.method === "POST") posts.push(path);
    if (path === "/api/session") return new Response(JSON.stringify({ csrf_token: "t" }));
    if (path === "/api/queue") return new Response(JSON.stringify(QUEUE));
    return new Response(JSON.stringify({ status: "approved", ticket_id: "tkt" }));
  }));
});
afterEach(() => vi.unstubAllGlobals());

describe("decision shortcuts", () => {
  it("A opens the approve confirm step and Enter alone does not approve", async () => {
    const user = userEvent.setup();
    render(<DecisionBar view={VIEW} onDecided={() => {}} />);
    await user.keyboard("a");
    const dialog = screen.getByRole("dialog", { name: "Approve paper trade" });
    expect(dialog).toHaveTextContent("credit $347.50");
    await user.keyboard("{Enter}");                       // focus starts on Cancel
    expect(screen.queryByRole("dialog")).toBeNull();
    await user.keyboard("a{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(posts).toEqual([]);
  });

  it("R opens the reject form and D the defer confirm step; neither sends anything", async () => {
    const user = userEvent.setup();
    render(<DecisionBar view={VIEW} onDecided={() => {}} />);
    await user.keyboard("r");
    expect(screen.getByRole("button", { name: "Reject brief" })).toBeDisabled();
    await user.keyboard("d");
    expect(screen.getByRole("dialog", { name: "Defer this brief" })).toBeInTheDocument();
    await user.keyboard("a");                             // a shortcut does nothing while a dialog is open
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    expect(posts).toEqual([]);
  });

  it("shortcuts are ignored while typing the reject reason", async () => {
    const user = userEvent.setup();
    render(<DecisionBar view={VIEW} onDecided={() => {}} />);
    await user.click(screen.getByRole("button", { name: "Reject with code" }));
    await user.click(screen.getByRole("textbox", { name: /Reason/ }));
    await user.keyboard("bad data");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByRole("textbox", { name: /Reason/ })).toHaveValue("bad data");
    expect(posts).toEqual([]);
  });

  it("a held modifier key never triggers a shortcut", () => {
    render(<DecisionBar view={VIEW} onDecided={() => {}} />);
    fireEvent.keyDown(document, { key: "a", metaKey: true });
    fireEvent.keyDown(document, { key: "a", ctrlKey: true });
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("queue navigation", () => {
  it("J and K move through the cards in queue order and Enter opens the focused brief", async () => {
    const user = userEvent.setup();
    render(<Today />);
    await screen.findByTestId("summary");
    await user.keyboard("j");
    expect(document.activeElement).toHaveAttribute("data-brief", "brf-esc");
    await user.keyboard("jj");
    expect(document.activeElement).toHaveAttribute("data-brief", "brf-std");
    await user.keyboard("k");
    expect(document.activeElement).toHaveAttribute("data-brief", "brf-look");
    await user.keyboard("{Enter}");
    expect(window.location.hash).toBe("#/review/brf-look");
  });

  it("a ticker filter shows only that ticker", async () => {
    render(<Today filter="AAPL" />);
    await screen.findByTestId("summary");
    expect([...document.querySelectorAll("[data-brief]")].map((e) => e.getAttribute("data-brief"))).toEqual(["brf-def"]);
    expect(screen.getByText("0 of 20 chunks passed")).toBeInTheDocument();   // the AAPL info item
    expect(screen.getByRole("link", { name: "show all" })).toHaveAttribute("href", "#/today");
  });
});

describe("app shell", () => {
  it("? opens the shortcut sheet and / focuses the ticker search", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("summary");
    await user.keyboard("?");
    const sheet = screen.getByRole("dialog", { name: "Keyboard shortcuts" });
    expect(within(sheet).getByText("Open the approve confirm step (Review)")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    await user.keyboard("/");
    const search = screen.getByRole("searchbox", { name: "Search tickers" });
    expect(document.activeElement).toBe(search);
    await user.keyboard("meta{Enter}");
    expect(window.location.hash).toBe("#/today/META");
  });

  it("has a skip link to the main content and every control has a name", async () => {
    render(<App />);
    await screen.findByTestId("summary");
    const skip = screen.getByRole("link", { name: "Skip to content" });
    fireEvent.click(skip);
    expect(document.activeElement).toBe(document.getElementById("main"));
    for (const el of [...screen.getAllByRole("button"), ...screen.getAllByRole("link")]) {
      expect(el).toHaveAccessibleName();
    }
  });

});

describe("live updates are announced", () => {
  it("Today's loading state is a status region", async () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
    render(<Today />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading the review queue");
    await act(async () => {});
  });

  it("errors are alerts", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { status: 500 })));
    render(<Today />);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Could not load the review queue"));
  });
});
