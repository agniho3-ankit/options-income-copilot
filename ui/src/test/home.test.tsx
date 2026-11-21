import { render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { parse } from "../lib/route";
import { Home } from "../screens/Home";
import { QUEUE } from "./fixtures";

const PB = { version: 6, sha256: "aa60459b91d888a4", source: "playbook.yaml", editable_by: "owner only, in the file",
  key_thresholds: [
    { path: "entry.pop_min", label: "", unit: "fraction", value: 0.65 },
    { path: "entry.position_risk_pct.block", label: "", unit: "%", value: 5 },
    { path: "entry.portfolio_heat_pct.block", label: "", unit: "%", value: 30 },
    { path: "entry.earnings.block_within_days", label: "", unit: "days", value: 7 }], tree: {} };

let calls: { path: string; method: string }[];
beforeEach(() => {
  calls = [];
  window.location.hash = "";
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    calls.push({ path, method: init?.method ?? "GET" });
    const body: Record<string, unknown> = {
      "/api/session": { csrf_token: "t", paper_only: true, databases: { main: true, eval: true },
        playbook: { version: 6, sha256: "aa60459b91d888a4" } },
      "/api/queue": QUEUE, "/api/playbook": PB, "/api/jobs": { active: null, recent: [] },
      "/api/lab": { after_floor: { unsafe_passed: 0 } },
    };
    return path in body ? new Response(JSON.stringify(body[path])) : new Response("{}", { status: 404 });
  }));
});
afterEach(() => vi.unstubAllGlobals());

describe("routing", () => {
  it("the app opens on the homepage and the review queue lives at #/today", () => {
    expect(parse("")).toEqual({ screen: "home" });
    expect(parse("#/")).toEqual({ screen: "home" });
    expect(parse("#/nope")).toEqual({ screen: "home" });
    expect(parse("#/today")).toEqual({ screen: "today", filter: null });
    expect(parse("#/today/msft")).toEqual({ screen: "today", filter: "MSFT" });
  });

  it("the brand is a link to the homepage", async () => {
    render(<App />);
    const brand = screen.getByRole("link", { name: "Options Income Copilot" });
    expect(brand).toHaveAttribute("href", "#/");
    expect(brand).toHaveAttribute("aria-current", "page");
    expect(await screen.findByRole("heading", { level: 1, name: "Options Income Copilot" })).toBeInTheDocument();
  });
});

describe("Home", () => {
  it("describes and links every tab", async () => {
    render(<Home />);
    const tabs = await screen.findByRole("region", { name: "The tabs" });
    const links = within(tabs).getAllByRole("link").map((a) => a.getAttribute("href"));
    expect(links).toEqual(["#/today", "#/review", "#/explore", "#/positions", "#/journal", "#/lab", "#/playbook"]);
  });

  it("shows live status from the API and thresholds from the loaded playbook", async () => {
    render(<Home />);
    const glance = await screen.findByRole("region", { name: "At a glance" });
    await within(glance).findByText("3");                       // three non-deferred briefs in the fixture
    expect(within(glance).getByText("1 escalated")).toBeInTheDocument();
    expect(within(glance).getByText("v6")).toBeInTheDocument();
    const rules = screen.getByRole("region", { name: "Rules that cannot be bent" });
    expect(await within(rules).findByText("65%")).toBeInTheDocument();
    expect(within(rules).getByText("30%")).toBeInTheDocument();
  });

  it("says it is educational and paper only, and changes nothing", async () => {
    render(<Home />);
    expect(screen.getByRole("note")).toHaveTextContent("Not financial advice. Paper trading only");
    await screen.findByText("v6");
    expect(calls.every((c) => c.method === "GET")).toBe(true);
  });
});
