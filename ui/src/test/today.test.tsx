import { render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { Today } from "../screens/Today";
import { mockFetch, QUEUE } from "./fixtures";

afterEach(() => vi.unstubAllGlobals());

describe("Today", () => {
  it("shows the summary line from the queue payload", async () => {
    mockFetch({ "/api/queue": QUEUE });
    render(<Today />);
    const summary = await screen.findByTestId("summary");
    expect(summary).toHaveTextContent("3 to review · 1 info · portfolio heat 1.3% of 30%");
  });

  it("orders lanes escalated, look closely, standard, info, deferred", async () => {
    mockFetch({ "/api/queue": QUEUE });
    render(<Today />);
    await screen.findByTestId("summary");
    const lanes = screen.getAllByRole("heading", { level: 2 }).map((h) => h.textContent?.replace(/\s*\d+$/, ""));
    expect(lanes).toEqual(["Escalated", "Look closely", "Standard", "Info, no trade", "Deferred"]);
    const order = [...document.querySelectorAll("[data-brief]")].map((e) => e.getAttribute("data-brief"));
    expect(order).toEqual(["brf-esc", "brf-look", "brf-std", "brf-def"]);
  });

  it("cards show the API's values unchanged", async () => {
    mockFetch({ "/api/queue": QUEUE });
    render(<Today />);
    const card = (await screen.findByText("META")).closest("a")!;
    expect(card).toHaveAttribute("href", "#/review/brf-esc");
    const c = within(card);
    expect(c.getByText("score 11")).toBeInTheDocument();
    expect(c.getByText("short call 545 · long call 560 · exp 2027-01-15")).toBeInTheDocument();
    expect(c.getByText("Call credit spread")).toBeInTheDocument();
    expect(c.getByText("the judge found a summary claim the facts do not support")).toBeInTheDocument();
  });

  it("info items show the reason and the next step", async () => {
    mockFetch({ "/api/queue": QUEUE });
    render(<Today />);
    expect(await screen.findByText("0 of 20 chunks passed")).toBeInTheDocument();
    expect(screen.getByText(/Next: no trade; run `ingest TICKER`/)).toBeInTheDocument();
  });

  it("shows an empty state and an error state", async () => {
    mockFetch({ "/api/queue": { trades: [], info: [], heat: { pct: 0, cap_pct: 30, warn_pct: 20 } } });
    const { unmount } = render(<Today />);
    expect(await screen.findByText(/No pending briefs/)).toBeInTheDocument();
    unmount();
    mockFetch({});
    render(<Today />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load the review queue");
  });
});

describe("App shell", () => {
  it("always shows the Paper only marker and a theme toggle", async () => {
    mockFetch({ "/api/queue": QUEUE });
    render(<App />);
    expect(screen.getByText("Paper only")).toBeVisible();
    expect(document.documentElement.dataset.theme).toBe("dark");
    screen.getByRole("button", { name: "Switch to light theme" }).click();
    expect(await screen.findByRole("button", { name: "Switch to dark theme" })).toBeInTheDocument();
    expect(document.documentElement.dataset.theme).toBe("light");
  });
});
