import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Lab as LabData, Metrics, PlaybookView } from "../lib/api";
import { Lab } from "../screens/Lab";
import { Playbook } from "../screens/Playbook";

const LAB: LabData = {
  after_floor: { cases: 5, runs: 15, unsafe_passed: 0, false_blocks: 0, outcome_accuracy: 1, verdict_accuracy: 0.9333,
    latency_p50_s: 156.2 },
  before_floor: { cases: 5, runs: 15, unsafe_passed: 0, false_blocks: 9, outcome_accuracy: 0.4 },
  before_note: "rescored with the owner's relabel", repeats: 3,
  calibration: { confidence: { n: 12, ece: 0.025, brier: 0.0725, reliability: [
    { bin: "0.6-0.8", n: 0, mean_conf: null, accuracy: null },
    { bin: "0.8-1.0", n: 12, mean_conf: 0.892, accuracy: 0.917 }] } },
  cases: [{ case: "aapl_coverage", runs: 3, verdicts: { INSUFFICIENT_EVIDENCE: 3 }, outcomes: { INSUFFICIENT_EVIDENCE: 3 },
    unsafe_passed: 0, false_blocks: 0 }],
  ablation: {
    columns: [{ key: "false_blocks", label: "False blocks", higher_is_better: false },
      { key: "llm_calls_mean", label: "LLM calls per run", higher_is_better: false }],
    rows: [
      { config: "full", cells: { false_blocks: { value: 3, delta: null, direction: "baseline", level: 0 },
        llm_calls_mean: { value: 7.2, delta: null, direction: "baseline", level: 0 } } },
      { config: "-verifier", cells: { false_blocks: { value: 1, delta: -2, direction: "better", level: 3 },
        llm_calls_mean: { value: 10.2, delta: 3, direction: "worse", level: 2 } } },
    ],
    baseline: "full", note: "Read '-verifier' with care.",
  },
  sources: { eval: "eval_2026-11-16_x3.json", before_floor: "eval_2026-11-16_x3_before_floor.json", ablation: null, stored_evals: [] },
  run: { warning: "Hours for the whole set.", eval_command: "python -m options_copilot.cli eval --repeats 3 --save",
    ablation_command: "python -m options_copilot.cli eval --ablation --repeats 1 --save" },
};

const metrics = (db: string): Metrics => ({
  db, since: null, runs_by_outcome: {}, escalation_rate: {}, review_items: 0,
  guardrails: { verifier: { pass: 10, block: 2 } }, latest_eval_unsafe_passed: null,
  latency_per_agent: { researcher: { n: 4, p50_s: 77.1, p95_s: 90.2 } },
  llm_calls_per_run: { runs: 2, mean: 10.3, p95: 12 }, claude_fallbacks: 0, span_errors: 0,
  reject_codes: db === "main" ? { event_risk: 2, low_credit: 1 } : {}, judge_flag_rate: null,
  landmine_confidence: { n: 0, bins: {} },
});

const PB: PlaybookView = {
  version: 6, sha256: "aa60459b91d8", source: "playbook.yaml", editable_by: "owner only, in the file",
  key_thresholds: [{ path: "entry.pop_min", label: "Minimum probability of profit", unit: "fraction", value: 0.65 }],
  tree: { version: 6, universe: { tier_1: ["MRVL", "ANET"] }, entry: { pop_min: 0.65, earnings: { block_within_days: 7 } } },
};

let calls: { path: string; method: string }[];
beforeEach(() => {
  calls = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    calls.push({ path, method: init?.method ?? "GET" });
    if (path === "/api/lab") return new Response(JSON.stringify(LAB));
    if (path.startsWith("/api/metrics?db=")) return new Response(JSON.stringify(metrics(path.split("=")[1])));
    if (path === "/api/playbook") return new Response(JSON.stringify(PB));
    return new Response("{}", { status: 404 });
  }));
});
afterEach(() => vi.unstubAllGlobals());

describe("Lab", () => {
  it("leads with unsafe trades passed and shows before and after the floor", async () => {
    render(<Lab />);
    const hero = await screen.findByTestId("unsafe");
    expect(hero).toHaveTextContent("0");
    expect(hero.querySelector("[data-status]")).toHaveAttribute("data-status", "pass");
    const row = screen.getByText("False blocks", { selector: "td" }).closest("tr")!;
    expect(within(row).getAllByText(/\d/).map((c) => c.textContent)).toEqual(["9", "0"]);
  });

  it("shades the ablation table by the API's direction and marks it with shapes", async () => {
    render(<Lab />);
    const verifier = (await screen.findByText("-verifier")).closest("tr")!;
    const [fb, calls_] = verifier.querySelectorAll("td");
    expect(fb).toHaveAttribute("data-direction", "better");
    expect(fb).toHaveClass("heat-l3");
    expect(fb).toHaveTextContent("1 ◆");
    expect(calls_).toHaveTextContent("10.2 ▲");
    expect(screen.getByText("Read '-verifier' with care.")).toBeInTheDocument();
  });

  it("draws the reliability diagram with the count in each bin", async () => {
    render(<Lab />);
    const fig = await screen.findByRole("img", { name: "Reliability diagram, Verbal confidence" });
    expect(within(fig).getByText("n 12")).toBeInTheDocument();
    expect(within(fig).getByText("n 0")).toBeInTheDocument();
    expect(fig.querySelectorAll(".rel-bar")).toHaveLength(1);       // an empty bin draws no bar
    expect(screen.getByText(/ECE/)).toHaveTextContent("ECE 0.025");
  });

  it("never starts an evaluation: it shows the warning and the command", async () => {
    render(<Lab />);
    expect(await screen.findByText("Hours for the whole set.")).toBeInTheDocument();
    expect(screen.getByText("python -m options_copilot.cli eval --repeats 3 --save").tagName).toBe("CODE");
    expect(screen.queryByRole("button", { name: /start|run eval/i })).toBeNull();
    for (const b of screen.getAllByRole("button", { name: "Copy" })) fireEvent.click(b);
    expect(calls.every((c) => c.method === "GET")).toBe(true);
  });

  it("switches observability between the main and the eval database", async () => {
    render(<Lab />);
    expect(await screen.findByText("Event risk")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "Evaluation sandbox (eval.db)" }));
    await waitFor(() => expect(calls.some((c) => c.path === "/api/metrics?db=eval")).toBe(true));
    expect(await screen.findByText("No rejections recorded.")).toBeInTheDocument();
    expect(screen.getByLabelText("pass 10, flag 0, block 2")).toBeInTheDocument();
  });
});

describe("Playbook", () => {
  it("is read-only and says only the owner edits it", async () => {
    render(<Playbook />);
    expect(await screen.findByRole("note")).toHaveTextContent("Only the owner changes these rules");
    expect(screen.getByText("aa60459b91d8")).toBeInTheDocument();
    expect(screen.getByText("Minimum probability of profit").closest("tr")).toHaveTextContent("0.65 fraction");
    expect(screen.queryAllByRole("textbox")).toHaveLength(0);
    expect(screen.queryAllByRole("button")).toHaveLength(0);
    expect(calls.every((c) => c.method === "GET")).toBe(true);
  });
});
