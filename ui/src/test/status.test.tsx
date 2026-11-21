import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SHAPES, Status, type StatusKind, tierStatus, verdictStatus } from "../components/Status";

const KINDS: StatusKind[] = ["pass", "caution", "block", "info"];

describe("status shapes", () => {
  it("every status has its own shape as well as a color and a label", () => {
    expect(new Set(Object.values(SHAPES)).size).toBe(KINDS.length);
    for (const kind of KINDS) {
      const { container, unmount } = render(<Status kind={kind} label={`label ${kind}`} />);
      const el = container.querySelector("[data-status]")!;
      expect(el).toHaveClass(`status-${kind}`);
      expect(el.querySelector(".status-shape")?.textContent).toBe(SHAPES[kind]);
      expect(el).toHaveTextContent(`label ${kind}`);
      unmount();
    }
  });

  it("a hidden label stays readable to screen readers", () => {
    const { container } = render(<Status kind="block" label="blocked" hideLabel />);
    expect(container.querySelector(".visually-hidden")).toHaveTextContent("blocked");
  });

  it("maps verdicts and tiers to statuses; missing evidence is never a pass", () => {
    expect(verdictStatus("CLEAR")).toBe("pass");
    expect(verdictStatus("CAUTION")).toBe("caution");
    expect(verdictStatus("BLOCK")).toBe("block");
    expect(verdictStatus("INSUFFICIENT_EVIDENCE")).toBe("caution");
    expect(verdictStatus(null)).toBe("info");
    expect(tierStatus("escalated")).toBe("caution");
    expect(tierStatus("standard")).toBe("info");
  });
});
