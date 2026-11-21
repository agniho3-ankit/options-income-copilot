// @vitest-environment node
// WCAG AA contrast for every token pair the UI uses, in both themes.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const css = readFileSync(fileURLToPath(new URL("../styles/tokens.css", import.meta.url)), "utf8");

function block(selector: RegExp): Record<string, string> {
  const m = css.match(selector);
  if (!m) throw new Error(`no block for ${selector}`);
  return Object.fromEntries([...m[1].matchAll(/--([\w-]+):\s*(#[0-9a-f]{6})/gi)].map((x) => [x[1], x[2]]));
}

function luminance(hex: string): number {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

export function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

const themes = {
  dark: block(/:root\[data-theme="dark"\]\s*\{([^}]*)\}/),
  light: block(/:root\[data-theme="light"\]\s*\{([^}]*)\}/),
};
const TEXT = ["text", "text-muted", "accent", "pass", "caution", "block", "info"];
const SURFACES = ["bg", "surface", "surface-2"];

describe.each(Object.entries(themes))("%s theme", (_name, t) => {
  it.each(TEXT.flatMap((fg) => SURFACES.map((bg) => [fg, bg])))("%s on %s is at least 4.5:1", (fg, bg) => {
    expect(contrast(t[fg], t[bg])).toBeGreaterThanOrEqual(4.5);
  });
  it("accent ink on accent is at least 4.5:1", () => {
    expect(contrast(t["accent-ink"], t.accent)).toBeGreaterThanOrEqual(4.5);
  });
  it("focus ring is at least 3:1 against every surface", () => {
    for (const bg of SURFACES) expect(contrast(t.focus, t[bg])).toBeGreaterThanOrEqual(3);
  });
});

describe("motion", () => {
  it("the stylesheet switches motion off for reduced-motion users", () => {
    const base = readFileSync(fileURLToPath(new URL("../styles/base.css", import.meta.url)), "utf8");
    const block = base.slice(base.indexOf("@media (prefers-reduced-motion: reduce)"));
    expect(block).toMatch(/animation: none !important/);
    expect(block).toMatch(/transition: none !important/);
  });
});
