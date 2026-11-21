// @vitest-environment node
// Numbers are only formatted, never computed. Two checks:
// 1) the formatters keep the API's digits (rounding for display is the only change);
// 2) screens and components contain no arithmetic: no Math.*, no multiply or divide, no reduce.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import * as f from "../lib/format";

describe("formatters", () => {
  it("keep the value they are given", () => {
    expect(f.num(347.5)).toBe("347.50");
    expect(f.pct(1.153)).toBe("1.2%");
    expect(f.pct(30, 0)).toBe("30%");
    expect(f.prob(0.7829)).toBe("78.3%");
    expect(f.money(1152.5)).toBe("$1,152.50");
    expect(f.strike(548.48)).toBe("548.48");
    expect(f.date("2027-01-15T00:00:00")).toBe("2027-01-15");
    expect(f.words("put_credit_spread")).toBe("Put credit spread");
    expect(f.num(null)).toBe("–");
  });
});

const ROOT = fileURLToPath(new URL("..", import.meta.url));

function sources(dir: string): string[] {
  return readdirSync(join(ROOT, dir), { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? sources(join(dir, e.name)) : /\.tsx?$/.test(e.name) ? [join(dir, e.name)] : []);
}

function code(text: string): string {
  return text
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/.*$/gm, "")
    .replace(/`(?:\\.|[^`\\])*`/g, "``")
    .replace(/"(?:\\.|[^"\\])*"/g, '""')
    .replace(/'(?:\\.|[^'\\])*'/g, "''")
    .replace(/^import .*$/gm, "")                // `import * as f` is not a multiply
    .replace(/<\/[\w.]*>|\/>/g, "")              // JSX closing tags
    .replace(/\/[^/\n]+\/[gimsuy]*/g, "/re/");   // regex literals
}

describe("screens and components compute nothing", () => {
  const files = [...sources("screens"), ...sources("components")];
  it("found the UI sources", () => expect(files.length).toBeGreaterThan(2));
  it.each(files)("%s has no arithmetic on values", (file) => {
    const c = code(readFileSync(join(ROOT, file), "utf8"));
    expect(c).not.toMatch(/\bMath\./);
    expect(c).not.toMatch(/\.reduce\(/);
    expect(c).not.toMatch(/[\w)\]]\s*[*/%]\s*[\w(]/);
  });
});
