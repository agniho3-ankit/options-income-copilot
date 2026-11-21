// Formatting only. These functions turn values from the API into text; they never derive a
// trading number (no credit, max loss, PoP, breakeven, risk or score is computed here).

const DASH = "–";

export function num(v: number | null | undefined, digits = 2): string {
  return v === null || v === undefined || Number.isNaN(v) ? DASH : v.toFixed(digits);
}

/** A value that is already a percentage (1.267 means 1.267%). */
export function pct(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined ? DASH : `${v.toFixed(digits)}%`;
}

/** A probability stored as a fraction (0.7829) shown as a percentage: a unit change for display only. */
export function prob(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined ? DASH : `${(v * 100).toFixed(digits)}%`;
}

export function money(v: number | null | undefined): string {
  if (v === null || v === undefined) return DASH;
  return `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function strike(v: number | null | undefined): string {
  return v === null || v === undefined ? DASH : String(v);
}

export function date(iso: string | null | undefined): string {
  return iso ? iso.slice(0, 10) : DASH;
}

/** snake_case code to sentence case words: "put_credit_spread" -> "Put credit spread". */
export function words(code: string | null | undefined): string {
  if (!code) return DASH;
  const s = code.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function legs(list: { side: string | null; option_type: string | null; strike: number | null }[]): string {
  return list.map((g) => `${g.side ?? ""} ${g.option_type ?? ""} ${strike(g.strike)}`.trim()).join(" · ");
}
