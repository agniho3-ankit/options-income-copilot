// Pixel geometry for charts. These map values the API already computed onto screen
// coordinates and pick round tick marks; they never derive a trading number.

export type Linear = ((v: number) => number) & { domain: [number, number]; range: [number, number] };

export function linear(domain: [number, number], range: [number, number]): Linear {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const f = ((v: number) => r0 + ((v - d0) / span) * (r1 - r0)) as Linear;
  f.domain = domain;
  f.range = range;
  return f;
}

export function dayNumber(iso: string): number {
  return Date.parse(`${iso.slice(0, 10)}T00:00:00Z`) / 86_400_000;
}

export function timeScale(domain: [string, string], range: [number, number]): (iso: string) => number {
  const s = linear([dayNumber(domain[0]), dayNumber(domain[1])], range);
  return (iso: string) => s(dayNumber(iso));
}

/** Round tick values inside [lo, hi], about `count` of them. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  const raw = (hi - lo) / Math.max(count, 1);
  if (!(raw > 0)) return [lo];
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(6)));
  return out;
}

/** First day of each month between two ISO dates (for time-axis ticks). */
export function monthTicks(start: string, end: string): string[] {
  const out: string[] = [];
  const d = new Date(`${start.slice(0, 7)}-01T00:00:00Z`);
  const stop = Date.parse(`${end.slice(0, 10)}T00:00:00Z`);
  for (d.setUTCMonth(d.getUTCMonth() + 1); d.getTime() <= stop; d.setUTCMonth(d.getUTCMonth() + 1)) {
    out.push(d.toISOString().slice(0, 10));
  }
  return out;
}

/** "M x,y L x,y ..." for a polyline. */
export function path(points: [number, number][]): string {
  return points.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
}

/** Label positions that keep at least `gap` pixels apart, as close to their lines as possible. */
export function spreadLabels(ys: number[], gap: number): number[] {
  const order = ys.map((y, i) => [y, i] as const).sort((a, b) => a[0] - b[0]);
  const placed: number[] = [];
  for (const [y] of order) placed.push(placed.length ? Math.max(y, placed[placed.length - 1] + gap) : y);
  const out = new Array<number>(ys.length);
  order.forEach(([, i], k) => { out[i] = placed[k]; });
  return out;
}

/** Clamp an open-ended zone bound to the visible domain. */
export function bound(v: number | null, fallback: number): number {
  return v === null ? fallback : v;
}

/** Map ISO timestamps (with time of day) onto pixels. */
export function instantScale(domain: [number, number], range: [number, number]): (iso: string | number) => number {
  const s = linear(domain, range);
  return (t: string | number) => s(typeof t === "number" ? t : Date.parse(t));
}

export function instant(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isNaN(t) ? null : t;
}

/** Elapsed m:ss between two instants (display only). */
export function elapsed(from: number, to: number): string {
  const s = Math.max(0, Math.round((to - from) / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** The time window a set of spans covers, ending at `now` while any is still open. */
export function spanWindow(spans: { started_at: string; ended_at: string | null }[], now: number): [number, number] {
  const starts = spans.map((s) => instant(s.started_at)).filter((t): t is number => t !== null);
  const ends = spans.map((s) => instant(s.ended_at) ?? now);
  if (starts.length === 0) return [now - 1000, now];
  const lo = Math.min(...starts);
  return [lo, Math.max(lo + 1000, ...ends)];
}

/** [x, width] for a bar from x0 to x1, at least `min` pixels wide so short spans stay visible. */
export function bar(x0: number, x1: number, min = 3): [number, number] {
  return [x0, Math.max(min, x1 - x0)];
}

/** Top of row `i` for rows `h` pixels tall. */
export function rowTop(i: number, h: number, offset = 0): number {
  return offset + i * h;
}

/** Bar lengths (percent of the longest) for a set of counts: display geometry only. */
export function barLengths(counts: number[]): number[] {
  const top = Math.max(1, ...counts);
  return counts.map((n) => Math.round((n / top) * 1000) / 10);
}
