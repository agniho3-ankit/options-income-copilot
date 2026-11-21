// Icicle layout for the Tree-of-Thought: pure geometry over the stored nodes (depth columns,
// one row per leaf). Statuses, scores and prune reasons are shown as stored, never recomputed.

export type TotNode = {
  id: string; parent_id: string | null; depth: number; status: string; score: number | null;
  prune_reason: string; source: string; rationale?: string;
  spec: { strategy: string | null; expiry: string | null; target_delta: number | null; wing_width_pct?: number | null };
  detail: Record<string, unknown> & {
    thin_expiries?: string[]; critic?: { action?: string; reason?: string; cite?: string };
    numbers?: { credit?: number; pop?: number; max_loss?: number }; verdict?: string; exploration?: boolean;
    revision?: boolean;
  };
};

export type Cell = { node: TotNode; x0: number; x1: number; y0: number; y1: number };

export function layout(nodes: TotNode[], width: number, rowHeight: number): { cells: Cell[]; height: number } {
  const kids = new Map<string | null, TotNode[]>();
  for (const n of nodes) kids.set(n.parent_id, [...(kids.get(n.parent_id) ?? []), n]);
  const maxDepth = Math.max(0, ...nodes.map((n) => n.depth));
  // root, strategy, expiry, delta: the leaf column carries the most text
  const weights = maxDepth === 3 ? [0.08, 0.3, 0.17, 0.45] : Array.from({ length: maxDepth + 1 }, () => 1 / (maxDepth + 1));
  const edges = weights.reduce<number[]>((acc, w) => [...acc, acc[acc.length - 1] + w * width], [0]);
  const cells: Cell[] = [];
  let row = 0;
  const place = (n: TotNode): [number, number] => {
    const children = kids.get(n.id) ?? [];
    let top: number;
    let bottom: number;
    if (children.length === 0) {
      top = row;
      bottom = row + 1;
      row += 1;
    } else {
      const spans = children.map(place);
      top = spans[0][0];
      bottom = spans[spans.length - 1][1];
    }
    cells.push({ node: n, x0: edges[n.depth], x1: edges[n.depth + 1], y0: top * rowHeight, y1: bottom * rowHeight });
    return [top, bottom];
  };
  for (const root of kids.get(null) ?? []) place(root);
  return { cells: cells.sort((a, b) => a.node.depth - b.node.depth || a.y0 - b.y0), height: row * rowHeight };
}

export function label(n: TotNode): string {
  if (n.depth === 0) return "root";
  if (n.depth === 1) return (n.spec.strategy ?? "?").replace(/_/g, " ");
  if (n.depth === 2) return n.spec.expiry ?? "?";
  return n.spec.target_delta === null ? n.id : `Δ ${n.spec.target_delta}`;
}

export function criticAction(n: TotNode): string | null {
  const a = n.detail?.critic?.action;
  return a && a !== "keep" ? a : null;
}
