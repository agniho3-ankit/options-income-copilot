// The Tree-of-Thought as an icicle: strategy, then expiry, then delta. Pruned branches are greyed
// with their stored reason; liquidity withholds and critic lowers are marked; the selected leaf is
// highlighted. Click a cell for its details.
import { useState } from "react";
import * as f from "../lib/format";
import { criticAction, label, layout, type TotNode } from "../lib/tree";
import { Status } from "./Status";

const W = 760;
const ROW = 22;

export function TotTree({ nodes }: { nodes: TotNode[] }) {
  const [picked, setPicked] = useState<string | null>(() => nodes.find((n) => n.status === "selected")?.id ?? null);
  if (nodes.length === 0) return <p className="muted">No Tree-of-Thought nodes were stored for this run.</p>;
  const { cells, height } = layout(nodes, W, ROW);
  const chosen = nodes.find((n) => n.id === picked);
  const selected = nodes.find((n) => n.status === "selected");
  return (
    <div className="tot">
      <p className="muted">
        {selected ? <>Selected <span className="num">{pathOf(selected, nodes)}</span> (score{" "}
          <span className="num">{f.num(selected.score, 3)}</span>).</> : "No leaf was selected."}{" "}
        Columns: strategy, expiry, delta. Greyed cells were pruned.
      </p>
      <div className="tot-scroll">
        <svg viewBox={`0 0 ${W} ${height}`} className="tot-svg" role="tree" aria-label="Tree-of-Thought">
          {cells.map(({ node: n, x0, x1, y0, y1 }) => {
            const critic = criticAction(n);
            const thin = n.detail?.thin_expiries?.length ?? 0;
            return (
              <g key={n.id} role="treeitem" aria-level={n.depth + 1} aria-selected={picked === n.id}
                tabIndex={0} className={`tot-cell tot-${n.status}${picked === n.id ? " tot-picked" : ""}`}
                data-node={n.id} data-status={n.status}
                onClick={() => setPicked(n.id)}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setPicked(n.id); } }}>
                <title>{`${label(n)}: ${n.status}${n.score !== null ? `, score ${n.score}` : ""}${n.prune_reason ? `. ${n.prune_reason}` : ""}`}</title>
                <rect x={x0 + 1} y={y0 + 1} width={x1 - x0 - 2} height={y1 - y0 - 2} rx={3} />
                {/* a nested svg clips the label to its cell */}
                <svg x={x0} y={y0} width={x1 - x0 - 4} height={y1 - y0} overflow="hidden">
                  <text x={6} y={ROW - 7} className="num">
                    {label(n)}{n.score !== null ? `  ${f.num(n.score, 2)}` : ""}
                    {critic ? `  ▲ critic ${critic.replace("_", " ")}` : ""}
                  </text>
                  {thin > 0 && (
                    <text x={6} y={ROW - 7} dy={ROW - 4} className="num tot-flag">
                      ▲ {thin} thin expir{thin === 1 ? "y" : "ies"} withheld
                    </text>
                  )}
                </svg>
              </g>
            );
          })}
        </svg>
      </div>
      {chosen && <NodeDetail node={chosen} nodes={nodes} />}
    </div>
  );
}

function pathOf(n: TotNode, nodes: TotNode[]): string {
  const byId = new Map(nodes.map((x) => [x.id, x]));
  const parts: string[] = [];
  for (let cur: TotNode | undefined = n; cur && cur.depth > 0; cur = cur.parent_id ? byId.get(cur.parent_id) : undefined) {
    parts.unshift(label(cur));
  }
  return parts.join(" / ");
}

function NodeDetail({ node: n, nodes }: { node: TotNode; nodes: TotNode[] }) {
  const kind = n.status === "selected" ? "pass" : n.status === "pruned" ? "block" : "info";
  const critic = n.detail?.critic;
  return (
    <div className="tot-detail" role="region" aria-label={`Node ${n.id} details`}>
      <p><Status kind={kind} label={f.words(n.status)} /> <span className="num">{n.id}</span> ·{" "}
        <span className="num">{pathOf(n, nodes) || "root"}</span>
        {n.score !== null && <> · score <span className="num">{f.num(n.score, 3)}</span></>}</p>
      {n.prune_reason && <p className="muted">Pruned: {n.prune_reason}</p>}
      {n.rationale && <p className="muted">Generator rationale: {n.rationale}</p>}
      {critic?.action && (
        <p><Status kind={critic.action === "keep" ? "pass" : "caution"} label={`Critic ${critic.action.replace("_", " ")}`} />{" "}
          {critic.reason}</p>
      )}
      {(n.detail?.thin_expiries?.length ?? 0) > 0 && (
        <div><p>Withheld by the liquidity floor:</p>
          <ul>{n.detail.thin_expiries!.map((t) => <li key={t} className="num">{t}</li>)}</ul></div>
      )}
      {n.detail?.numbers && (
        <p className="num">credit {f.money(n.detail.numbers.credit)} · max loss {f.money(n.detail.numbers.max_loss)} ·
          PoP {f.prob(n.detail.numbers.pop)}{n.detail.verdict ? ` · verifier ${n.detail.verdict}` : ""}</p>
      )}
    </div>
  );
}
