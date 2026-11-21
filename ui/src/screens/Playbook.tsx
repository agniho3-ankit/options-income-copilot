// Playbook viewer: read-only. The rules live in playbook.yaml; only the owner edits them, in the file.
import { useEffect, useState } from "react";
import { get, type PlaybookView } from "../lib/api";
import * as f from "../lib/format";

function show(v: unknown): string {
  if (Array.isArray(v)) return v.map(show).join(", ");
  if (v === null || v === undefined) return "–";
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
}

function Tree({ node }: { node: Record<string, unknown> }) {
  return (
    <dl className="pb-tree">
      {Object.entries(node).map(([k, v]) => (
        <div key={k} className="pb-row">
          <dt className="num">{k}</dt>
          <dd>{v !== null && typeof v === "object" && !Array.isArray(v)
            ? <Tree node={v as Record<string, unknown>} /> : <span className="num">{show(v)}</span>}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Playbook() {
  const [pb, setPb] = useState<PlaybookView | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { get<PlaybookView>("/api/playbook").then(setPb).catch((e: Error) => setError(e.message)); }, []);
  if (error) return <p className="error" role="alert">Could not load the playbook: {error}</p>;
  if (!pb) return <p className="muted" role="status">Loading the playbook…</p>;
  const sections = Object.entries(pb.tree).filter(([k]) => k !== "version");
  return (
    <div className="playbook">
      <h1 className="visually-hidden">Playbook</h1>
      <section className="panel" aria-labelledby="pb-title">
        <h2 id="pb-title">Playbook</h2>
        <p className="readonly-note" role="note">Read-only. Only the owner changes these rules, by editing{" "}
          <code>{pb.source}</code> and bumping its version. The agent and this screen never change them.</p>
        <p className="muted">Version <span className="num">{pb.version}</span> · sha256{" "}
          <span className="num sha">{pb.sha256}</span></p>
      </section>
      <section className="panel" aria-labelledby="key-title">
        <h2 id="key-title">Key thresholds</h2>
        <table>
          <tbody>
            {pb.key_thresholds.map((k) => (
              <tr key={k.path}>
                <td>{k.label}</td>
                <td className="num">{show(k.value)} <span className="muted">{k.unit}</span></td>
                <td className="num muted">{k.path}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="panel" aria-labelledby="sections-title">
        <h2 id="sections-title">All sections</h2>
        {sections.map(([name, body]) => (
          <details key={name} className="pb-section">
            <summary>{f.words(name)}</summary>
            {body !== null && typeof body === "object" && !Array.isArray(body)
              ? <Tree node={body as Record<string, unknown>} /> : <p className="num">{show(body)}</p>}
          </details>
        ))}
      </section>
    </div>
  );
}
