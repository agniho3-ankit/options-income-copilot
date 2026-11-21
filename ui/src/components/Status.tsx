// Every status pairs a shape with its color and a text label, so nothing relies on color alone.
export type StatusKind = "pass" | "caution" | "block" | "info";

export const SHAPES: Record<StatusKind, string> = { pass: "◆", caution: "▲", block: "■", info: "●" };

export function Status({ kind, label, hideLabel = false }: { kind: StatusKind; label: string; hideLabel?: boolean }) {
  return (
    <span className={`status status-${kind}`} data-status={kind}>
      <span className="status-shape" aria-hidden="true">{SHAPES[kind]}</span>
      <span className={hideLabel ? "visually-hidden" : undefined}>{label}</span>
    </span>
  );
}

export function tierStatus(tier: string): StatusKind {
  return tier === "escalated" || tier === "look_closely" ? "caution" : "info";
}

export function verdictStatus(v: string | null | undefined): StatusKind {
  switch (v) {
    case "CLEAR":
    case "pass":
      return "pass";
    case "CAUTION":
    case "flag":
    case "INSUFFICIENT_EVIDENCE":
      return "caution";
    case "BLOCK":
    case "block":
      return "block";
    default:
      return "info";
  }
}
