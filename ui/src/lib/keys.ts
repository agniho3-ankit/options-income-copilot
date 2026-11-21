// Keyboard shortcuts. A shortcut never acts while the owner is typing, while a dialog is open, or
// with a modifier key held; and no shortcut performs a decision: A, R and D only open the same
// confirm steps the buttons open.
import { useEffect, useRef } from "react";

export function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || !el.tagName) return false;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) || el.isContentEditable;
}

export function dialogOpen(): boolean {
  return document.querySelector('[aria-modal="true"]') !== null;
}

/** Call `handler(key, event)` for plain key presses that are allowed to act as shortcuts. */
export function useShortcuts(handler: (key: string, e: KeyboardEvent) => void, enabled = true): void {
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => {
    if (!enabled) return;
    const on = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
      if (isTyping(e.target) || dialogOpen()) return;
      ref.current(e.key, e);
    };
    document.addEventListener("keydown", on);
    return () => document.removeEventListener("keydown", on);
  }, [enabled]);
}

export const SHORTCUTS: { keys: string; what: string }[] = [
  { keys: "J / K", what: "Next / previous brief in the queue (Today)" },
  { keys: "Enter", what: "Open the focused brief" },
  { keys: "A", what: "Open the approve confirm step (Review)" },
  { keys: "R", what: "Open the reject form (Review)" },
  { keys: "D", what: "Open the defer confirm step (Review)" },
  { keys: "/", what: "Search tickers" },
  { keys: "?", what: "Show this list" },
  { keys: "Esc", what: "Close a dialog without doing anything" },
];

/** Move focus to the next (1) or previous (-1) element matching `selector`, in document order. */
export function moveFocus(selector: string, step: 1 | -1): void {
  const items = [...document.querySelectorAll<HTMLElement>(selector)];
  if (items.length === 0) return;
  const at = items.indexOf(document.activeElement as HTMLElement);
  const next = at < 0 ? (step === 1 ? 0 : items.length - 1) : Math.min(items.length - 1, Math.max(0, at + step));
  items[next].focus();
}
