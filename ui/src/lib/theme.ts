export type Theme = "dark" | "light";
const KEY = "copilot.theme";

export function initialTheme(): Theme {
  try {
    const saved = window.localStorage.getItem(KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    // storage blocked: fall back to the default
  }
  return "dark";
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem(KEY, theme);
  } catch {
    // a per-viewer convenience only
  }
}
