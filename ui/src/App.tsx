import { useCallback, useRef, useState } from "react";
import { Dialog } from "./components/Dialog";
import { Header } from "./components/Header";
import { SHORTCUTS, useShortcuts } from "./lib/keys";
import { useRoute } from "./lib/route";
import { applyTheme, initialTheme, type Theme } from "./lib/theme";
import { Explore } from "./screens/Explore";
import { Home } from "./screens/Home";
import { Journal } from "./screens/Journal";
import { Lab } from "./screens/Lab";
import { Playbook } from "./screens/Playbook";
import { Positions } from "./screens/Positions";
import { Review } from "./screens/Review";
import { Today } from "./screens/Today";

export function App() {
  const route = useRoute();
  const [theme, setTheme] = useState<Theme>(() => {
    const t = initialTheme();
    applyTheme(t);
    return t;
  });
  const [sheet, setSheet] = useState(false);
  const search = useRef<HTMLInputElement>(null);
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    applyTheme(next);
    setTheme(next);
  };
  useShortcuts((key, e) => {
    if (key === "?") { e.preventDefault(); setSheet(true); }
    if (key === "/") { e.preventDefault(); search.current?.focus(); }
  });
  const closeSheet = useCallback(() => setSheet(false), []);
  return (
    <div className="app">
      <a className="skip-link" href="#main" onClick={(e) => {
        e.preventDefault();
        document.getElementById("main")?.focus();
      }}>Skip to content</a>
      <Header route={route} theme={theme} onToggleTheme={toggle} searchRef={search}
        onShortcuts={() => setSheet(true)} />
      <main className="main" id="main" tabIndex={-1}>
        {route.screen === "home" ? <Home />
          : route.screen === "today" ? <Today filter={route.filter} />
          : route.screen === "review" ? <Review key={route.id ?? "none"} id={route.id} />
          : route.screen === "explore" ? <Explore runId={route.id} />
          : route.screen === "positions" ? <Positions />
          : route.screen === "journal" ? <Journal />
          : route.screen === "lab" ? <Lab />
          : <Playbook />}
      </main>
      {sheet && (
        <Dialog title="Keyboard shortcuts" onClose={closeSheet}>
          <dl className="shortcut-list">
            {SHORTCUTS.map((s) => (
              <div key={s.keys}><dt><kbd>{s.keys}</kbd></dt><dd>{s.what}</dd></div>
            ))}
          </dl>
          <p className="muted">Shortcuts never approve, reject or defer by themselves: A, R and D open the same
            confirm steps as the buttons.</p>
          <button type="button" data-autofocus onClick={closeSheet}>Close</button>
        </Dialog>
      )}
    </div>
  );
}
