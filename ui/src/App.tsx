import { useState } from "react";
import { Header } from "./components/Header";
import { useRoute, type Route } from "./lib/route";
import { applyTheme, initialTheme, type Theme } from "./lib/theme";
import { Explore } from "./screens/Explore";
import { Review } from "./screens/Review";
import { Today } from "./screens/Today";

const LATER: Record<Exclude<Route["screen"], "today" | "review" | "explore">, string> = {
  positions: "slice 6", journal: "slice 6", lab: "slice 7", playbook: "slice 7",
};

export function App() {
  const route = useRoute();
  const [theme, setTheme] = useState<Theme>(() => {
    const t = initialTheme();
    applyTheme(t);
    return t;
  });
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    applyTheme(next);
    setTheme(next);
  };
  return (
    <div className="app">
      <Header route={route} theme={theme} onToggleTheme={toggle} />
      <main className="main" id="main">
        {route.screen === "today" ? <Today />
          : route.screen === "review" ? <Review key={route.id ?? "none"} id={route.id} />
          : route.screen === "explore" ? <Explore runId={route.id} />
          : <p className="empty">This screen is not built yet ({LATER[route.screen]}).</p>}
      </main>
    </div>
  );
}
