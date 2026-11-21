import { useState } from "react";
import { Header } from "./components/Header";
import { useRoute } from "./lib/route";
import { applyTheme, initialTheme, type Theme } from "./lib/theme";
import { Explore } from "./screens/Explore";
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
          : route.screen === "positions" ? <Positions />
          : route.screen === "journal" ? <Journal />
          : route.screen === "lab" ? <Lab />
          : <Playbook />}
      </main>
    </div>
  );
}
