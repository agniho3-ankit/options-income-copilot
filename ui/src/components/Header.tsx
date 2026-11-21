import { useState, type RefObject } from "react";
import type { Route } from "../lib/route";
import type { Theme } from "../lib/theme";

const NAV: { screen: Route["screen"]; label: string; href: string }[] = [
  { screen: "today", label: "Today", href: "#/today" },
  { screen: "review", label: "Review", href: "#/review" },
  { screen: "explore", label: "Explore", href: "#/explore" },
  { screen: "positions", label: "Positions", href: "#/positions" },
  { screen: "journal", label: "Journal", href: "#/journal" },
  { screen: "lab", label: "Lab", href: "#/lab" },
  { screen: "playbook", label: "Playbook", href: "#/playbook" },
];

export function Header({ route, theme, onToggleTheme, searchRef, onShortcuts }: {
  route: Route; theme: Theme; onToggleTheme: () => void;
  searchRef?: RefObject<HTMLInputElement | null>; onShortcuts?: () => void;
}) {
  const next = theme === "dark" ? "light" : "dark";
  const [q, setQ] = useState("");
  return (
    <header className="header">
      <span className="brand">Options Income Copilot</span>
      <nav className="nav" aria-label="Screens">
        {NAV.map((n) => (
          <a key={n.screen} href={n.href} aria-current={route.screen === n.screen ? "page" : undefined}>{n.label}</a>
        ))}
      </nav>
      <form className="search" role="search" onSubmit={(e) => {
        e.preventDefault();
        const t = q.trim().toUpperCase();
        window.location.hash = t ? `#/today/${encodeURIComponent(t)}` : "#/today";
        searchRef?.current?.blur();
      }}>
        <input ref={searchRef} type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ticker  /"
          aria-label="Search tickers" maxLength={8} autoComplete="off" spellCheck={false} className="num"
          onKeyDown={(e) => { if (e.key === "Escape") { setQ(""); e.currentTarget.blur(); } }} />
      </form>
      <div className="header-right">
        <span className="paper-only" title="Nothing here places a real order. Approve writes a paper ticket.">
          Paper only
        </span>
        {onShortcuts && <button type="button" onClick={onShortcuts} aria-label="Keyboard shortcuts">?</button>}
        <button type="button" onClick={onToggleTheme} aria-label={`Switch to ${next} theme`}>
          {next === "light" ? "Light" : "Dark"} theme
        </button>
      </div>
    </header>
  );
}
