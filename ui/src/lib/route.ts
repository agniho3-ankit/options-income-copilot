import { useEffect, useState } from "react";

export type Route =
  | { screen: "today"; filter: string | null }
  | { screen: "review"; id: string | null }
  | { screen: "explore"; id: string | null }
  | { screen: "positions" }
  | { screen: "journal" }
  | { screen: "lab" }
  | { screen: "playbook" };

export function parse(hash: string): Route {
  const [screen, id] = hash.replace(/^#\/?/, "").split("/");
  switch (screen) {
    case "review":
      return { screen: "review", id: id ? decodeURIComponent(id) : null };
    case "explore":
      return { screen: "explore", id: id ? decodeURIComponent(id) : null };
    case "positions":
    case "journal":
    case "lab":
    case "playbook":
      return { screen };
    default:
      return { screen: "today", filter: screen === "today" && id ? decodeURIComponent(id).toUpperCase() : null };
  }
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
