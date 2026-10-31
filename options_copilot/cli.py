"""Command-line entry point: python -m options_copilot.cli <command>."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .config import load_settings
from .models import Reflection, ReflectionTrigger, to_jsonable
from .playbook import PlaybookError, load_playbook

DEFAULT_CONTEXT = "selling a 45-DTE put credit spread"


def cmd_playbook(args: argparse.Namespace) -> int:
    settings = load_settings()
    try:
        pb = load_playbook(settings.playbook_path)
    except PlaybookError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Playbook {pb.source}  version {pb.version}  sha256 {pb.sha256[:12]}  (valid)")
    print(f"  Tier 1: {', '.join(pb['universe.tier_1'])}")
    print(f"  Tier 2: {', '.join(pb['universe.tier_2'])}")
    print(f"  Watch:  {', '.join(pb['universe.watch'])}")
    print(f"  Entry:  {pb['entry.dte_min']}-{pb['entry.dte_max']} DTE (target {pb['entry.dte_target']}), "
          f"PoP >= {pb['entry.pop_min']:.0%}, short delta "
          f"{pb['builder.short_delta_min']:.2f}-{pb['builder.short_delta_max']:.2f}, "
          f"wings {pb['builder.wing_width_pct']}% of price")
    print(f"  Limits: position risk warn {pb['entry.position_risk_pct.warn']}% / block "
          f"{pb['entry.position_risk_pct.block']}%, heat warn {pb['entry.portfolio_heat_pct.warn']}% / "
          f"block {pb['entry.portfolio_heat_pct.block']}%")
    print(f"  Earnings: block within {pb['entry.earnings.block_within_days']}d, spreads only within "
          f"{pb['entry.earnings.spreads_only_within_days']}d, expiry must precede earnings")
    print(f"  Exits: take profit at {pb['management.r1_profit_target_pct']}%, roll below "
          f"{pb['management.r2_roll.dte_below']} DTE, stop at "
          f"{pb['management.r3_stop_loss.loss_credit_multiple']}x credit")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    s = load_settings()
    print(f"ollama:     {s.ollama_model} at {s.ollama_url} (think={s.ollama_think}, temperature={s.llm_temperature})")
    print(f"fallback:   {s.anthropic_model if s.has_anthropic_fallback else 'off (no ANTHROPIC_API_KEY)'}")
    print(f"sec agent:  {'set' if s.sec_user_agent else 'NOT SET (needed for ingest)'}")
    print(f"playbook:   {s.playbook_path.relative_to(s.repo_root) if s.playbook_path.is_relative_to(s.repo_root) else 'custom path'}")
    print(f"data dir:   {s.data_dir.relative_to(s.repo_root) if s.data_dir.is_relative_to(s.repo_root) else 'custom path'}")
    print(f"portfolio:  synthetic, {s.synthetic_portfolio_value:,.0f}")
    return 0


# --- Phase 1: RAG landmine check ----------------------------------------------------------

def _store(settings):
    from .store import Store
    return Store(settings.chroma_dir)


def _as_of(args) -> date:
    return date.fromisoformat(args.as_of) if args.as_of else date.today()


def _universe(settings) -> list[str]:
    pb = load_playbook(settings.playbook_path)
    return [*pb["universe.tier_1"], *pb["universe.tier_2"], *pb["universe.watch"]]


def cmd_ingest(args: argparse.Namespace) -> int:
    from .edgar import EdgarClient, EdgarError
    from .ingest import ingest_ticker

    settings = load_settings()
    tickers = [t.upper() for t in args.tickers] or (_universe(settings) if args.universe else [])
    if not tickers:
        print("name at least one ticker, or pass --universe", file=sys.stderr)
        return 2
    try:
        edgar = EdgarClient(settings.sec_user_agent, settings.raw_dir)
    except EdgarError as exc:
        print(exc, file=sys.stderr)
        return 1
    store, as_of = _store(settings), _as_of(args)
    failed = 0
    for ticker in tickers:
        r = ingest_ticker(ticker, edgar=edgar, store=store, raw_dir=settings.raw_dir, as_of=as_of,
                          cfg=store.cfg, headlines=not args.no_headlines, reindex=args.reindex)
        print(f"{ticker}: {len(r.filings)} new filing(s), {r.chunks} chunks, {r.headlines} headlines"
              + (f", {len(r.skipped)} already indexed" if r.skipped else ""))
        for line in r.filings:
            print(f"  + {line}")
        for line in r.errors:
            print(f"  ! {line}")
        failed += bool(r.errors) and not r.filings
    print(f"SEC requests this run: {edgar.requests_made} (cached files are reused)")
    return 1 if failed == len(tickers) else 0


def cmd_filings(args: argparse.Namespace) -> int:
    rows = _store(load_settings()).list_filings(args.ticker)
    if not rows:
        print(f"nothing indexed for {args.ticker.upper()}")
    for r in rows:
        items = f" items {r['items']}" if r["items"] else ""
        print(f"{r['date']}  {r['form']:<8}{items:<22} {r['chunks']:>4} chunks  {r['url']}")
    return 0


def _render(result: dict) -> str:
    title = "WITH RAG" if result["use_retrieval"] else "WITHOUT RAG (model memory only)"
    lines = [f"== {result['ticker']} landmine check, {title} ==",
             f"as of {result['as_of']} | trade: {result['trade_context']} | model: {result['model']}"]
    if "coverage" in result:
        lines.append(f"coverage: {'ok' if result['coverage']['ok'] else 'FAILED'} ({result['coverage']['reason']})")
    lines.append(f"VERDICT: {result['verdict']}"
                 + (f"   (model proposed {result['llm_verdict']})" if result.get("llm_verdict")
                    and result["llm_verdict"] != result["verdict"] else ""))
    for r in result.get("reasons", []):
        lines.append(f"  - {r}")
    for c in result.get("citations", []):
        lines.append(f"  cite: {c['form']} {c['date']} {c['item']}  {c['url']}")
    for n in result.get("guard_notes", []):
        lines.append(f"  guard: {n}")
    for lesson in result.get("lessons", []):
        lines.append(f"  lesson: {lesson[:110]}{'...' if len(lesson) > 110 else ''}")
    if result.get("llm_error"):
        lines.append(f"  llm error: {result['llm_error']}")
    return "\n".join(lines)


def _save(settings, name: str, payload, text: str) -> None:
    out = settings.repo_root / "demo_outputs"
    out.mkdir(exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")
    (out / f"{name}.txt").write_text(text + "\n")
    print(f"saved demo_outputs/{name}.json and .txt")


def _run_check(settings, args, use_retrieval: bool) -> dict:
    from .agent import landmine_check
    from .llm import build_llm

    store = _store(settings) if use_retrieval else None
    return landmine_check(args.ticker, args.context, use_retrieval, store=store, llm=build_llm(settings),
                          as_of=_as_of(args))


def cmd_check(args: argparse.Namespace) -> int:
    settings = load_settings()
    result = _run_check(settings, args, use_retrieval=not args.no_rag)
    text = _render(result)
    print(json.dumps(result, indent=2) if args.json else text)
    if args.save:
        _save(settings, f"check_{result['ticker']}_{result['as_of']}{'_norag' if args.no_rag else ''}",
              result, text)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    settings = load_settings()
    without = _run_check(settings, args, use_retrieval=False)
    with_rag = _run_check(settings, args, use_retrieval=True)
    changed = "CHANGED" if without["verdict"] != with_rag["verdict"] else "unchanged"
    text = "\n\n".join([_render(without), _render(with_rag),
                        f"Verdict without RAG: {without['verdict']} (model said {without['llm_verdict']}) "
                        f"-> with RAG: {with_rag['verdict']}  [{changed}]"])
    print(text)
    if args.save:
        _save(settings, f"compare_{with_rag['ticker']}_{with_rag['as_of']}",
              {"without_rag": without, "with_rag": with_rag}, text)
    return 0


def cmd_reflect(args: argparse.Namespace) -> int:
    lesson = Reflection(lesson=args.lesson, trigger=ReflectionTrigger(args.trigger), ticker=args.ticker.upper(),
                        tags=[t.strip() for t in args.tags.split(",") if t.strip()])
    if len(lesson.lesson) > 600:
        print("keep lessons under 600 characters", file=sys.stderr)
        return 2
    rid = _store(load_settings()).add_reflection(lesson)
    print(f"stored {rid} for {lesson.ticker} ({lesson.trigger.value})")
    return 0


def cmd_seed_lessons(args: argparse.Namespace) -> int:
    import yaml

    data = yaml.safe_load((Path(__file__).parent / "seed_lessons.yaml").read_text())
    store = _store(load_settings())
    for row in data["lessons"]:
        store.add_reflection(Reflection(lesson=row["lesson"], trigger=ReflectionTrigger(row["trigger"]),
                                        ticker=row["ticker"], tags=row.get("tags", []), synthetic=True))
    print(f"seeded {len(data['lessons'])} synthetic lessons (re-running is safe)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="copilot", description="Options Income Copilot (educational; no live trading)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("playbook", help="validate playbook.yaml and print its key rules").set_defaults(func=cmd_playbook)
    sub.add_parser("config", help="show settings (secrets are never printed)").set_defaults(func=cmd_config)

    p = sub.add_parser("ingest", help="download recent SEC filings and headlines into the index")
    p.add_argument("tickers", nargs="*")
    p.add_argument("--universe", action="store_true", help="ingest every ticker in playbook.yaml")
    p.add_argument("--no-headlines", action="store_true")
    p.add_argument("--reindex", action="store_true", help="re-chunk filings that are already indexed")
    p.add_argument("--as-of", help="YYYY-MM-DD (default today)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("filings", help="list what is indexed for a ticker")
    p.add_argument("ticker")
    p.set_defaults(func=cmd_filings)

    for name, func, helptext in (("check", cmd_check, "run the landmine check for one ticker"),
                                 ("compare", cmd_compare, "landmine check with and without RAG")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("ticker")
        p.add_argument("--context", default=DEFAULT_CONTEXT, help=f"planned trade (default: {DEFAULT_CONTEXT!r})")
        p.add_argument("--as-of", help="YYYY-MM-DD (default today)")
        p.add_argument("--save", action="store_true", help="write the output to demo_outputs/ (gitignored)")
        if name == "check":
            p.add_argument("--no-rag", action="store_true", help="skip retrieval (baseline)")
            p.add_argument("--json", action="store_true")
        p.set_defaults(func=func)

    p = sub.add_parser("reflect", help="store a lesson in the reflections collection")
    p.add_argument("lesson")
    p.add_argument("--ticker", default="ANY")
    p.add_argument("--trigger", required=True, choices=[t.value for t in ReflectionTrigger])
    p.add_argument("--tags", default="")
    p.set_defaults(func=cmd_reflect)

    sub.add_parser("seed-lessons", help="load the synthetic seed lessons").set_defaults(func=cmd_seed_lessons)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
