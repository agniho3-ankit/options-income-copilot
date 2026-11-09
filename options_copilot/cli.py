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
          f"{pb['builder.short_delta_min']:.2f}-{pb['builder.short_delta_max']:.2f} "
          f"(condors {pb['builder.condor_short_delta_min']:.2f}-{pb['builder.condor_short_delta_max']:.2f}), "
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


# --- Phase 2: calculators, verifier, management, mask -------------------------------------

def _edgar_or_none(settings):
    from .edgar import EdgarClient, EdgarError
    try:
        return EdgarClient(settings.sec_user_agent, settings.raw_dir)
    except EdgarError:
        return None


def _atm_iv_proxy(market, ticker, expiry, spot, closes, pb):
    from .indicators import iv_rank_proxy, realized_vol
    quotes, _ = market.chain(ticker, expiry)
    near = sorted((q for q in quotes if q.iv), key=lambda q: abs(q.strike - spot))[:4]
    rv = realized_vol(closes, pb["iv_rank.realized_vol_window_days"])
    if not near or not rv:
        return None
    atm = sum(q.iv for q in near) / len(near)
    return iv_rank_proxy(atm, rv), f"proxy: ATM IV {atm:.0%} / 30d RV {rv:.0%}, no IV history yet"


def _render_build(o, settings) -> str:
    lines = [f"== {o.ticker} {o.strategy.value} build, as of {o.as_of} ==",
             f"spot {o.data.profile.price:.2f} | support {o.data.support} | resistance {o.data.resistance} | "
             f"analyst target {o.data.profile.analyst_target} | earnings {o.data.profile.earnings_date}"]
    for n in o.data.notes:
        lines.append(f"  note: {n}")
    if o.error:
        lines.append(f"NO TRADE: {o.error}")
        return "\n".join(lines)
    p, n, x = o.proposal, o.numbers, o.extras
    lines.append(f"expiry {o.expiry} ({(o.expiry - o.as_of).days} DTE), {p.contracts} contract(s), "
                 f"short strikes widened {o.widen_steps} delta step(s) from the band midpoint")
    for g in p.legs:
        lines.append(f"  {g.side.value:<5} {g.option_type.value:<4} {g.strike:>8g}  bid {g.bid:.2f} ask {g.ask:.2f} "
                     f"mid {g.mid:.2f}  delta {g.delta}  IV {g.iv if g.iv is None else round(g.iv, 3)}  OI {g.open_interest}")
    lines.append(f"credit ${n.credit:.2f}  max loss ${n.max_loss:.2f}  breakevens {n.breakevens}  "
                 f"PoP {n.pop:.1%}  risk {n.risk_pct_of_portfolio:.2f}% of the synthetic portfolio")
    if x.profit_zone_width is not None:
        lines.append(f"profit zone {x.profit_zone_width:.2f} wide ({x.profit_zone_pct:.1f}% of price), "
                     f"risk/reward {x.risk_reward}:1")
    v = o.verifier
    lines.append(f"VERIFIER: {v.verdict.value} ({'passes' if v.passed else 'fails'} the gate)")
    for c in v.checks:
        mark = {"CLEAR": "ok ", "CAUTION": "!  ", "INSUFFICIENT_EVIDENCE": "?  ", "BLOCK": "X  "}[c.verdict.value]
        lines.append(f"  {mark}{c.name:<22}{'' if c.hard else '(advisory) '}{c.reason}")
    return "\n".join(lines)


def cmd_build(args: argparse.Namespace) -> int:
    from .market import Market
    from .models import Strategy, Verdict
    from .tools import build_and_verify

    settings = load_settings()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    market, edgar = Market(), _edgar_or_none(settings)
    landmine, events = None, ()
    if not args.no_landmine:
        from .agent import landmine_check
        from .llm import build_llm
        lm = landmine_check(args.ticker, f"selling a {args.strategy.replace('_', ' ')}", store=_store(settings),
                            llm=build_llm(settings), as_of=as_of)
        cites = "; ".join(f"{c['form']} {c['date']} {c['item']}" for c in lm["citations"])
        landmine = (Verdict(lm["verdict"]), f"landmine {lm['verdict']}" + (f" ({cites})" if cites else "")
                    + (f": {lm['reasons'][0]}" if lm.get("reasons") else ""))
        from .agent_tools import event_facts
        events = event_facts(lm)
    iv_rank = None
    out = build_and_verify(args.ticker, Strategy(args.strategy), as_of, pb, market, edgar,
                           settings.synthetic_portfolio_value, landmine=landmine, event_dates=events)
    if out.expiry and not out.error:
        iv_rank = _atm_iv_proxy(market, args.ticker, out.expiry, out.data.profile.price,
                                market.history(args.ticker).closes, pb)
        if iv_rank:  # re-run so the advisory IV check sees the proxy
            out = build_and_verify(args.ticker, Strategy(args.strategy), as_of, pb, market, edgar,
                                   settings.synthetic_portfolio_value, landmine=landmine, iv_rank=iv_rank,
                                   event_dates=events)
    text = _render_build(out, settings)
    print(text)
    if args.masked and out.numbers:
        from .mask import Masker
        m = Masker(settings.synthetic_portfolio_value)
        view = m.value(to_jsonable({"ticker": out.ticker, "numbers": out.numbers, "proposal": out.proposal}))
        print("\nWhat the LLM sees (masked):\n" + json.dumps(view, indent=2))
    if args.save:
        payload = {"outcome": out, "masked_view": None}
        _save(settings, f"build_{out.ticker}_{args.strategy}_{as_of}", payload, text)
    return 0


def _demo_positions(as_of: date):
    from datetime import timedelta
    from .models import OptionLeg, OptionType, Position, Side, Strategy

    exp = as_of + timedelta(days=30)
    def condor(pid, opened_days_ago, credit, entry_spot=100.0, expiry=exp):
        legs = (OptionLeg(OptionType.PUT, Side.SHORT, 90, expiry, 1.0, 1.1),
                OptionLeg(OptionType.PUT, Side.LONG, 87, expiry, 0.5, 0.6),
                OptionLeg(OptionType.CALL, Side.SHORT, 110, expiry, 1.0, 1.1),
                OptionLeg(OptionType.CALL, Side.LONG, 113, expiry, 0.5, 0.6))
        return Position(pid, "TICKER_A", Strategy.IRON_CONDOR, legs, as_of - timedelta(days=opened_days_ago),
                        entry_credit=credit, entry_max_loss=300 - credit, entry_spot=entry_spot, entry_iv=0.35)
    return [
        ("winner: 55% of max profit", condor("SYN-1", 15, 100.0), dict(spot=101.0, close_cost=45.0)),
        ("stop: loss of 1.2x credit", condor("SYN-2", 10, 100.0), dict(spot=96.0, close_cost=220.0)),
        ("big drop below 200-DMA", condor("SYN-3", 9, 100.0), dict(spot=91.0, close_cost=150.0, sma200=95.0)),
        ("late, short of target: roll", condor("SYN-4", 30, 100.0, expiry=as_of + timedelta(days=15)),
         dict(spot=100.5, close_cost=70.0)),
        ("earnings in 5 days", condor("SYN-5", 12, 100.0), dict(spot=100.0, close_cost=80.0,
                                                               earnings_date=as_of + timedelta(days=5))),
        ("conflict: 50% profit and IV crush", condor("SYN-6", 14, 100.0),
         dict(spot=100.0, close_cost=48.0, iv_rank_now=30.0, iv_rank_window_ago=55.0)),
        ("missing data", condor("SYN-7", 5, 100.0), dict(spot=100.0)),
    ]


def cmd_manage_demo(args: argparse.Namespace) -> int:
    from .management import Snapshot, evaluate, heat_review
    from .mask import Masker

    settings = load_settings()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    masker = Masker(settings.synthetic_portfolio_value)
    lines = [f"== Rule engine on SYNTHETIC positions (TICKER_A), as of {as_of} =="]
    payload = []
    for label, pos, snap_kw in _demo_positions(as_of):
        snap_kw.setdefault("earnings_known_none", "earnings_date" not in snap_kw)
        ev = evaluate(pos, Snapshot(as_of=as_of, **snap_kw), pb)
        lines.append(f"\n[{label}] {masker.token(pos.position_id, 'position')}  profit {ev.profit_pct}%")
        for t in ev.triggers:
            lines.append(f"  -> {t.rule}: {t.action.value}  ({t.reason})")
        if not ev.triggers:
            lines.append("  -> no rule fired")
        if ev.not_evaluated:
            lines.append(f"  ?  not evaluated (missing data): {', '.join(ev.not_evaluated)}")
        payload.append({"label": label, "evaluation": ev})
    heat = heat_review(32.0, as_of, pb)
    lines.append(f"\nR8 heat review at 32% heat: {heat.action.value if heat else 'not the review day / under the max'}")
    text = "\n".join(lines)
    print(text)
    if args.save:
        _save(settings, f"manage_demo_{as_of}", masker.value(to_jsonable(payload)), text)
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    from .market import Market
    from .tools import screen_ticker

    settings = load_settings()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    market, edgar = Market(), _edgar_or_none(settings)
    tickers = [t.upper() for t in args.tickers] or _universe(settings)
    rows, lines = [], [f"== Quality screen, as of {as_of} (pass >= {pb['screener.quality_score_min']}) =="]
    for t in tickers:
        cand, data = screen_ticker(t, as_of, pb, market, edgar)
        rows.append(cand)
        c = cand.score_components
        lines.append(f"{t:<5} {cand.tier or '-':<7} score {cand.quality_score:>5.2f}  "
                     f"[analyst {c['analyst']:.2f} | fund {c['fundamentals']:.2f} | trend {c['trend']:.2f} | "
                     f"insider {c['insider']:.2f}]  {'PASS' if cand.passed_screen else 'fail'}")
        for r in cand.reasons:
            lines.append(f"        {r}")
        for n in data.notes:
            lines.append(f"        note: {n}")
    scores = sorted(c.quality_score for c in rows)
    lines.append(f"\ndistribution: min {scores[0]:.2f}  median {scores[len(scores) // 2]:.2f}  max {scores[-1]:.2f}  "
                 f"| {sum(c.passed_screen for c in rows)}/{len(rows)} pass")
    text = "\n".join(lines)
    print(text)
    if args.save:
        _save(settings, f"screen_{as_of}", rows, text)
    return 0


# --- Phase 3: ReAct agent, briefs, Reflexion -----------------------------------------------

def _db(settings):
    from .db import Database
    return Database(settings.db_path)


def _render_brief(b: dict) -> str:
    if b["kind"] in ("exit", "roll"):
        return _render_exit(b)
    return _render_entry(b)


def _render_exit(b: dict) -> str:
    p = b["payload"]
    d = p["decision"]
    lines = [f"== Brief {b['brief_id']} [{b['status'].upper()}] {b['kind']} {b['ticker']} ({p['position_id']}) ==",
             f"as of {p['as_of']} | spot {p['spot']} | {p['dte']} DTE | run {p['run_id']}"
             + (f" | {p['simulated']}" if p.get("simulated") else "")]
    lines.append(f"position: {p['strategy']} " + " ".join(
        f"{g['side']} {g['option_type']} {g['strike']:g}" for g in p["legs"]) + f" exp {p['legs'][0]['expiry']}")
    lines.append(f"close (calculators, mid): debit ${p['close_cost']:.2f} per contract -> limit "
                 f"{p['limit_debit_per_share']:.2f}/share; entry credit ${p['entry_credit']:.2f}; est. P&L "
                 f"${p['est_pnl']:.2f} ({p['profit_pct']}% of max profit)")
    lines.append("rules fired:")
    for t in p["triggers"]:
        lines.append(f"  {t['rule']:<18}{t['action']:<18}{t['reason']}")
    if p["not_evaluated"]:
        lines.append(f"not evaluated (missing data): {', '.join(p['not_evaluated'])}")
    lines.append(f"decision: {d['action']} (by {d['chosen_by']}; options {', '.join(d['options'])}): {d['rationale']}")
    for n in d["notes"]:
        lines.append(f"  note: {n}")
    if b["kind"] == "roll":
        new = p["new"]
        lines.append("roll into (Tree-of-Thought pick, verified):")
        lines.append("  " + _render_entry({"brief_id": b["brief_id"], "status": b["status"], "kind": "new leg",
                                           "ticker": b["ticker"], "verdict": b["verdict"],
                                           "payload": {**new, "as_of": p["as_of"], "run_id": new["tot_run_id"]}}
                                          ).replace("\n", "\n  "))
    if b.get("decided_at"):
        lines.append(f"decision: {b['status']} at {b['decided_at']}: {b.get('decision_reason')}")
    return "\n".join(lines)


def _render_entry(b: dict) -> str:
    p = b["payload"]
    prop, n = p["proposal"], p["numbers"]
    lines = [f"== Brief {b['brief_id']} [{b['status'].upper()}] {b['kind']} {b['ticker']} ==",
             f"as of {p['as_of']} | verifier {b['verdict']} | playbook {p['playbook_sha256'][:12]} | run {p['run_id']}",
             f"trade: {prop['strategy']} exp {prop['legs'][0]['expiry']}, {prop['contracts']} contract(s)"]
    for g in prop["legs"]:
        lines.append(f"  {g['side']:<5} {g['option_type']:<4} {g['strike']:>8g}  mid {(g['bid'] + g['ask']) / 2:.2f}  delta {g['delta']}")
    lines.append(f"numbers (calculators): credit ${n['credit']:.2f}, max loss ${n['max_loss']:.2f}, breakevens "
                 f"{n['breakevens']}, PoP {n['pop']:.1%}, risk {n['risk_pct_of_portfolio']:.2f}% (synthetic portfolio)")
    lines.append("rules checked:")
    for c in p["verifier"]:
        lines.append(f"  {c['verdict']:<22}{c['name']:<22}{c['reason']}")
    lines.append(f"filings: {p.get('landmine_verdict')}")
    for c in p.get("citations", []):
        lines.append(f"  cite: {c['form']} {c['date']} {c['item']}  {c['url']}")
    for lesson in p.get("lessons", []):
        lines.append(f"  lesson: {lesson[:120]}")
    for r in p.get("risks", []):
        lines.append(f"  risk: {r}")
    if p.get("alternatives"):
        lines.append("alternatives considered (Tree-of-Thought):")
        for a in p["alternatives"]:
            tag = f"score {a['score']}" if a["status"] == "leaf" else a["status"]
            lines.append(f"  {a['node']:<5}{a['path']:<52}{tag}: {a['reason']}")
    lines.append(f"agent summary: {p.get('summary') or '(none)'}")
    if b.get("decided_at"):
        lines.append(f"decision: {b['status']} at {b['decided_at']}: {b.get('decision_reason')}")
    return "\n".join(lines)


def _tot_brief(args, settings, pb, as_of, store, db, masker, box) -> int:
    from .briefs import make_brief, reflect_on_failures
    from .llm import build_llm
    from .react import AgentResult
    from .tot import alternatives, explore, render_tree, summary_text

    llm = build_llm(settings)
    result = explore(box, pb, llm, db, masker)
    lines = [f"run {result.run_id}: Tree-of-Thought {result.outcome}", render_tree(result.tree, result.stats)]
    brief_id = None
    if result.selected is not None:
        brief_id = make_brief(AgentResult(result.run_id, "accepted", summary_text(result)), box, db, as_of,
                              extra={"planner": "tot", "alternatives": alternatives(result),
                                     "tot": {"selected": result.selected.id, "score": result.selected.score,
                                             "components": result.selected.detail.get("components"),
                                             "stats": result.stats}})
        lines += ["", _render_brief(db.brief(brief_id)),
                  f"\nApprove with: copilot approve {brief_id}   or reject with: copilot reject {brief_id} --reason \"...\""]
    else:
        lesson = reflect_on_failures(box, llm, store, masker)
        if lesson is not None:
            lines.append(f"Reflexion (verifier): {lesson.lesson}")
        lines.append(f"No brief: {summary_text(result)}")
    text = "\n".join(lines)
    print(text)
    if args.save:
        _save(settings, f"brief_tot_{box.ticker}_{as_of}", {"run": db.run(result.run_id),
              "tree": json.loads(result.tree.to_json()), "brief": db.brief(brief_id) if brief_id else None}, text)
    return 0 if brief_id else 1


def cmd_brief(args: argparse.Namespace) -> int:
    from .agent_tools import ToolBox
    from .briefs import make_brief, reflect_on_failures
    from .llm import build_llm
    from .market import Market
    from .mask import Masker
    from .react import run_agent
    from .tools import LiveSources

    settings = load_settings()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    store, llm = _store(settings), build_llm(settings, num_ctx=16384)
    db = _db(settings)
    masker = Masker(settings.synthetic_portfolio_value)
    box = ToolBox(args.ticker, LiveSources(pb, as_of, Market(), _edgar_or_none(settings), store, build_llm(settings)),
                  pb, as_of, settings.synthetic_portfolio_value, open_max_losses=_open_losses(db),
                  iv_history=tuple(v for _, v in db.iv_history(args.ticker.upper())))
    if args.tot:
        return _tot_brief(args, settings, pb, as_of, store, db, masker, box)
    task = f"Plan one entry trade on {box.ticker} as of {as_of.isoformat()}."
    result = run_agent(task, box, llm, db, masker, pb, max_steps=args.max_steps)
    lines = [f"run {result.run_id}: {result.outcome} after {result.stats['steps']} step(s), "
             f"{result.stats['tool_calls']} tool call(s) ({result.stats['bad_calls']} malformed), "
             f"{result.stats['eval_tokens']} tokens"]
    for row in db.trace(result.run_id):
        if row["role"] == "tool":
            t = json.loads(row["content"])
            obs = t["observation"]
            short = obs.get("error") or obs.get("verdict") or obs.get("proposal_id") or "ok"
            lines.append(f"  -> {row['name']}({json.dumps(t['arguments'])})  <- {short}")
    brief_id = make_brief(result, box, db, as_of)
    lesson = reflect_on_failures(box, build_llm(settings), store, masker)
    if lesson is not None:
        lines.append(f"Reflexion (verifier): {lesson.lesson}")
    if brief_id:
        lines.append("")
        lines.append(_render_brief(db.brief(brief_id)))
        lines.append(f"\nApprove with: copilot approve {brief_id}   or reject with: copilot reject {brief_id} --reason \"...\"")
    else:
        lines.append(f"No brief: {result.final_text.strip() or result.outcome}")
    text = "\n".join(lines)
    print(text)
    if args.save:
        _save(settings, f"brief_{box.ticker}_{as_of}", {"run": db.run(result.run_id), "trace": db.trace(result.run_id),
              "brief": db.brief(brief_id) if brief_id else None}, text)
    return 0 if brief_id else 1


def _live_box(settings, pb, as_of, ticker, cached=False):
    from .agent_tools import ToolBox
    from .llm import build_llm
    from .market import Market
    from .tools import CachedSources, LiveSources

    src = LiveSources(pb, as_of, Market(), _edgar_or_none(settings), _store(settings), build_llm(settings))
    db = _db(settings)
    return ToolBox(ticker, CachedSources(src) if cached else src, pb, as_of, settings.synthetic_portfolio_value,
                   open_max_losses=_open_losses(db), iv_history=tuple(v for _, v in db.iv_history(ticker.upper())))


def cmd_tot(args: argparse.Namespace) -> int:
    from .llm import build_llm
    from .mask import Masker
    from .tot import explore, render_tree, summary_text

    settings = load_settings()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    db = _db(settings)
    box = _live_box(settings, pb, as_of, args.ticker)
    result = explore(box, pb, None if args.no_llm else build_llm(settings), db,
                     Masker(settings.synthetic_portfolio_value), use_critic=False if args.no_critic else None,
                     exhaustive=args.exhaustive)
    mode = " | EXHAUSTIVE (no beam)" if args.exhaustive else ""
    text = "\n".join([f"== Tree-of-Thought {box.ticker} as of {as_of} | run {result.run_id} | {result.outcome}{mode} ==",
                      render_tree(result.tree, result.stats), summary_text(result),
                      "(no brief written; use `brief TICKER --tot` to write one)"])
    print(text)
    if args.save:
        _save(settings, f"tot_{box.ticker}_{as_of}" + ("_exhaustive" if args.exhaustive else ""), {"run": db.run(result.run_id), "outcome": result.outcome,
              "reasons": result.reasons, "stats": result.stats, "tree": json.loads(result.tree.to_json())}, text)
    return 0


def _prewarm(box, pb) -> None:
    """Fetch market data, the landmine verdict, and every eligible chain before timing."""
    from .market import eligible_expiries
    d = box._data()
    box.search_filings()
    for e in eligible_expiries(box.sources.expiries(box.ticker), box.as_of, pb["entry.dte_min"], pb["entry.dte_max"],
                               pb["entry.dte_target"], d.profile.earnings_date):
        box.sources.chain(box.ticker, e)


def cmd_tot_compare(args: argparse.Namespace) -> int:
    import time

    from .agent_tools import ToolBox
    from .llm import build_llm
    from .mask import Masker
    from .react import run_agent
    from .tot import explore, render_tree, score_proposal

    settings = load_settings()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    db = _db(settings)
    masker = Masker(settings.synthetic_portfolio_value)
    shared = _live_box(settings, pb, as_of, args.ticker, cached=True)
    _prewarm(shared, pb)  # both planners then see identical data and one landmine verdict
    fresh = lambda: ToolBox(shared.ticker, shared.sources, pb, as_of, settings.synthetic_portfolio_value,  # noqa: E731
                            open_max_losses=shared.open_max_losses, iv_history=shared.iv_history)

    def pick(box) -> dict:
        if box.accepted is None:
            return {"trade": None, "verdict": None, "score": None}
        pid, res = box.accepted
        prop = box.proposals[pid]
        legs = " ".join(f"{g.side.value[0]}{g.option_type.value[0].upper()}{g.strike:g}" for g in prop.proposal.legs)
        return {"trade": f"{prop.proposal.strategy.value} {prop.proposal.expiry} {legs}", "verdict": res.verdict.value,
                "pop": prop.numbers.pop, "score": score_proposal(box, pb, pid)[0]}

    rbox = fresh()
    t0 = time.monotonic()
    react = run_agent(f"Plan one entry trade on {rbox.ticker} as of {as_of.isoformat()}.", rbox,
                      build_llm(settings, num_ctx=16384), db, masker, pb, max_steps=args.max_steps)
    react_ms = int((time.monotonic() - t0) * 1000)
    linear = {"planner": "ReAct (linear)", "run_id": react.run_id, "outcome": react.outcome,
              "nodes_explored": len(rbox.proposals),
              "nodes_pruned": sum(not r.passed for _, r in rbox.submissions),
              "llm_calls": react.stats["steps"], "latency_ms": react_ms, **pick(rbox)}

    tbox = fresh()
    tot = explore(tbox, pb, build_llm(settings), db, masker)
    tree = {"planner": "Tree-of-Thought", "run_id": tot.run_id, "outcome": tot.outcome,
            "nodes_explored": tot.stats["nodes"], "nodes_pruned": tot.stats["pruned"],
            "llm_calls": tot.stats["llm_calls"], "latency_ms": tot.stats["latency_ms"], **pick(tbox),
            "critic_score": tot.selected.score if tot.selected else None}

    rows = [("outcome", "outcome"), ("nodes explored", "nodes_explored"), ("nodes pruned", "nodes_pruned"),
            ("LLM calls", "llm_calls"), ("latency (s)", "latency_ms"), ("chosen trade", "trade"),
            ("verifier", "verdict"), ("PoP", "pop"), ("score", "score"), ("score w/ critic", "critic_score")]

    def fmt(key, v):
        if v is None:
            return "-"
        return f"{v / 1000:.1f}" if key == "latency_ms" else f"{v:.1%}" if key == "pop" else str(v)
    lines = [f"== ReAct vs Tree-of-Thought: {shared.ticker} as of {as_of} ==",
             f"filings verdict {shared.landmine_result['verdict']} (one shared check); data prefetched, so latency "
             "is planning only. Scores use the same deterministic tot.weights for both picks.",
             "score = deterministic tot.weights score of the chosen trade; the ToT critic may only lower it.",
             f"{'':<17}{'ReAct (linear)':<52}{'Tree-of-Thought':<52}"]
    for label, key in rows:
        lines.append(f"{label:<17}{fmt(key, linear.get(key)):<52}{fmt(key, tree.get(key)):<52}".rstrip())
    lines += ["", "ToT tree:", render_tree(tot.tree, tot.stats)]
    text = "\n".join(lines)
    print(text)
    if args.save:
        _save(settings, f"tot_compare_{shared.ticker}_{as_of}", {"react": linear, "tot": tree,
              "react_trace": db.trace(react.run_id), "tot_tree": json.loads(tot.tree.to_json())}, text)
    return 0


def _open_losses(db) -> tuple[float, ...]:
    from .broker import open_positions
    return tuple(p.entry_max_loss * p.contracts for p in open_positions(db))


def _live_sources(settings, pb, as_of):
    from .llm import build_llm
    from .market import Market
    from .tools import CachedSources, LiveSources
    return CachedSources(LiveSources(pb, as_of, Market(), _edgar_or_none(settings), _store(settings), build_llm(settings)))


def cmd_fill(args: argparse.Namespace) -> int:
    from .broker import PaperBroker
    from .llm import build_llm
    from .mask import Masker

    settings = load_settings()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    db = _db(settings)
    broker = PaperBroker(db, _live_sources(settings, pb, as_of), as_of, llm=build_llm(settings),
                         store=_store(settings), masker=Masker(settings.synthetic_portfolio_value))
    tickets = [db.ticket(args.ticket_id)] if args.ticket_id else db.tickets("approved")
    if not tickets or tickets == [None]:
        print("no approved tickets to fill")
        return 0
    for t in tickets:
        f = broker.process(t)
        print(f"{f.ticket_id}: {f.status.upper()}  {f.message}")
    return 0


def cmd_positions(args: argparse.Namespace) -> int:
    from .models import position_from_dict
    db = _db(load_settings())
    rows = db.positions(None if args.all else "open")
    if not rows:
        print("no positions" if args.all else "no open positions")
    for r in rows:
        p = position_from_dict(r["payload"])
        legs = " ".join(f"{g.side.value[0]}{g.option_type.value[0].upper()}{g.strike:g}" for g in p.legs)
        print(f"{r['position_id']}  {r['status']:<7}{p.ticker:<6}{p.strategy.value:<19}{legs:<30} exp {p.expiry}  "
              f"opened {p.opened_on}  credit {p.entry_credit:.2f}  max loss {p.entry_max_loss:.2f}  (synthetic)")
    return 0


def cmd_journal(args: argparse.Namespace) -> int:
    from .journal import metrics, render
    settings = load_settings()
    db = _db(settings)
    rows = db.journal()
    m = metrics(rows, args.month)
    text = render([r for r in rows if not args.month or r["date_opened"].startswith(args.month)], m)
    print(text)
    if args.save:
        _save(settings, f"journal_{args.month or 'all'}", {"rows": rows, "metrics": m}, text)
    return 0


def cmd_iv_snapshot(args: argparse.Namespace) -> int:
    from .indicators import atm_iv
    from .market import Market

    settings = load_settings()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    db, market = _db(settings), Market()
    tickers = [t.upper() for t in args.tickers] or _universe(settings)
    for t in tickers:
        try:
            spot = market.profile(t, as_of).price
            exps = [e for e in market.expiries(t) if (e - as_of).days >= 20]
            if not exps:
                raise ValueError("no expiry 20+ days out")
            exp = min(exps, key=lambda e: abs((e - as_of).days - 30))   # constant ~30-day maturity
            iv = atm_iv(market.chain(t, exp)[0], spot)
            if iv is None:
                raise ValueError("no IV near the money")
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the job
            print(f"{t}: skipped ({exc})")
            continue
        db.add_iv_snapshot(t, as_of.isoformat(), round(iv, 4), exp.isoformat(), (exp - as_of).days)
        n = len(db.iv_history(t))
        print(f"{t}: ATM IV {iv:.1%} ({exp}, {(exp - as_of).days} DTE); {n} snapshot(s), "
              f"percentile from {pb['iv_rank.percentile_min_snapshots']}, rank from {pb['iv_rank.full_history_days']}")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    from .llm import build_llm
    from .mask import Masker
    from .monitor import render, run_monitor

    settings = load_settings()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    db = _db(settings)
    masker = Masker(settings.synthetic_portfolio_value)
    llm = None if args.no_llm else build_llm(settings)
    run_id, items, summary = run_monitor(db, _live_sources(settings, pb, as_of), pb, as_of, llm=llm, masker=masker,
                                         portfolio_value=settings.synthetic_portfolio_value, store=_store(settings),
                                         tot_llm=llm)
    text = f"run {run_id}\n" + render(items, summary)
    for it in items:
        if it.brief_id:
            text += "\n\n" + _render_brief(db.brief(it.brief_id))
    print(text)
    if args.save:
        _save(settings, f"monitor_{as_of}", {"run": db.run(run_id), "summary": summary,
              "briefs": [db.brief(i.brief_id) for i in items if i.brief_id]}, text)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo
    from .llm import build_llm
    from .market import Market
    from .mask import Masker
    from .tools import screen_ticker

    settings = load_settings()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    sources = _live_sources(settings, pb, as_of)
    market, edgar = Market(), _edgar_or_none(settings)

    def ask(brief_id: str, text: str) -> bool:
        if not sys.stdin.isatty():
            print(f"\n[no terminal: {brief_id} is NOT approved. Run the demo in your own terminal to approve.]")
            return False
        return input(f"\nApprove {brief_id}? This writes a paper ticket only. [y/N] ").strip().lower() in ("y", "yes")

    res = run_demo(args.ticker, sources=sources, pb=pb, as_of=as_of, llm=build_llm(settings),
                   masker=Masker(settings.synthetic_portfolio_value), store=_store(settings),
                   portfolio_value=settings.synthetic_portfolio_value, approve_fn=ask, render_brief=_render_brief,
                   screen_fn=None if args.skip_screen else (lambda t: screen_ticker(t, as_of, pb, market, edgar)),
                   move_pct=args.move_pct)
    print(f"\n(demo stopped at: {res.stopped_at})")
    if args.save:
        _save(settings, f"demo_e2e_{args.ticker.upper()}_{as_of}", {"stopped_at": res.stopped_at, "steps": res.steps},
              res.text + f"\n\n(demo stopped at: {res.stopped_at})")
    return 0 if res.stopped_at == "done" else 1


def cmd_briefs(args: argparse.Namespace) -> int:
    for b in _db(load_settings()).briefs(args.status):
        print(f"{b['brief_id']}  {b['status']:<9} {b['kind']:<6} {b['ticker']:<5} {b['verdict'] or '-':<22} {b['created_at']}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    b = _db(load_settings()).brief(args.brief_id)
    if b is None:
        print(f"no brief {args.brief_id}", file=sys.stderr)
        return 1
    print(_render_brief(b))
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    from .briefs import approve
    db = _db(load_settings())
    try:
        ticket_id = approve(db, args.brief_id, args.note)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    t = next(t for t in db.tickets() if t["ticket_id"] == ticket_id)
    print(f"approved {args.brief_id}; wrote order ticket {ticket_id} (paper only, nothing was sent anywhere)")
    print(json.dumps(t["payload"], indent=2))
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    from .briefs import reject
    from .llm import build_llm
    from .mask import Masker
    settings = load_settings()
    try:
        lesson = reject(_db(settings), args.brief_id, args.reason, llm=build_llm(settings), store=_store(settings),
                        masker=Masker(settings.synthetic_portfolio_value))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"rejected {args.brief_id}. Reflexion lesson stored: {lesson.lesson}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    db = _db(load_settings())
    run = db.run(args.run_id)
    if run is None:
        print(f"no run {args.run_id}", file=sys.stderr)
        return 1
    print(json.dumps(run, indent=2))
    spans = db.spans(args.run_id)
    if spans and not args.full:
        print(_render_spans(spans))
        print("(use --full for every prompt, reply and tool call)")
        return 0
    for row in db.trace(args.run_id):
        body = row["content"] if len(row["content"]) < 600 or args.full else row["content"][:600] + " ..."
        print(f"[{row['seq']:>3}] {row['role']:<9}{row['name'] or '':<18}{body}")
    return 0


def _render_spans(spans: list[dict]) -> str:
    """Indented span timeline: session > supervisor > the agent it routed to."""
    from datetime import datetime
    kids: dict[str | None, list[dict]] = {}
    for s in spans:
        kids.setdefault(s["parent_id"], []).append(s)
    t0 = next((datetime.fromisoformat(s["started_at"]) for s in spans if s["started_at"]), None)
    lines = [f"{'offset':>8} {'ms':>7}  span"]

    def walk(s: dict, depth: int) -> None:
        off = "" if not s["started_at"] or t0 is None else \
            f"{(datetime.fromisoformat(s['started_at']) - t0).total_seconds():7.1f}s"
        out = s.get("output_summary") or ""
        lines.append(f"{off:>8} {s['duration_ms'] or 0:>7}  {'  ' * depth}{s['agent']:<14}[{s['pattern'] or '-'}]"
                     f"  llm {s['llm_calls'] or 0}  {out[:90]}" + (f"  ERROR {s['error']}" if s.get("error") else ""))
        for k in kids.get(s["span_id"], []):
            walk(k, depth + 1)

    for root in kids.get(None, []):
        walk(root, 0)
    totals: dict[str, list[int]] = {}
    for s in spans:
        if s["agent"] != "session":
            t = totals.setdefault(s["agent"], [0, 0, 0])
            t[0] += 1
            t[1] += s["duration_ms"] or 0
            t[2] += s["llm_calls"] or 0
    lines.append("per agent: " + "; ".join(f"{a} x{n} {ms / 1000:.1f}s llm {c}" for a, (n, ms, c) in totals.items()))
    return "\n".join(lines)


def cmd_ma(args: argparse.Namespace) -> int:
    from .llm import build_llm
    from .mask import Masker
    from .multi_agent.runner import run_multi_agent

    settings = load_settings()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    db = _db(settings)
    r = run_multi_agent(args.ticker, sources=_live_sources(settings, pb, as_of), pb=pb, as_of=as_of,
                        llm=None if args.no_llm else build_llm(settings, num_ctx=16384), db=db,
                        masker=Masker(settings.synthetic_portfolio_value),
                        portfolio_value=settings.synthetic_portfolio_value, store=_store(settings),
                        open_max_losses=_open_losses(db), use_agents=not args.no_agents,
                        graph_out=settings.repo_root / "demo_outputs" / "agent_graph.md")
    text = _render_ma(r, db)
    print(text)
    if args.save:
        _save(settings, f"ma_{args.ticker.upper()}_{as_of}", {"run": db.run(r.run_id), "state": r.state,
              "spans": db.spans(r.run_id), "brief": db.brief(r.state["brief_id"]) if r.state.get("brief_id") else None},
              text)
    return 0


def _render_ma(r, db) -> str:
    s = r.state
    research = s.get("research") or {}
    lines = [f"== Multi-agent run {r.run_id} ({s['ticker']} as of {s['as_of']}): {r.outcome} ==",
             f"route: {' -> '.join(s.get('route_log', []))}",
             f"researcher: filings {research.get('verdict')}, rewrites {research.get('rewrites', 0)}; attempts "
             + "; ".join(f"{a['verdict']} (coverage {'ok' if a['coverage_ok'] else 'FAILED'}, relevant "
                         f"{a['relevant']})" for a in research.get("attempts", [])),
             f"critic rounds {s.get('critic_rounds', 0)}; LLM calls {r.llm_calls}; builds {r.builds}; MCP calls "
             f"{r.mcp_calls}; latency {r.latency_ms / 1000:.1f}s; agents {', '.join(r.agents)}; messages "
             f"{len(s.get('messages', []))}"]
    if s.get("chosen"):
        lines.append(f"chosen: {s['chosen']['path']} (score {s['chosen']['score']}), gate verdict "
                     f"{(s.get('verifier') or {}).get('verdict')}")
    for why in s.get("reasons", []):
        lines.append(f"reason: {why}")
    for e in s.get("errors", [])[:6]:
        lines.append(f"note: {e}")
    lines += ["", "spans:", _render_spans(db.spans(r.run_id))]
    if s.get("brief_id"):
        lines += ["", _render_brief(db.brief(s["brief_id"])),
                  f"\nApprove with: copilot approve {s['brief_id']}   or reject with: copilot reject {s['brief_id']} "
                  "--reason \"...\""]
    if s.get("lesson"):
        lines.append(f"lesson (learner): {s['lesson']}")
    return "\n".join(lines)


def cmd_ma_compare(args: argparse.Namespace) -> int:
    import time

    from .agent_tools import ToolBox
    from .llm import build_llm
    from .mask import Masker
    from .multi_agent.runner import run_multi_agent
    from .react import run_agent
    from .tot import explore, score_proposal

    settings = load_settings()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    db = _db(settings)
    masker = Masker(settings.synthetic_portfolio_value)
    shared = _live_box(settings, pb, as_of, args.ticker, cached=True)
    _prewarm(shared, pb)   # one landmine check and every eligible chain, shared by all three planners
    fresh = lambda: ToolBox(shared.ticker, shared.sources, pb, as_of, settings.synthetic_portfolio_value,  # noqa: E731
                            open_max_losses=shared.open_max_losses, iv_history=shared.iv_history)

    def pick(box) -> dict:
        if box.accepted is None:
            return {"trade": None, "verdict": None, "score": None}
        pid, res = box.accepted
        prop = box.proposals[pid]
        legs = " ".join(f"{g.side.value[0]}{g.option_type.value[0].upper()}{g.strike:g}" for g in prop.proposal.legs)
        return {"trade": f"{prop.proposal.strategy.value} {prop.proposal.expiry} {legs}", "verdict": res.verdict.value,
                "score": score_proposal(box, pb, pid)[0]}

    rows = {}
    rbox = fresh()
    t0 = time.monotonic()
    react = run_agent(f"Plan one entry trade on {rbox.ticker} as of {as_of.isoformat()}.", rbox,
                      build_llm(settings, num_ctx=16384), db, masker, pb, max_steps=args.max_steps)
    rows["ReAct (single)"] = {"run_id": react.run_id, "outcome": react.outcome, "llm_calls": react.stats["steps"],
                              "latency_ms": int((time.monotonic() - t0) * 1000), "agents": 1,
                              "messages": len(db.trace(react.run_id)), "critic_rounds": 0, "rewrites": 0,
                              "builds": len(rbox.proposals), **pick(rbox)}
    tbox = fresh()
    tot = explore(tbox, pb, build_llm(settings), db, masker)
    rows["ToT (single)"] = {"run_id": tot.run_id, "outcome": tot.outcome, "llm_calls": tot.stats["llm_calls"],
                            "latency_ms": tot.stats["latency_ms"], "agents": 1, "messages": len(db.trace(tot.run_id)),
                            "critic_rounds": tot.stats.get("critic_runs", 0),
                            "rewrites": 0, "builds": tot.stats["builds"], **pick(tbox)}
    ma = run_multi_agent(shared.ticker, sources=shared.sources, pb=pb, as_of=as_of,
                         llm=build_llm(settings, num_ctx=16384), db=db, masker=masker,
                         portfolio_value=settings.synthetic_portfolio_value, store=_store(settings),
                         open_max_losses=shared.open_max_losses,
                         graph_out=settings.repo_root / "demo_outputs" / "agent_graph.md")
    from .mcp_server import MCPContext  # noqa: F401 - the MA box lives in the run's MCP context
    mbox_pick = {"trade": None, "verdict": None, "score": None}
    if ma.state.get("chosen"):
        v = ma.state["verifier"]
        legs = " ".join(f"{g['side'][0]}{g['type'][0].upper()}{g['strike']:g}" for g in v["legs"])
        mbox_pick = {"trade": f"{v['strategy']} {v['expiry']} {legs}", "verdict": v["verdict"],
                     "score": ma.state["chosen"]["score"]}
    rows["Multi-agent"] = {"run_id": ma.run_id, "outcome": ma.outcome, "llm_calls": ma.llm_calls,
                           "latency_ms": ma.latency_ms, "agents": len([a for a in ma.agents if a != "supervisor"]),
                           "messages": len(ma.state.get("messages", [])), "critic_rounds": ma.state.get("critic_rounds", 0),
                           "rewrites": (ma.state.get("research") or {}).get("rewrites", 0), "builds": ma.builds,
                           **mbox_pick}
    fields = [("outcome", "outcome"), ("chosen trade", "trade"), ("verifier", "verdict"), ("score", "score"),
              ("LLM calls", "llm_calls"), ("latency (s)", "latency_ms"), ("builds", "builds"), ("agents", "agents"),
              ("messages", "messages"), ("critic rounds", "critic_rounds"), ("query rewrites", "rewrites")]
    fmt = lambda k, v: "-" if v is None else f"{v / 1000:.1f}" if k == "latency_ms" else str(v)  # noqa: E731
    names = list(rows)
    lines = [f"== Single agent vs multi-agent: {shared.ticker} as of {as_of} ==",
             f"shared inputs: prefetched quotes and chains, one default filings check "
             f"({shared.landmine_result['verdict']}); the multi-agent researcher may add one rewritten query.",
             "score = deterministic tot.weights score of the chosen trade (multi-agent: after its critic).",
             f"{'':<16}" + "".join(f"{n:<46}" for n in names)]
    for label, key in fields:
        lines.append((f"{label:<16}" + "".join(f"{fmt(key, rows[n].get(key)):<46}" for n in names)).rstrip())
    lines += ["", "multi-agent spans:", _render_spans(db.spans(ma.run_id))]
    text = "\n".join(lines)
    print(text)
    if args.save:
        _save(settings, f"ma_compare_{shared.ticker}_{as_of}", {"rows": rows, "ma_state": ma.state,
              "ma_spans": db.spans(ma.run_id)}, text)
    return 0


def cmd_mcp_serve(args: argparse.Namespace) -> int:
    from .mcp_server import MCPContext, serve_stdio
    from .multi_agent import privacy

    settings = load_settings()
    privacy.configure()
    pb = load_playbook(settings.playbook_path)
    as_of = _as_of(args)
    db = _db(settings)
    serve_stdio(MCPContext(_live_sources(settings, pb, as_of), pb, as_of, settings.synthetic_portfolio_value, db,
                           _open_losses(db)))
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

    from .models import Strategy
    p = sub.add_parser("build", help="build a trade from the live chain and run the verifier")
    p.add_argument("ticker")
    p.add_argument("--strategy", default="iron_condor", choices=[s.value for s in Strategy])
    p.add_argument("--as-of", help="YYYY-MM-DD (default today; market data is always current)")
    p.add_argument("--no-landmine", action="store_true", help="skip the RAG check (the verifier then escalates)")
    p.add_argument("--masked", action="store_true", help="also print the masked view the LLM would get")
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("manage-demo", help="run the 8 management rules on synthetic positions")
    p.add_argument("--as-of")
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_manage_demo)

    p = sub.add_parser("screen", help="quality composite for the universe (or named tickers)")
    p.add_argument("tickers", nargs="*")
    p.add_argument("--as-of")
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("brief", help="run the ReAct agent to plan one entry and write a brief")
    p.add_argument("ticker")
    p.add_argument("--max-steps", type=int, default=14)
    p.add_argument("--tot", action="store_true", help="plan with the Tree-of-Thought explorer instead of ReAct")
    p.add_argument("--as-of")
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_brief)

    p = sub.add_parser("fill", help="PaperBroker: try to fill approved tickets at their limit (synthetic)")
    p.add_argument("ticket_id", nargs="?")
    p.add_argument("--as-of")
    p.set_defaults(func=cmd_fill)

    p = sub.add_parser("positions", help="synthetic paper positions")
    p.add_argument("--all", action="store_true", help="include closed positions")
    p.set_defaults(func=cmd_positions)

    p = sub.add_parser("journal", help="trade log (Skill #7) and summary metrics")
    p.add_argument("--month", help="YYYY-MM (by date opened)")
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_journal)

    p = sub.add_parser("iv-snapshot", help="record today's ~30-day ATM IV per ticker (run once a day)")
    p.add_argument("tickers", nargs="*")
    p.add_argument("--as-of")
    p.set_defaults(func=cmd_iv_snapshot)

    p = sub.add_parser("monitor", help="run the 8 management rules on open positions; write exit/roll briefs")
    p.add_argument("--as-of")
    p.add_argument("--no-llm", action="store_true", help="rules only; conflicts take the most conservative action")
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("demo", help="end-to-end demo: screen -> brief -> approve -> fill -> monitor (simulated "
                       "clock) -> exit -> close -> lesson -> journal")
    p.add_argument("ticker")
    p.add_argument("--as-of")
    p.add_argument("--move-pct", type=float, default=0.0, help="simulated stock move after entry (default flat)")
    p.add_argument("--skip-screen", action="store_true")
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("ma", help="multi-agent entry workflow (LangGraph supervisor, CrewAI agents, MCP tools)")
    p.add_argument("ticker")
    p.add_argument("--as-of")
    p.add_argument("--no-agents", action="store_true", help="no CrewAI agents: deterministic strategist, no critic")
    p.add_argument("--no-llm", action="store_true", help="rules only (implies --no-agents)")
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_ma)

    p = sub.add_parser("ma-compare", help="single ReAct vs ToT vs multi-agent graph on the same prefetched data")
    p.add_argument("ticker")
    p.add_argument("--as-of")
    p.add_argument("--max-steps", type=int, default=14)
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_ma_compare)

    p = sub.add_parser("mcp-serve", help="serve the deterministic tools over MCP (stdio, local only)")
    p.add_argument("--as-of")
    p.set_defaults(func=cmd_mcp_serve)

    p = sub.add_parser("tot", help="explore candidate trades as a Tree-of-Thought and print the tree")
    p.add_argument("ticker")
    p.add_argument("--as-of")
    p.add_argument("--no-llm", action="store_true", help="deterministic generator only (no LLM calls)")
    p.add_argument("--no-critic", action="store_true", help="skip the optional LLM critic")
    p.add_argument("--exhaustive", action="store_true",
                   help="offline check: price every strategy x expiry x 5 deltas (no beam, no LLM)")
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_tot)

    p = sub.add_parser("tot-compare", help="linear ReAct pick vs Tree-of-Thought pick, side by side")
    p.add_argument("ticker")
    p.add_argument("--as-of")
    p.add_argument("--max-steps", type=int, default=14)
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_tot_compare)

    p = sub.add_parser("briefs", help="list briefs")
    p.add_argument("--status", choices=["pending", "approved", "rejected"])
    p.set_defaults(func=cmd_briefs)

    p = sub.add_parser("show", help="show one brief")
    p.add_argument("brief_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("approve", help="approve a pending brief (writes a paper order ticket)")
    p.add_argument("brief_id")
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser("reject", help="reject a pending brief; the reason becomes a lesson")
    p.add_argument("brief_id")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_reject)

    p = sub.add_parser("trace", help="print the masked trace of an agent run")
    p.add_argument("run_id")
    p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_trace)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
