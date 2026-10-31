"""Command-line entry point: python -m options_copilot.cli <command>."""
from __future__ import annotations

import argparse
import sys

from .config import load_settings
from .playbook import PlaybookError, load_playbook


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="copilot", description="Options Income Copilot (educational; no live trading)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("playbook", help="validate playbook.yaml and print its key rules").set_defaults(func=cmd_playbook)
    sub.add_parser("config", help="show settings (secrets are never printed)").set_defaults(func=cmd_config)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
