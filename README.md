# Options Income Copilot

> **Disclaimer.** This is an educational capstone project. It is **not financial advice**, it does
> **not** place live trades, and it has no brokerage connection. All positions are synthetic.
> Options trading carries substantial risk. Use at your own risk.

An agent that runs one person's written, rules-based options-income playbook (cash-secured puts,
credit spreads, and iron condors on large-cap US tech and financial stocks). It screens candidates,
builds trades, checks them against hard rules, monitors positions, and logs outcomes. A human
approves every entry and roll.

Built for the Carnegie Mellon University Agentic AI course capstone.

## Architecture

| Part | Role |
|---|---|
| **ReAct loop** | A local LLM (Qwen3-8B through Ollama) plans tool calls, reads observations, and re-plans. It never does arithmetic. |
| **Deterministic tools** | Python computes strikes, credit, max loss, breakevens, probability of profit, position risk, and portfolio heat. |
| **Verifier gate** | A pure rule engine checks every proposal against `playbook.yaml`. Failures go back to the agent as observations. The LLM can add caution but can never override a risk rule. |
| **RAG landmine check** | Public SEC EDGAR filings (plus local-only headlines) in a Chroma store, hard-filtered by ticker and a 90-day window, with MMR selection. Too little evidence returns `INSUFFICIENT_EVIDENCE`, never `CLEAR`. |
| **Reflexion** | Rejected briefs, verifier rejections, and closed trades produce short lessons stored for later retrieval. |
| **Memory** | `playbook.yaml` (procedural, read-only), SQLite journal and IV history, Chroma `filings` and `reflections`. |
| **Mask layer** | Normalizes prices and dollar amounts so the LLM, logs, and traces never see raw account values. |

Retrieved text is treated as data, never as instructions.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). [Ollama](https://ollama.com) with
`qwen3:8b` is needed for agent features, but not for the tests.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env      # then set SEC_USER_AGENT to "Your Name your-email"
pre-commit install
```

## Run

```bash
python -m options_copilot.cli playbook   # validate playbook.yaml and print its key rules
python -m options_copilot.cli config     # show settings (secrets are never printed)
```

More commands arrive with each build phase (see Status).

## Tests

```bash
pytest -q              # unit tests: no network, no LLM
pytest -q -m llm       # optional tests against a running Ollama model
```

Unit tests block network access, and use fake LLM and data clients.

## Data and privacy

- Market data and filings come only from free public sources (SEC EDGAR, yfinance). Downloaded
  data, the Chroma index, and the SQLite journal live in the gitignored `data/` folder and are
  rebuilt locally.
- Positions, journal rows, and seeded lessons are **synthetic** and carry `synthetic: true`.
  Unit-test fixtures use made-up tickers such as `TICKER_A`.
- `scripts/privacy_audit.py` runs before every commit, together with gitleaks. It fails on email
  addresses, home-directory paths, key-like strings, and a private, gitignored denylist.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Repo guardrails, playbook config and loader, models | done |
| 1 | RAG landmine check (EDGAR ingest, chunking, MMR, coverage check) | next |
| 2 | Calculators, verifier, management rules, mask | planned |
| 3 | ReAct agent, approval briefs, Reflexion | planned |
| 4 | Journal, monitor, end-to-end demo | planned |

## License

MIT. See [LICENSE](LICENSE).
