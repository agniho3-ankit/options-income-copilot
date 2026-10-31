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
| **Tree-of-Thought** | Explores strategy -> expiry -> delta as a beam search: LLM generator, deterministic verifier-and-score critic, budgets, `NO_TRADE` if nothing survives. |
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

# RAG landmine check (Phase 1)
python -m options_copilot.cli seed-lessons                 # load the synthetic seed lessons
python -m options_copilot.cli ingest --universe            # 90 days of 8-K/10-Q/10-K + headlines
python -m options_copilot.cli filings MRVL                 # what is indexed for a ticker
python -m options_copilot.cli check MRVL                   # verdict with citations
python -m options_copilot.cli check MRVL --no-rag          # baseline from model memory only
python -m options_copilot.cli compare MRVL --save          # both, side by side, saved to demo_outputs/
python -m options_copilot.cli reflect "lesson text" --ticker MRVL --trigger rejection

# Calculators, verifier, management, screener (Phase 2)
python -m options_copilot.cli build MSFT                          # iron condor from the live chain, verified
python -m options_copilot.cli build MSFT --strategy put_credit_spread --masked
python -m options_copilot.cli manage-demo                         # the 8 rules on synthetic positions
python -m options_copilot.cli screen                              # 0-10 quality composite for the universe
```

`build` picks the expiry nearest 45 DTE inside 30-60 (preferring one before earnings), picks short
strikes by delta (0.20-0.30), and widens them inside the band until PoP passes. It sets wings at
3% of price and then runs every verifier check against the synthetic portfolio. Hard checks BLOCK.
Advisory checks add CAUTION only. Any missing fact gives `INSUFFICIENT_EVIDENCE`, which fails the
gate. `--masked` prints the view the LLM gets: account values as % of portfolio and IDs as tokens.

```bash
# ReAct agent, approval briefs, Reflexion (Phase 3)
python -m options_copilot.cli brief MSFT                 # agent plans one entry with tools; writes a pending brief
python -m options_copilot.cli briefs --status pending
python -m options_copilot.cli show BRIEF_ID
python -m options_copilot.cli approve BRIEF_ID           # writes a paper order ticket; nothing is sent anywhere
python -m options_copilot.cli reject BRIEF_ID --reason "too close to resistance"   # reason becomes a lesson
python -m options_copilot.cli trace RUN_ID               # the masked step-by-step trace
```

The agent (Qwen through Ollama tool calling) has ten tools: market data, the filings check,
lessons, `build_trade`, `price_trade`, `submit_trade`, and `flag_concern`. It has no approve tool.
Observations are masked before the model sees them, and every step is logged to
`data/copilot.db`. When the verifier rejects a trade, the reasons come back to the agent so it can
revise. If a failure applies to the whole ticker (earnings within 7 days, a filings BLOCK, tier, or
insider BLOCK), the agent is told to stop. Verifier failures, rejections, and (later) closed trades
each produce a short Reflexion lesson in the `reflections` collection.

```bash
# Tree-of-Thought trade explorer (Phase 4, Module 4)
python -m options_copilot.cli tot MSFT --save            # explore candidate trades; print the tree
python -m options_copilot.cli tot MSFT --no-llm          # deterministic generator only
python -m options_copilot.cli brief MSFT --tot           # brief from the ToT pick, with alternatives considered
python -m options_copilot.cli tot-compare MSFT --save    # linear ReAct pick vs ToT pick, side by side
```

### How the Tree-of-Thought explorer decides (generator -> critic -> controller)

`brief TICKER` (ReAct) builds one trade and commits to it. `tot.py` instead searches a small tree
of partial trade specs: depth 1 is the strategy, depth 2 the expiry (30-60 DTE, before earnings),
depth 3 the short-delta target inside the strategy's band. Wings stay at the playbook's 3%.

1. **Landmine first.** The filings check runs once per ticker before any branching. `BLOCK` or
   `INSUFFICIENT_EVIDENCE` ends the search with no tree.
2. **Generator (LLM role).** For each node Qwen proposes up to `children_per_node` children as
   schema-constrained JSON, each with a one-line rationale. Python checks every candidate
   against the playbook and drops (and logs) any that fall outside it; nothing is clamped. If the
   LLM fails or returns nothing valid, a deterministic enumerator proposes the children.
3. **Critic (deterministic first).** Each depth-3 leaf runs build -> compute -> verify. A verifier
   failure, or a build or calculator error, prunes the branch and keeps the reason. Survivors get
   a 0-1 score from tool numbers only, weighted by `tot.weights` in `playbook.yaml`: PoP, return on
   risk, position-risk headroom, IV rank, and short-strike distance beyond support/resistance. An
   optional LLM critic may lower a score (by a playbook penalty factor) or veto a leaf, with a
   cited reason. It can never raise a score or revive a pruned node.
4. **Controller.** Beam search keeps `beam_width` nodes per depth (round-robin across parents,
   leaves by score), with a depth limit and hard budgets on builds and LLM calls; hitting a budget
   stops the search. The best surviving leaf wins. If nothing survives the result is `NO_TRADE`
   with the prune reasons; a trade is never forced.
5. **State.** Every node (spec, rationale, score, prune reason, timing) goes to the `tot_nodes`
   table under its `runs` row, and LLM calls and builds go to the masked trace.

The verifier also compares event dates in the cited filings (for example an 8-K effective date)
with the option expiry. Python reads the dates from the cited text; the LLM's statement about
timing is never used. An event inside the trade window fails the gate; one after expiry adds
CAUTION.

The first `ingest` or `seed-lessons` run downloads Chroma's built-in MiniLM embedding model
(about 80 MB, cached under your home directory). `check` and `compare` need Ollama running.

### How the landmine check decides

1. Retrieval is hard-filtered to the ticker and the 90 days up to `--as-of`, fetches 20
   candidates, and uses MMR to pick 5 diverse chunks.
2. **Coverage check.** If no chunk passes the similarity threshold, no filing (only headlines)
   matches, or fewer than 2 chunks pass, the verdict is `INSUFFICIENT_EVIDENCE` and the LLM is
   not asked.
3. Qwen reads the chunks, which are wrapped in `<<<DOC ...>>>` delimiters as untrusted data, and
   proposes a verdict with cited chunk IDs. It is offered no tools.
4. Python enforces the rules. `CLEAR` needs a cited filing. A `BLOCK` that rests only on
   headlines becomes `CAUTION`. An uncited `BLOCK` and any unparseable reply escalate.
   Invented citations are dropped. Citations always come from the stored metadata, never
   from model text.

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
| 1 | RAG landmine check (EDGAR ingest, chunking, MMR, coverage check) | done |
| 2 | Calculators, verifier, management rules, mask, screener | done |
| 3 | ReAct agent, approval briefs, Reflexion | done |
| 4 | Tree-of-Thought explorer (Module 4) | done |
| 4 | Journal, monitor, end-to-end demo | next |

## License

MIT. See [LICENSE](LICENSE).
