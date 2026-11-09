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
python -m options_copilot.cli tot MSFT --exhaustive      # offline check: every strategy x expiry x 5 deltas
python -m options_copilot.cli brief MSFT --tot           # brief from the ToT pick, with alternatives considered
python -m options_copilot.cli tot-compare MSFT --save    # linear ReAct pick vs ToT pick, side by side
```

```bash
# Paper positions, monitoring, journal, end-to-end demo (Phase 4)
python -m options_copilot.cli fill                       # PaperBroker: fill approved tickets at their limit (synthetic)
python -m options_copilot.cli positions                  # open synthetic positions (--all for closed too)
python -m options_copilot.cli monitor --save             # 8 management rules on open positions; exit/roll briefs
python -m options_copilot.cli iv-snapshot                # record today's ~30-day ATM IV per ticker (once a day)
python -m options_copilot.cli journal --month 2026-10    # trade log (Skill #7) and summary metrics
python -m options_copilot.cli demo MSFT --save           # screen -> brief -> approve -> fill -> monitor -> exit -> journal
```

**PaperBroker.** Approving a brief writes a ticket; `fill` simulates it against the live chain
with limit-order semantics: an opening ticket fills at its limit credit when the live mid credit
is at least the limit, and a closing ticket fills at its limit debit when the live mid debit is at
most the limit. Otherwise it stays working. A roll fills only if both halves would, so it is never
half done. Expired positions settle at intrinsic value. Nothing is sent anywhere.

**Monitor.** For each open position it refreshes the quote, prices the close with the
calculators, and runs the 8 rules. Mandatory risk rules (stop loss, 1-sigma move, earnings,
emergency exits) close the position, and no LLM is asked. Partial actions (buy back one side) are
escalated to a full close on paper, the stricter choice. When only discretionary rules fire and
they disagree (say 50% profit versus roll), Qwen picks one of the triggered actions,
schema-constrained, with a one-line rationale; an invalid answer falls back to the most
conservative one. A close writes an exit brief. A roll writes one roll brief: the close plus a new
~45-DTE trade picked by the Tree-of-Thought explorer and verified. Every brief waits for you.
Rules whose data is missing (for example IV crush before there is IV history) are listed as not
evaluated.

**IV history.** `iv-snapshot` stores one ~30-day ATM IV per ticker per day. After 20 snapshots
the IV advisory uses an IV percentile, after 252 a true IV Rank; until then the ATM-IV /
realized-vol proxy is used, and every brief says which one.

**Journal.** One row per position with the Skill #7 columns. P&L % is of the entry credit (as in
the playbook example), return on max loss has its own column, and annualized % = P&L % x 365 /
days held. Closing a position writes a Reflexion lesson.

**End-to-end demo.** Live public data runs through screen, Tree-of-Thought brief, your approval
at the prompt, and the paper fill. A simulated clock then moves forward one day at a time
(options re-priced by Black-Scholes from their live mids, stock flat unless `--move-pct`) until a
rule writes an exit or roll brief, which you approve, and the demo closes the position, writes
the lesson, and prints the journal. The demo uses its own in-memory database, so it never touches
your real paper journal, and every simulated step is labelled. Run it in your own terminal: it
never approves on its own.

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
4. **Controller.** Beam search keeps `beam_width` nodes per depth. Depth 1 is round-robin by
   generator rank. Depth 2 is round-robin across strategies, each ordered by a deterministic
   expiry score (DTE fit, days of clearance before earnings, median open interest where the short
   strikes sit; weights in `tot.expiry_weights`), plus one extra exploration node on an expiry not
   yet in the beam. Leaves are ranked by score, with a depth limit and hard budgets on builds and LLM calls; hitting a budget
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

## Multi-agent workflow (Phase 5, Module 5)

```bash
python -m options_copilot.cli ma MSFT --save          # one run of the agent team; writes a pending brief
python -m options_copilot.cli ma-compare MSFT --save  # single ReAct vs ToT vs the team, same prefetched data
python -m options_copilot.cli trace RUN_ID            # span timeline per agent (--full for every message)
python -m options_copilot.cli mcp-serve               # the tools over MCP, stdio only, for outside clients
```

The single-agent entry path (ReAct, then ToT, then brief) becomes a team under a **supervisor**.
`brief`, `brief --tot` and `tot-compare` are unchanged.

```
START -> supervisor -> researcher -> supervisor -> strategist <-> risk_critic (through the supervisor)
      -> supervisor -> verifier_gate -> brief_writer -> learner (only on NO_TRADE) -> END
```

| Role | Built with | What it may do |
|---|---|---|
| Supervisor | LangGraph conditional edges | Routes from state with a pure function. No LLM picks a path. |
| Researcher | LangGraph node | Landmine check through MCP, an LLM relevance grader, and at most one query rewrite. |
| Strategist | CrewAI agent | Proposes ToT candidates inside the playbook bands. Read-only tools; never a number. |
| Risk Critic | CrewAI agent | Reviews verified candidates: keep, lower, or veto with an allowed citation. Python applies it and can only lower. |
| Verifier gate | Deterministic tool via MCP | Re-verifies the best candidate. Final authority. |
| Brief writer | LLM node | Prose only; sentences with digits are dropped. Every number comes from the tools. |
| Learner | Reflexion | A lesson when the run ends with no trade. |

**Supervisor pattern.** Money is at stake, so predictability beats flexibility: one hub routes
every step from typed shared state (`multi_agent/state.py`). BLOCK or INSUFFICIENT_EVIDENCE from
the researcher ends the run with an escalation, and no strategist runs. An exhausted budget (LLM
calls, builds, critic rounds, wall clock, all in `playbook.yaml` `multi_agent:`) goes straight to
the verifier gate. The only edge into `brief_writer` comes from `verifier_gate`; a test checks
this on the compiled graph. The graph is exported as Mermaid to `demo_outputs/agent_graph.md`.

**Communication patterns** (each recorded on the spans and in the state's message log):
- *One-way pipeline:* supervisor -> researcher -> strategist, gate -> brief writer -> learner.
- *Two-way:* strategist <-> risk critic, at most `max_critic_rounds` (default 2). The critic's
  reviews go back to the strategist as feedback (wrapped as data), and it proposes new deltas
  under the expiries the critic hit.
- *Brainstorming:* the Tree-of-Thought branching inside the strategist.

**Researcher's RAG loop.** The grader marks which retrieved excerpts are about a material event.
If fewer than `min_relevant_chunks` are relevant (or nothing was retrieved), the query is rewritten
once and retrieval runs again. The coverage guard is unchanged. A rewrite can cure a coverage miss
(INSUFFICIENT_EVIDENCE only because too little was retrieved), but a BLOCK or CAUTION from either
query always stands, and the verifier sees every citation and event date from both queries.

**MCP.** `options_copilot/mcp_server.py` exposes `get_quote`, `get_option_chain`,
`get_earnings_date`, `search_filings`, `search_lessons`, `build_and_verify`, and `score_trade`, plus
the resources `tot://{run_id}` and `trace://{run_id}`. During a run the graph and the agents call it
through an in-memory MCP client session on the same cached data; `mcp-serve` exposes the same
server over stdio. There is no network listener and no token. Tests check that every tool returns
exactly what the direct function returns.

**Observability.** Each run is a session and each node execution is a span in SQLite (`spans`):
parent span, agent, start and end, duration, LLM calls, masked input and output summaries, error,
and the communication pattern. Every LLM prompt and reply, including the CrewAI agents', goes to
the masked trace. `trace RUN_ID` prints the timeline. No cloud tracing: CrewAI telemetry and
tracing, OpenTelemetry exporters, and LangSmith/LangChain tracing are forced off at startup
(`multi_agent/privacy.py`, tested).

**Local LLM only.** Every CrewAI agent runs on `CopilotLLM`, a small adapter over the same Ollama
client (`qwen3:8b`), with the Claude fallback only if Ollama fails and a key is set. Building an
agent with any other LLM raises, and the OpenAI endpoint is pointed at a closed local port so a
stray default call fails on this machine.

**What stays single-path and deterministic, and why.** Monitoring, the calculators, the verifier,
and the scorer are tools, not agents. Their answers must be the same every time for the same
inputs, they are what the safety rules rest on, and none of them improves with debate: a second
agent cannot make a Black-Scholes price or a stop-loss rule more correct, only less predictable.
Monitoring decisions are already rule-first, and the LLM only resolves genuine conflicts.

**Fine-tuning (LoRA): considered, not built.** The rules are config the owner edits, the math is
deterministic Python, and fresh facts come from RAG over filings. A fine-tuned model would add
training data, evaluation and safety-review cost, and it would not fix any failure we have seen:
the failures so far were search breadth, data coverage and number hallucination in prose, which
the beam, the coverage guard and code-filled numbers already address.

**Side-by-side runs** (`ma-compare TICKER --save`, live public data, US market hours, qwen3:8b
on this laptop; score is the deterministic `tot.weights` score after any critic):

| Ticker | ReAct (single) | ToT (single) | Multi-agent |
|---|---|---|---|
| MSFT | no_trade, 14 LLM calls, 128 s | SELECTED call spread 580/595, 0.5457, 9 calls, 127 s | same trade, 0.382 after 2 critic rounds, 1 query rewrite, 16 calls, 247 s |
| META | budget exhausted, 14 calls, 123 s | SELECTED call spread 695/715, 0.7235, 9 calls, 172 s | SELECTED a safer call spread 710/730, 0.5219, 1 critic round, 14 calls, 312 s |
| NVDA | no_trade, 14 calls, 195 s | NO_TRADE (no expiry inside the DTE window before earnings), 1 call, 19 s | NO_TRADE, same reason, 3 calls, 46 s, lesson written |

**Where the multi-agent version helped.** On META the critic sent three expiries back and the team
chose a further out-of-the-money spread than single ToT: a more conservative trade, not just a
lower score. The researcher's rewrite (MSFT) adds a second retrieval at the cost of one LLM call.
Every hand-off is a span, so `trace RUN_ID` shows who decided what.

**Where it did not help.** It is about 1.8 to 2 times slower than single ToT and never safer than
the gate: the safety guarantees come from the deterministic verifier either way, and on NVDA the
earnings rule stopped all three modes. On MSFT it picked the same trade as single ToT. With
qwen3:8b the CrewAI Strategist first echoed the JSON schema instead of answering (every expansion
fell back to the enumerator) until the task showed an example object instead. The Risk Critic's
citations can be loose (it cited a check that had passed); that is harmless only because Python
lets it lower, never raise.

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
| 4 | Journal, PaperBroker, monitor, IV history, end-to-end demo | done |
| 5 | Multi-agent workflow: LangGraph supervisor, CrewAI roles, MCP server, spans | done |

## License

MIT. See [LICENSE](LICENSE).
