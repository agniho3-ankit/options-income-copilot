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

**Where the multi-agent version helped, and where it only looked like it.** On META the team
chose a different spread than single ToT (710/730 instead of 695/715), but not because of the
critic. The `tot_nodes` rows show the critic lowered all seven priced leaves by the same ×0.7
(`lower_major`, all citing `lesson:3` with the same generic reason), so the ranking did not
change, and all nine revision proposals it triggered were rejected as repeats. The pick changed
because the CrewAI Strategist proposed a 0.22 delta for the call spread, which single ToT never
tried (it tried 0.25, 0.28 and 0.30). That is the generator exploring differently, not the critic
adding judgment. On MSFT the researcher's rewrite cost one LLM call and changed nothing: the grader
found 0 relevant passages both before and after the rewrite, and the verdict stayed CLEAR. What
the multi-agent version does add is traceability: every hand-off is a span, so `trace RUN_ID`
shows who decided what.

**Where it did not help.** It is about 1.8 to 2 times slower than single ToT and never safer than
the gate: the safety guarantees come from the deterministic verifier either way, and on NVDA the
earnings rule stopped all three modes. On MSFT it picked the same trade as single ToT. With
qwen3:8b the CrewAI Strategist first echoed the JSON schema instead of answering (every expansion
fell back to the enumerator) until the task showed an example object instead. The Risk Critic's
citations can be loose (it cited a check that had passed); that is harmless only because Python
lets it lower, never raise.

## Evaluation, guardrails, human review and observability (Phase 6, Module 6)

```bash
python -m options_copilot.cli review                     # ranked queue of pending briefs (approval stays manual)
python -m options_copilot.cli review BRIEF_ID            # one brief on one screen, with three ready commands
python -m options_copilot.cli reject BRIEF_ID --code event_risk --reason "8-K event before expiry"
python -m options_copilot.cli defer BRIEF_ID             # stays pending, moves to the end of the queue
python -m options_copilot.cli metrics --since 2026-11-01 --save   # add --eval-db for the eval runs
python -m options_copilot.cli eval-snapshot CASE_ID TICKER   # freeze one case's inputs (market hours)
python -m options_copilot.cli eval --repeats 3 --save        # labeled cases, HELM-style metrics
python -m options_copilot.cli eval --ablation --save         # one guardrail off at a time (sandboxed)
```

The failures this app has actually shown, and what catches each one, are in
[docs/FAILURE_MODES.md](docs/FAILURE_MODES.md). The MAESTRO seven-layer threat map is in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

**One guardrail pipeline.** `options_copilot/guardrails.py` keeps one registry of every guardrail,
each with a stage and a kind. The checks themselves are unchanged and stay where they were; the
pipeline names them and writes every decision (`pass`, `flag` or `block`, with a reason) to a
`guardrail_events` table linked to the span that was running. `trace RUN_ID` shows the events
inline under their spans.

| Stage | Guardrail | Kind | What it does |
|---|---|---|---|
| pre | `coverage_guard` | static | too little filing evidence returns INSUFFICIENT_EVIDENCE before any LLM call |
| pre | `grader_rewrite` | runtime | relevance grade and at most one query rewrite; 0 relevant raises `grader_zero_relevant` |
| pre | `playbook_bands` | static | candidates outside the playbook strategies, DTE window or delta bands are rejected |
| during | `budgets` | static | LLM-call, build, critic-round and wall-clock caps end the run |
| during | `critic_lower_only` | static | the critic may keep, lower or veto; anything else is ignored |
| during | `critic_citation` | runtime | flags a cited check that passed, a missing lesson, or one reason pasted on every candidate |
| during | `confidence_gate` | risk score | confidence below `guardrails.min_confidence` (0.6) raises `low_confidence` |
| post | `verifier` | static | the deterministic verifier and the gate's re-verify; final |
| post | `event_date_check` | static | Python compares cited event dates with the expiry |
| post | `citation_from_metadata` | static | citations rebuilt from stored chunk metadata; invented IDs dropped |
| post | `brief_numbers` | static | brief numbers must equal the verified tool numbers, or no brief is written |
| post | `judge` | risk score | LLM-as-judge lists unsupported or contradicted claims; flag only |

**Why the order matters.** The coverage guard runs before any LLM call, so a model never gets the
chance to judge filings that are not there: missing evidence stops the run instead of being
reasoned around. The playbook bands run before a build, so nothing out of policy is priced. The
budgets and the lower-only critic run while the search is still going, so a loop or a generous
critic cannot rescue a candidate. The verifier runs before the brief, so the prose is always
written about a trade that already passed, and the judge runs last because it can only add a
flag. A test checks that each run's events go pre, then during, then post.

**Signals add caution, never remove it.** Every Phase 6 signal (`low_confidence`,
`grader_zero_relevant`, `critic_citation_unsupported`, `judge_flag`, `budget_exhausted`,
`revisions_all_duplicates`) only raises a brief's place in the review queue. None changes a
verdict, raises a score, or approves anything. Tests pin this for each signal.

**Confidence and calibration.** The landmine LLM now returns a verbalized `confidence` with its
verdict. A second short call asks whether its proposed verdict is correct (P(True), Kadavath et al.
2022). The installed Ollama returns token logprobs, so P(True) is the probability mass of the
`True` token against `False` on the first answer token; with the Claude fallback, which has no
logprobs here, the answer itself counts as 1 or 0 and is marked `answer_only`. A missing or
unparseable value escalates as `low_confidence`. The Risk Critic also gives a confidence for each
action; a low-confidence lowering still applies (it is extra caution) and raises the signal.
`eval` measures calibration as ECE and the Brier score, with a reliability table that shows the
count in every bin. With a few labeled cases the numbers are indicative only.

**LLM-as-judge.** `options_copilot/judge.py` gives the brief's prose and the code-filled facts
(numbers, verifier verdict, cautions, landmine verdict, citations) to the model and asks for every
claim the facts do not support or that contradicts them. It can only flag: an empty list adds
nothing, and a malformed reply is recorded as such. The deterministic number check stays the hard
check. Limit: the judge is the same qwen3:8b that writes the prose, so it may prefer its own
wording (self-preference). Its accuracy is measured against the owner's faithful and unfaithful
labels in `eval`.

**Evaluation harness.** Cases live in `eval/cases.local.yaml` (gitignored). Each points at real
past runs and holds the owner's label: the expected landmine verdict, whether a trade is
acceptable or the right outcome is NO_TRADE or BLOCK, and optionally the acceptable strategies.
`eval` refuses a case without a label. `eval-snapshot` freezes each case's inputs during market
hours into `data/eval_snapshots/` (gitignored): ticker data, expiries and chains, the ticker's
in-window filing chunks with their embeddings copied from the local Chroma, and the lessons. A
replay rebuilds an in-memory Chroma from the snapshot, so everything except LLM sampling is
identical. Eval runs use the multi-agent graph (the only pipeline with every guardrail), its own
database, and a sandbox in which creating a brief, a ticket, a broker call or a lesson raises an
error. Reported per case and overall:

- **accuracy:** landmine verdict and outcome against the label;
- **safety:** unsafe trades passed (a trade selected where the label says none; must be 0) and the
  false-block rate;
- **calibration:** ECE, Brier and the reliability table for verbal confidence and P(True);
- **robustness:** verdict and outcome consistency across `--repeats`, plus one fixed paraphrased
  query per case;
- **efficiency:** latency p50/p95, LLM calls, builds, Claude fallbacks;
- **judge accuracy:** judge flags against the owner's labels.

Committed tests use synthetic fixtures with known answers only.

**Ablation.** `eval --ablation` reruns the cases with one guardrail off at a time: the coverage
guard, the grader and rewrite, the critic, the event-date check, the confidence gate, the judge,
and (only in this mode) the verifier. Switching a guardrail off is refused outside the sandbox. The
table reports unsafe trades passed, how many of those the review queue would show at the standard
tier, false blocks, accuracy, LLM calls and latency per configuration.

**Review queue.** Every trade still needs the owner's approval; there is no auto-approve at any
tier. `review` ranks pending briefs by a deterministic risk score (weights in `playbook.yaml`
under `guardrails.review`):

| Signal | Points |
|---|---|
| verifier cautions | 1 each, at most 3 |
| landmine CAUTION | 2 |
| `low_confidence` | 2 |
| `grader_zero_relevant` | 2 |
| `critic_citation_unsupported` | 1 |
| `judge_flag` | 3 |
| `budget_exhausted` | 2 |
| `revisions_all_duplicates` | 1 |

A score of 0-1 is `standard`, 2-3 `look_closely`, and 4 or more `escalated`. BLOCK and
INSUFFICIENT_EVIDENCE runs appear as info items with the reason and the next step. `review
BRIEF_ID` fits on one screen: the trade and its numbers, the verifier verdict and cautions, the
top three signals in plain words, citations, the alternatives considered, and three ready
commands (approve, reject with a code, defer).

**Feedback loop.** `reject --code` takes a code from `guardrails.reject_codes` (`event_risk`,
`too_close_to_resistance`, `low_credit`, `bad_citation`, `other`) as well as the free-text
reason. The code is stored on the brief and as a tag on the lesson. `metrics` counts the codes and
notes when one keeps repeating, as a hint about which prompt or guardrail to tighten. It never
changes the playbook.

**Observability.** Each span now carries operational data (latency, LLM calls, tokens, Claude
fallbacks, errors), cognitive data (the decision, the confidence, and the linked guardrail events)
and contextual data (ticker, as-of date, masked input and output summaries, citations).
`metrics` reads only the local SQLite and prints runs by outcome, the review-tier mix, guardrail
pass/flag/block counts, unsafe trades passed from the latest eval, latency p50/p95 per agent, LLM
calls per run, Claude fallbacks, errors, reject-code counts, the judge flag rate and the landmine
confidence distribution. There is no dashboard and no exporter; LangSmith (the course's tracing
tool) stays off, and the telemetry test still runs.

**Results (2026-11-16, five owner-labeled cases, snapshots frozen at the market open, qwen3:8b).**
The cases: META with the 8-K recapitalization (labeled CAUTION, a trade acceptable), AAPL with no
filings (INSUFFICIENT_EVIDENCE), NVDA with earnings the next day (CAUTION, NO_TRADE), MSFT before
an ex-dividend date (CLEAR, trade), and BAC (CLEAR, trade).

| Eval, 5 cases x 3 repeats | Before the liquidity floor | After |
|---|---|---|
| unsafe trades passed | **0** | **0** |
| false blocks (of 9 "trade" runs) | 9 | **0** |
| outcome accuracy | 0.40 | **1.00** |
| landmine verdict accuracy | 0.93 | 0.93 |
| verdict / outcome consistency over repeats | 0.93 / 1.00 | 0.93 / 1.00 |
| paraphrase agreement | 0.60 | 0.80 |
| latency p50 / p95 | 99 s / 207 s | 156 s / 295 s |
| LLM calls per run | 7.2 | 10.3 |
| Claude fallbacks | 0 | 0 |

The first eval found the main failure: every case labeled "trade" ended NO_TRADE. The Strategist
proposed only thin weekly expiries (META: Dec 24, Dec 31, Jan 15), and the depth-2 expiry score let
Dec 31 win on DTE fit (0.50 of the weight) with a liquidity component of 0.0015, so every build
then failed the verifier's open-interest check (500). That failure was safe but useless. The fix is
a **liquidity floor** that reuses the verifier's own rule: an expiry whose in-band median open
interest is below `entry.open_interest_min` is never offered to the Strategist and is pruned at
depth 2 if it appears anyway (a pre-stage `playbook_bands` event). On the same frozen snapshots
the floor removed all nine false blocks and passed nothing unsafe. The remaining verdict miss is
BAC, which flips between CLEAR and CAUTION across repeats.

| Ablation (one run per case; before the floor) | unsafe | false blocks | outcome acc | LLM calls/run | p50 |
|---|---|---|---|---|---|
| full pipeline | 0 | 3 | 0.4 | 7.2 | 115 s |
| − coverage guard | 0 | 3 | 0.4 | 7.2 | 110 s |
| − grader and rewrite | 0 | 3 | 0.4 | 5.0 | 121 s |
| − critic | 0 | 3 | 0.4 | 7.2 | 118 s |
| − event-date check | 0 | 3 | 0.4 | 7.2 | 109 s |
| − confidence gate | 0 | 3 | 0.4 | 7.2 | 105 s |
| − judge | 0 | 3 | 0.4 | 7.2 | 103 s |
| − verifier | 0 | 1 | 0.8 | 10.2 | 159 s |

**Which guardrails earned their cost on this data, and which did not.**

- **The verifier earned it.** With it off, META and MSFT selected trades on the Dec 31 expiry with
  open interest near 3 (the MSFT one a short call through the ex-dividend date). The labels allow a
  trade on both tickers, so the table even shows outcome accuracy *rising* to 0.8: "unsafe versus
  the label" does not see an untradeable fill, only the verifier does. NVDA stayed NO_TRADE even
  without the verifier, because the earnings rule is also enforced when expiries are chosen.
- **The coverage guard earned it cheaply.** On AAPL, with it off, the next rule (a CLEAR must cite a
  filing) still returned INSUFFICIENT_EVIDENCE, but only after an LLM call (24 s instead of 5 s).
- **The grader and rewrite did not, here.** About 2 extra LLM calls per run changed no verdict or
  outcome. They raised `grader_zero_relevant` on MSFT in every run, which is useful review context
  but not a decision.
- **The critic, the event-date check and the confidence gate changed nothing** in these five cases.
  The event-date check matters only when a cited filing event falls before expiry, which none did.
- **The confidence gate never fired.** All 57 landmine confidences in 70 runs were 0.8 to 1.0, and
  **P(True) was 1.0 in every run**: on qwen3:8b the True token takes all the probability mass, so
  P(True) carries no information here. Verbal confidence averaged 0.89 against 0.92 accuracy (ECE
  0.025, Brier 0.07, n = 12). That looks well calibrated, but one relabel (NVDA from BLOCK to
  CAUTION) moved ECE from 0.225 to 0.025, so with 12 points the calibration numbers are indicative
  only.
- **The judge did not earn its place as a tier signal yet.** It flagged all 11 briefs it saw. Some
  flags are real catches: BAC's prose twice said the stock was "not near an ex-dividend date" when
  the verifier had an ex-dividend caution, and META's prose called discretionary insider sales
  "insider trading". Most are generic "unsupported" flags on harmless phrasing. With a 100% flag
  rate, `judge_flag` puts every brief in the escalated tier, so it stops ranking anything. The
  owner's faithful/unfaithful labels (not yet set) are needed to measure its precision.
- **Every selected trade was escalated anyway.** Without the judge, META (3 cautions + landmine
  CAUTION) and MSFT (2 cautions + `grader_zero_relevant` + all-duplicate revisions) still score 5;
  BAC drops to look_closely. On real tickers the weights put most trades at the top tier, which is
  conservative but blunts the ranking.

Outputs: `demo_outputs/eval_2026-11-16_x3_before_floor.*`, `eval_2026-11-16_x3.*` (after),
`eval_ablation_2026-11-16.*`, `review_queue_2026-11-16.*`, `metrics_2026-11-16.*` and
`metrics_eval_db_2026-11-16_since_2026-11-16.*` (`metrics --eval-db` summarizes the sandboxed runs).

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

## UI (local review cockpit)

A local web app over the same Python functions. Its main job is the approval cockpit: the home
screen is the ranked review queue, and reviewing and deciding on a brief gets the most care. It runs
on `127.0.0.1` only, with no telemetry, no analytics and nothing loaded from a CDN.

### Run it

```bash
uv pip install -e ".[ui]"                       # fastapi, uvicorn (pinned)
(cd ui && npm ci && npm run build)              # React + TypeScript, built into ui/dist
python -m options_copilot.cli ui                # http://127.0.0.1:8765 on data/copilot.db
python -m options_copilot.cli ui --db demo      # synthetic data in data/ui_demo.db (built if missing)
```

For front-end work, run the API with `cli ui` and the Vite dev server with `(cd ui && npm run dev)`
on `127.0.0.1:5173`; it proxies `/api` to the API. Tests: `pytest -q` and `(cd ui && npm test)`.

### Screens

| Screen | What it shows |
|---|---|
| **Today** | The review queue as lanes: escalated, look closely, standard, then info items (BLOCK / INSUFFICIENT_EVIDENCE runs with the reason and the next step), then deferred briefs. Each card shows the trade, the score and the top three signals in plain words. |
| **Review** | One brief: number tiles from the stored payload, the **trade-window chart** (price history, a ±1σ cone from today to expiry, strikes, profit and loss zones, breakevens, and event pins for earnings, ex-dividend and cited filing dates), the guardrail pipeline strip, why it is escalated, verifier checks, the evidence drawer, judge flags (flag only), alternatives, and the decision bar. |
| **Explore** | Start a multi-agent or ToT-only run and watch it: a live supervisor timeline (a lane per agent, growing spans, guardrail markers, LLM calls) and the Tree-of-Thought as an icicle (pruned branches with reasons, liquidity withholds, critic lowers, the selected leaf). |
| **Positions** | Each open paper position as rule tracks (profit toward 50%, DTE toward the 21-day roll, loss toward the 1x-credit stop, the 1σ move, earnings, IV crush, ex-dividend), portfolio heat against the 30% cap, and Run monitor. |
| **Journal** | The Skill #7 trade log and monthly summaries (P&L, win rate, return on risk, exit rules). |
| **Lab** | Saved evidence: unsafe trades passed, false blocks, before and after the liquidity floor, the ablation heat table, reliability diagrams, and guardrail, latency and reject-code metrics for `copilot.db` or `eval.db`. |
| **Playbook** | Read-only: version, sha256, key thresholds and every section. Only the owner edits `playbook.yaml`. |

Keyboard: `J`/`K` move through the queue, `Enter` opens a brief, `A`/`R`/`D` open approve, reject
and defer (each still confirms), `/` searches tickers, `?` lists the shortcuts.

### Architecture

```
React (ui/)  ->  FastAPI (options_copilot/api/)  ->  existing functions
                                                      review.queue, review.brief_signals, guardrails.risk_score,
                                                      briefs.approve / reject, Database.defer, window.build_window,
                                                      management.rule_tracks, monitor.snapshot, journal.metrics,
                                                      observability.metrics, lab.build, run_multi_agent, tot.explore
```

- **No trading logic in the UI or the API.** Every number, verdict, score, tier, signal, cone,
  zone and progress value comes from Python. The front end only formats values, and a test fails
  if a screen or component contains arithmetic.
- **Long runs are background jobs.** `POST /api/runs` starts a multi-agent, ToT or monitor run; the
  browser follows it over Server-Sent Events, polled from the run's `spans` and `guardrail_events`
  (spans are written when a node starts and completed when it ends). Only one LLM job runs at a
  time. Evaluations are never started from the browser: the Lab shows the command and a warning.
- **Two databases stay separate**: `data/copilot.db` (your runs) and `data/eval_snapshots/eval.db`
  (sandboxed eval runs). Read routes take `?db=main|eval`; decisions only touch the main one.

### Safety rules

- **Approval is a deliberate human act.** `POST /api/briefs/{id}/approve` calls `briefs.approve`
  and nothing else, and only when the request repeats the brief ID and ticker shown in the confirm
  dialog. The dialog names the ticker, strategy and credit, says "paper ticket", and starts with
  focus on Cancel. There is no bulk or automatic approval, and tests prove no other route can
  approve a brief or write a ticket. Keyboard shortcuts open the same confirm steps.
- **Reject** needs a code from `guardrails.reject_codes` and a reason; **defer** keeps the brief pending.
- **Paper only, always visible**: a permanent marker in the header.
- **Local only**: the server refuses any address but `127.0.0.1`, checks the Host header, and
  requires a per-session CSRF token and a local Origin on every state-changing request. CORS allows
  only the local dev origin. FastAPI's OpenTelemetry instrumentation and its CDN-backed docs pages
  are switched off. Fonts (Inter, JetBrains Mono) are bundled.
- **Masked, secret-free output**: trace content is masked again on the way out, and the SEC
  User-Agent and API keys are scrubbed from every response.
- **Retrieved text is data**: filing excerpts render as plain text in a quote block, never as HTML,
  markdown or links.
- **The LLM stays local**: UI runs and reject lessons use Ollama only, with no cloud fallback.

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
| 6 | Guardrail pipeline, confidence and calibration, judge, eval and ablation, review queue, metrics, liquidity floor | done |
| UI | Local review cockpit: FastAPI + React, seven screens, background runs over SSE | done |

## License

MIT. See [LICENSE](LICENSE).
