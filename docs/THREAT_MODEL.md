# Threat model: the copilot on the MAESTRO seven layers

MAESTRO stands for Multi-Agent Environment, Security, Threat, Risk and Outcome. It is the
agentic-AI threat-modeling framework published by the Cloud Security Alliance (Ken Huang,
6 February 2025):
<https://cloudsecurityalliance.org/blog/2025/02/06/agentic-ai-threat-modeling-framework-maestro>.
The layer names below follow that post. Layer 6, Security and Compliance, is a vertical layer
that cuts across the others.

**Scope.** This is an educational, paper-only copilot that runs on one local machine. It has no
brokerage connection, it places no orders, and the owner approves every trade. All data is public
(SEC EDGAR, yfinance) or synthetic.

| # | Layer | Here |
|---|---|---|
| 1 | Foundation Models | qwen3:8b on local Ollama; Claude API fallback only if Ollama fails and a key is set |
| 2 | Data Operations | SEC EDGAR filings and yfinance quotes and chains; Chroma index; SQLite; evaluation snapshots |
| 3 | Agent Frameworks | LangGraph supervisor; CrewAI Strategist and Risk Critic; ReAct and Tree-of-Thought planners |
| 4 | Deployment and Infrastructure | one laptop; Ollama on localhost; the stdio MCP server; a Python venv |
| 5 | Evaluation and Observability | eval harness, ablation, spans, guardrail events, `metrics` |
| 6 | Security and Compliance (vertical) | mask layer, secrets in `.env`, public-repo hygiene, privacy audit |
| 7 | Agent Ecosystem | the owner (approver), the PaperBroker, the journal |

## Layer 1: Foundation Models

| Threat | Guardrail | Residual risk |
|---|---|---|
| Hallucinated facts, dates or numbers | The LLM does no arithmetic. Numbers come from the calculators. `event_date_check` reads dates in Python. `brief_numbers` is a hard check. | The prose can still mislead without using any number. `judge` flags this, but the judge is the same model. |
| Overconfidence | Verbal confidence plus P(True) from token logprobs. The `confidence_gate` raises `low_confidence`. Calibration (ECE and Brier) is measured in `eval`. | Calibration rests on a small labeled set and may not hold under distribution shift. |
| Model-generated instructions slipping past the deterministic rules | The verifier is final. The critic is lower-only. Signals can only add caution. | None identified for numeric rules. Judgment calls such as how "material" a filing is remain the model's. |
| Fallback to a cloud model | It is used only if Ollama fails and a key is set. Fallbacks are counted per run. CrewAI's default OpenAI route points at a closed local port. | When the fallback is used, the prompt (masked) goes to the Claude API. |
| Self-preference in the judge | The judge's accuracy is measured against the owner's labels. It can only flag. | The judge may approve its own model's phrasing. Its recall is reported, not assumed. |

## Layer 2: Data Operations

| Threat | Guardrail | Residual risk |
|---|---|---|
| Prompt injection in filing, news or lesson text | Retrieved text is wrapped as untrusted data and its delimiters are neutralized. The landmine LLM gets no tools. No agent has a write or approve tool. | A filing could still bias the model's verdict through its content. The verifier and the owner remain the backstop. |
| Stale or missing data (weekend chains, unindexed filings) | `coverage_guard` returns INSUFFICIENT_EVIDENCE. Builds with no quotes are pruned. Missing data is never treated as safe. | A stale but present quote can pass. Live `ma-compare` and the eval snapshots are taken during market hours. |
| Retrieval misses a material event | The grader with one rewrite, the `grader_zero_relevant` signal, and paraphrase agreement measured in `eval`. | An event outside the 90-day window, or one phrased unlike the query, can be missed. |
| Citation fabrication | `citation_from_metadata` rebuilds every citation from stored chunk metadata. | None for IDs. The model can still misread a real excerpt. |
| Snapshot tampering | Snapshots are local and gitignored. Their pickle files are written and read only by this tool. | Anyone with write access to `data/` can alter them, as with the rest of the local data. |
| SEC fair-access misuse | The User-Agent comes from `.env`. Requests stay under 10 per second and are cached. | Not applicable beyond rate limits. |

## Layer 3: Agent Frameworks

| Threat | Guardrail | Residual risk |
|---|---|---|
| An agent choosing its own path, looping, or skipping a check | The supervisor's routing is a pure function of state. The only edge into `brief_writer` comes from `verifier_gate` (tested). Budgets cover calls, builds, rounds and wall-clock time. | A framework bug could change behavior. Versions are pinned and the routing tests pin the graph. |
| A critic raising or rescuing a candidate | `apply_critique` allows only keep, lower or veto. `critic_citation` flags unsupported reasons. | The critic can over-penalize: a false-block risk that `eval` measures. |
| Schema echo or malformed agent output | The deterministic enumerator fallback. Malformed replies are ignored and noted. | The fallback is less diverse. On META, the CrewAI Strategist proposed a delta (0.22) that the enumerator never tries. |
| Framework telemetry leaking prompts | CrewAI, OpenTelemetry and LangSmith telemetry are forced off and tested. | A future framework release could add a new channel. Versions are pinned. |

## Layer 4: Deployment and Infrastructure

| Threat | Guardrail | Residual risk |
|---|---|---|
| A network listener exposing tools | The MCP server runs over stdio only (tested). Runs use an in-memory session. There are no auth tokens. | A local process that can start `mcp-serve` can call the read and verify tools. None of them can approve or trade. |
| Ollama reachable from the network | Ollama is used at 127.0.0.1. | Its bind address is the owner's machine setting, outside this repo. |
| Supply chain (pip packages) | Installs are pinned. The owner runs every install. | Pinned packages can still carry flaws. They are not vendored or audited. |

## Layer 5: Evaluation and Observability

| Threat | Guardrail | Residual risk |
|---|---|---|
| A guardrail that silently stops working | Every decision is a `guardrail_events` row linked to its span. `trace` shows them inline. A test checks that every registered guardrail emits an event. | An event shows that a check ran, not that its threshold is right. |
| An eval run causing side effects | The sandbox makes briefs, tickets, broker calls and lesson writes raise an error (tested). Eval uses its own database. | None identified. |
| Ablation leaking into normal runs | Switching guardrails off is refused outside the sandbox. The verifier can be turned off only there (tested). | None identified. |
| Overclaiming from a small sample | Counts are printed next to every calibration bin. Unsafe passes are reported raw. | A handful of labeled cases cannot show a rare failure. |
| Traces holding personal data | Everything logged passes through the mask layer. Data stays local. No exporter is configured. | The masker is pattern-based and could miss a novel format. The privacy audit runs before every commit. |

## Layer 6: Security and Compliance (vertical)

| Threat | Guardrail | Residual risk |
|---|---|---|
| Secrets or personal data in the public repo | `.env`, `data/`, `demo_outputs/`, `*.local.yaml` and `eval_snapshots/` are gitignored. The pre-commit privacy audit checks home paths, keys, emails, denylisted names, and these paths. | The audit is pattern-based, and a human still reviews each diff. |
| The SEC User-Agent reaching logs or outputs | It is read from `.env` and kept out of traces and demo outputs (checked before each push). | Manual check, so it depends on the owner. |
| Real account data | No broker connection. The portfolio value is synthetic, and every position is synthetic. | Not applicable. |
| Advice or compliance risk | Educational, paper-only use. Every entry and roll is approved by the owner. | The owner may act on a brief elsewhere. That is outside the system. |

## Layer 7: Agent Ecosystem

| Threat | Guardrail | Residual risk |
|---|---|---|
| An agent approving or placing a trade | No approve tool exists for agents. `approve` is a human CLI command. Approval writes a paper ticket, and the PaperBroker only simulates fills. No module can auto-approve at any tier (tested). | The owner can approve a bad brief. The review queue ranks risky briefs first, with reasons. |
| Reviewer overload or rubber-stamping | Deterministic tiers (standard, look_closely, escalated). A one-screen view with the top three reasons and three choices. Reject codes feed `metrics`. | Approving every trade is by design a universal gate. The tiers set the order, not whether a trade needs approval. |
| A feedback loop drifting the rules | Reject codes and lessons are reported only. The playbook is read-only to the agent, and only the owner edits thresholds. | A misleading lesson can bias later prompts. Lessons are wrapped as data and the verifier stays final. |
