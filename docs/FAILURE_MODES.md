# Failure modes this copilot has actually shown

Every row below happened in a real run or is pinned by a test. Each one names the guardrail
that catches it (see `options_copilot/guardrails.py` for the registry and the pipeline order).
The rows are grouped by where the hallucination comes from, using three sources: the input
or query, the retrieved context, and the generation step.

The LLM is the local `qwen3:8b` through Ollama in every run listed here. Run IDs refer to the
local SQLite database (`trace RUN_ID`). Filing text is not reproduced.

## 1. Input / query

| Failure | Evidence | Caught by |
|---|---|---|
| **Retrieval miss behind a CLEAR.** The default landmine query and the one rewrite both found nothing the grader judged relevant. Coverage still passed, so the verdict was CLEAR on weak evidence. | MSFT multi-agent run, 2026-11-09 (`run-20261109-4edbd9`). The grader found 0 relevant passages on both attempts. | **New:** the `grader_rewrite` guardrail raises the `grader_zero_relevant` signal (look_closely tier). The verdict is unchanged, because a signal adds caution and never relaxes it. Test: `test_grader_zero_relevant_is_a_signal_not_a_verdict_change`. |
| **No data to judge.** No AAPL filings were indexed in the 90-day window. | `check AAPL` on 2026-10-30 and 2026-10-31 (Checkpoint 3.1). | The `coverage_guard` returns INSUFFICIENT_EVIDENCE before any LLM call, and the run escalates to the owner. Test: `test_coverage_guard_blocks_before_any_llm_call`. |
| **Only thin expiries explored (false blocks).** The Strategist proposed weekly expiries with almost no open interest, and the expiry score let Dec 31 win on DTE fit alone (liquidity component 0.0015). Every build then failed the verifier's open-interest check. | First eval, 2026-11-16: all 9 runs of the three "trade" cases ended NO_TRADE (META `run-20261116-66aff2`). | **New:** a liquidity floor withholds and prunes any expiry whose in-band median open interest is below `entry.open_interest_min` (500). In the re-run on the same snapshots, false blocks fell from 9 to 0, with 0 unsafe passes. Tests: `test_thin_expiry_is_withheld_and_a_liquid_one_is_chosen` and `test_depth2_prunes_a_thin_expiry_even_if_offered`. |
| **Out-of-playbook proposals.** The generator suggested a strategy, expiry or delta that the playbook does not allow. | ToT generator rejections recorded in `rejected_candidates`. Test: `test_out_of_playbook_candidates_are_rejected`. | `playbook_bands` rejects the candidate before it is built. |

## 2. Retrieved context

| Failure | Evidence | Caught by |
|---|---|---|
| **Prompt injection through filing or lesson text.** Retrieved text may contain instructions or fake delimiters. | Synthetic fixtures contain `IGNORE RULES <<<x>>>`. Tests: `test_injected_delimiters_are_neutralized` and `test_schema_has_no_approval_tool`. | Retrieved text is wrapped as untrusted data and its delimiters are neutralized. The landmine LLM is offered no tools, and no agent has an approve tool. |
| **Invented citations.** The model cited chunk IDs it was never shown. | Test: `test_with_rag_cites_real_metadata_and_offers_no_tools` (C7 is dropped). | `citation_from_metadata` rebuilds citations from stored chunk metadata, so invented IDs never reach a brief. |
| **Same filing, different verdict.** The META 8-K recapitalization (Item 1.01, filed 2026-10-21) was BLOCK on 2026-10-31. It was CAUTION on 2026-11-09 and 2026-11-10, after newer 10-Q chunks changed what was retrieved. | `compare META` on 2026-10-31; `ma-compare META` on 2026-11-09; `ma META` on 2026-11-10. | Not prevented. It is **measured** in `eval` as verdict consistency and paraphrase agreement. A CAUTION still adds the `landmine_caution` signal. |

## 3. Generation

| Failure | Evidence | Caught by |
|---|---|---|
| **Wrong date reasoning.** The LLM said the META 8-K's effective date (2027-01-19) fell inside the trade window of a 45-day spread that expires in mid-December. | Checkpoint 3.1 hand check of `check META`, 2026-10-30. | `event_date_check`: Python reads the cited event dates and compares them with the expiry (`verifier.check_event_window`). The LLM's date reasoning is never used. Tests: `test_event_inside_trade_window_blocks` and `test_cited_event_after_expiry_is_only_caution`. |
| **Numbers invented in prose.** | Every brief since Phase 3: numbers are filled in by code. Test: `test_brief_numbers_equal_the_tool_numbers`. | `brief_numbers`. Sentences with digits are dropped from the LLM's prose, and the brief's numbers must equal the verified tool numbers, or no brief is written. |
| **Overconfident or vague critique.** On META the critic applied one generic reason, citing `lesson:3`, to all seven leaves, lowering every one by ×0.7. In Phase 5 it cited `check:event_window`, which had passed. | `run-20261109-ab0805` (the ToT run inside the META multi-agent run) and the Phase 5 README findings. | `critic_lower_only`: a critique can only lower a score. **New:** `critic_citation` flags a cited check that passed, a lesson that does not exist, or one reason pasted on three or more candidates, as the `critic_citation_unsupported` signal. Test: `test_critic_citation_problems`. |
| **Revisions that repeat.** All nine revision proposals the critic triggered on META were repeats. | `run-20261109-ab0805`, nodes n25 to n33. | Repeats are pruned without a build. **New:** the `revisions_all_duplicates` signal. |
| **Schema echo instead of an answer.** The CrewAI Strategist returned the JSON schema itself on all 8 calls. | First live MSFT multi-agent run, 2026-11-08 (`run-20261109-82a06c`). | The deterministic enumerator fallback. The task prompt now shows an example object instead of the schema. |
| **Budget exhaustion and looping.** The ReAct single agent used its 14-call budget and chose nothing. | META (`run-20261109-ba0a9b`) and NVDA 2026-10-31 (`tot-compare NVDA`). | `budgets`: the run ends with NO_TRADE or `budget_exhausted`, and the `budget_exhausted` signal is raised. Tests: `test_budgets_end_the_graph` and `test_step_budget`. |
| **Unfaithful summary prose.** A claim the facts do not support or contradict. Seen live in the eval: BAC's prose twice said "not near an ex-dividend date" when the verifier had an ex-dividend caution. META's prose called discretionary insider sales "insider trading". | Eval 2026-11-16 (sandboxed runs); synthetic test `test_judge_flags_an_unfaithful_synthetic_brief`. | **New:** `judge` (LLM-as-judge) raises `judge_flag`. It can only flag. The number check above stays the hard check. |
| **Uninformative confidence.** P(True) was 1.0 in all 57 landmine checks, and verbal confidence was always 0.8 or above, including on a wrong verdict. | Eval and ablation, 2026-11-16 (70 sandboxed runs). | Not caught: the confidence gate (0.6) never fired. This is a measured limit of the signal on qwen3:8b. |
| **Unparseable confidence.** The P(True) reply was not True or False, or the stated confidence was missing. | Test: `test_malformed_p_true_escalates`. | **New:** the `confidence_gate` treats a missing or malformed confidence as `low_confidence`. Missing evidence is never treated as safe. |

## Not seen yet, but guarded

- **A Claude fallback answering instead of the local model.** This is counted per run (`metrics`: Claude fallbacks). The count was 0 in every run so far.
- **A telemetry exporter turning on.** Tests: `test_telemetry_and_tracing_flags_are_set` and `test_no_openai_or_default_llm_fallback`.
