"""Phase 6: guardrail pipeline, confidence and calibration, judge, eval harness and ablation sandbox,
review queue, metrics. Synthetic data and fake LLMs only; no network, no Ollama."""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pytest
import yaml

from agent_fakes import AS_OF, TICKER, FakeSources
from conftest import BagOfWordsEmbedding
from options_copilot import evaluation as E
from options_copilot import guardrails as G
from options_copilot.agent import landmine_check, p_true_from_reply
from options_copilot.chunker import Chunk
from options_copilot.db import Database
from options_copilot.judge import judge_brief
from options_copilot.llm import LLMResponse
from options_copilot.mask import Masker
from options_copilot.models import Reflection, ReflectionTrigger
from test_multi_agent import PV, DenseSources, EvidenceSources, RoleLLM, run, veto_first

LM_AS_OF = date(2026, 10, 30)


# --- fakes -------------------------------------------------------------------------------------

class VerdictLLM:
    """Landmine fake: a verdict with a stated confidence, then a P(True) answer (optionally with logprobs)."""
    name = "verdict-fake"

    def __init__(self, verdict="CLEAR", confidence=0.9, p_true_reply="True", logprobs=None, cite="C1"):
        self.verdict, self.confidence, self.p_true_reply, self.logprobs, self.cite = \
            verdict, confidence, p_true_reply, logprobs, cite
        self.supports_logprobs = logprobs is not None
        self.calls = []

    def chat(self, messages, *, json_schema=None, tools=None, top_logprobs=None, max_tokens=None):
        self.calls.append(messages[-1]["content"][:60])
        if "Is the proposed verdict" in messages[-1]["content"]:
            return LLMResponse(self.p_true_reply, self.name, [], 1, self.logprobs)
        return LLMResponse(json.dumps({"verdict": self.verdict, "reasons": ["synthetic"], "cited_ids": [self.cite],
                                       "confidence": self.confidence}), self.name)


def filing(cid, text, item="Item 1.01", filed=20261015, ticker="TICKER_A"):
    return Chunk(chunk_id=cid, text=text, ticker=ticker, form="8-K", filed_date=filed, item=item,
                 source_url=f"https://example.com/{cid}", chunk_idx=0, accession=f"acc-{cid}", doc_type="8-K")


@pytest.fixture
def loaded(store):
    text = "merger agreement acquisition material definitive agreement tender offer results of operations"
    store.add_chunks([filing("f1", text), filing("f2", text + " shareholder approval closing", "Item 8.01")])
    return store


class ConfidenceSources(EvidenceSources):
    """Evidence with a configurable landmine confidence (what the MCP view passes on)."""

    def __init__(self, conf=0.9, p_true=0.9, **kw):
        super().__init__(**kw)
        self.conf, self.pt = conf, p_true

    def landmine(self, ticker, context, query=None):
        out = super().landmine(ticker, context, query)
        out.update(confidence=self.conf, p_true=self.pt, p_true_method="logprobs", llm_verdict=out["verdict"])
        return out


class JudgeLLM(RoleLLM):
    """RoleLLM that also answers the judge: flags any summary claiming there are no risks."""

    def chat(self, messages, *, json_schema=None, tools=None):
        text = " ".join(str(m.get("content", "")) for m in messages)
        if "check a short trade summary for faithfulness" in text:
            self.calls.append("judge")
            claims = []
            if "no risks" in json.loads(messages[-1]["content"])["summary"].lower():
                claims.append({"claim": "there are no risks", "issue": "contradicted",
                               "conflicts_with": "verifier cautions"})
            return LLMResponse(json.dumps({"claims": claims}), self.name, [], 1)
        return super().chat(messages, json_schema=json_schema, tools=tools)


# --- guardrail registry and pipeline -----------------------------------------------------------

def test_every_registered_guardrail_emits_an_event(pb):
    r, db = run(pb, DenseSources(), llm=RoleLLM(critic=veto_first))
    emitted = {e["guardrail"] for e in db.guardrail_events(r.run_id)}
    assert emitted == {g.name for g in G.REGISTRY}
    assert all(e["span_id"] for e in db.guardrail_events(r.run_id))   # every event is linked to a span
    span_ids = {s["span_id"] for s in db.spans(r.run_id)}
    assert {e["span_id"] for e in db.guardrail_events(r.run_id)} <= span_ids


def test_unregistered_guardrail_or_signal_is_refused():
    log = G.GuardrailLog(None, "run-x")
    with pytest.raises(KeyError):
        log.emit("made_up_guardrail", "pass", "x")
    with pytest.raises(ValueError):
        log.emit("verifier", "approve", "x")
    with pytest.raises(ValueError):
        log.emit("verifier", "flag", "x", signal="auto_approve")


def test_registry_stages_and_kinds():
    assert [g.stage for g in G.REGISTRY] == sorted((g.stage for g in G.REGISTRY), key=G.STAGES.index)
    assert all(g.kind in G.KINDS for g in G.REGISTRY)
    pre = {g.name for g in G.REGISTRY if g.stage == "pre"}
    assert pre == {"coverage_guard", "grader_rewrite", "playbook_bands"}
    assert {"verifier", "brief_numbers", "judge"} <= {g.name for g in G.REGISTRY if g.stage == "post"}


@pytest.mark.parametrize("sources, llm", [
    (lambda: DenseSources(), lambda: RoleLLM(critic=veto_first)),
    (lambda: FakeSources(), lambda: None),
    (lambda: EvidenceSources(first="BLOCK"), lambda: RoleLLM()),
    (lambda: EvidenceSources(first="INSUFFICIENT_EVIDENCE", cov1=False, cov2=False), lambda: RoleLLM()),
])
def test_pipeline_runs_pre_then_during_then_post(pb, sources, llm):
    r, db = run(pb, sources(), llm=llm())
    events = db.guardrail_events(r.run_id)
    assert events and G.stage_order_ok(events)
    assert not G.stage_order_ok(list(reversed(events))) or len({e["stage"] for e in events}) == 1


def test_coverage_guard_blocks_before_any_llm_call(pb):
    llm = RoleLLM()
    r, db = run(pb, EvidenceSources(first="INSUFFICIENT_EVIDENCE", second="INSUFFICIENT_EVIDENCE",
                                    cov1=False, cov2=False), llm=llm)
    first = db.guardrail_events(r.run_id)[0]
    assert first["guardrail"] == "coverage_guard" and first["decision"] == "block"
    assert r.outcome == "INSUFFICIENT_EVIDENCE" and "generator" not in llm.calls


def test_grader_zero_relevant_is_a_signal_not_a_verdict_change(pb):
    r, db = run(pb, EvidenceSources(first="CLEAR", second="CLEAR"), llm=RoleLLM(relevant=0))
    assert r.state["research"]["verdict"] == "CLEAR"
    ev = [e for e in db.guardrail_events(r.run_id) if e["guardrail"] == "grader_rewrite"]
    assert ev[0]["signal"] == "grader_zero_relevant" and r.state["signals"]["grader_zero_relevant"] == 1


def test_trace_shows_guardrail_events_inline(pb):
    from options_copilot.cli import _render_spans
    r, db = run(pb, DenseSources(), llm=RoleLLM(critic=veto_first))
    text = _render_spans(db.spans(r.run_id), db.guardrail_events(r.run_id))
    assert "guardrail pre    coverage_guard" in text and "guardrail post   verifier" in text


# --- confidence --------------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", ["BLOCK", "CAUTION", "CLEAR"])
def test_low_confidence_never_relaxes_the_verdict(loaded, verdict):
    hi = landmine_check("TICKER_A", "put credit spread", store=loaded, llm=VerdictLLM(verdict, 0.95), as_of=LM_AS_OF)
    lo = landmine_check("TICKER_A", "put credit spread", store=loaded, llm=VerdictLLM(verdict, 0.05, "False"),
                        as_of=LM_AS_OF)
    assert hi["verdict"] == lo["verdict"] == verdict
    assert lo["confidence"] == 0.05 and lo["p_true"] == 0.0 and lo["p_true_method"] == "answer_only"


def test_confidence_gate_only_adds_a_signal(pb):
    hi, db_hi = run(pb, ConfidenceSources(conf=0.95, p_true=0.9, first="CAUTION"), llm=RoleLLM())
    lo, db_lo = run(pb, ConfidenceSources(conf=0.2, p_true=0.1, first="CAUTION"), llm=RoleLLM())
    assert hi.state["research"]["verdict"] == lo.state["research"]["verdict"] == "CAUTION"
    assert lo.state["signals"].get("low_confidence") and not hi.state["signals"].get("low_confidence")
    assert hi.outcome == lo.outcome
    s_hi = sorted((c["node"], c["score"]) for c in hi.state["candidates"])
    s_lo = sorted((c["node"], c["score"]) for c in lo.state["candidates"])
    assert s_hi == s_lo   # no score was raised (or changed) by low confidence


def test_low_confidence_critic_action_still_lowers(pb):
    def low_conf_lower(ids):
        return [{"node_id": ids[0], "action": "lower_major", "reason": "near resistance", "cite": "lesson:1",
                 "confidence": 0.1}] + [{"node_id": i, "action": "keep", "reason": "", "cite": "", "confidence": 0.9}
                                        for i in ids[1:]]
    base, _ = run(pb, DenseSources(), llm=RoleLLM())
    r, db = run(pb, DenseSources(), llm=RoleLLM(critic=low_conf_lower))
    base_scores = {c["node"]: c["score"] for c in base.state["candidates"]}
    for c in r.state["candidates"]:
        if c["node"] in base_scores and c["score"] is not None:
            assert c["score"] <= base_scores[c["node"]]
    gate = [e for e in db.guardrail_events(r.run_id) if e["guardrail"] == "confidence_gate" and "critic" in e["reason"]]
    assert gate and gate[0]["signal"] == "low_confidence"


def test_p_true_from_logprobs():
    lp = [{"token": "True", "logprob": math.log(0.7),
           "top_logprobs": [{"token": "True", "logprob": math.log(0.7)}, {"token": "False", "logprob": math.log(0.2)},
                            {"token": " True", "logprob": math.log(0.05)}]}]
    p, method = p_true_from_reply("True", lp)
    assert method == "logprobs" and p == pytest.approx(0.75 / 0.95, abs=1e-4)
    assert p_true_from_reply("False", None) == (0.0, "answer_only")
    assert p_true_from_reply("maybe", None) == (None, "answer_only")
    assert p_true_from_reply("x", [{"token": "x", "logprob": 0, "top_logprobs": [{"token": "x", "logprob": 0}]}])[0] is None


def test_p_true_logprobs_end_to_end(loaded):
    lp = [{"token": "True", "logprob": math.log(0.6),
           "top_logprobs": [{"token": "True", "logprob": math.log(0.6)}, {"token": "False", "logprob": math.log(0.4)}]}]
    out = landmine_check("TICKER_A", "put credit spread", store=loaded, llm=VerdictLLM("CLEAR", 0.8, "True", lp),
                         as_of=LM_AS_OF)
    assert out["p_true"] == pytest.approx(0.6) and out["p_true_method"] == "logprobs"


def test_malformed_p_true_escalates(loaded, pb):
    out = landmine_check("TICKER_A", "put credit spread", store=loaded, llm=VerdictLLM("CLEAR", 0.9, "I think so"),
                         as_of=LM_AS_OF)
    assert out["p_true"] is None and any("P(True)" in n for n in out["guard_notes"])
    assert out["verdict"] == "CLEAR"   # the verdict rules are unchanged; the gate escalates instead
    decision, _ = G.confidence_gate(out["p_true"], pb["guardrails.min_confidence"])
    assert decision == "flag"
    r, _ = run(pb, ConfidenceSources(conf=0.9, p_true=None), llm=RoleLLM())
    assert r.state["signals"].get("low_confidence") == 1


@pytest.mark.parametrize("value, decision", [(0.9, "pass"), (0.6, "pass"), (0.59, "flag"), (None, "flag"),
                                             ("high", "flag"), (1.7, "flag"), (-0.1, "flag")])
def test_confidence_gate_values(value, decision):
    assert G.confidence_gate(value, 0.6)[0] == decision


# --- critic citation check -----------------------------------------------------------------------

def test_critic_citation_problems():
    verdicts = {"n1": {"event_window": "CLEAR", "ex_dividend": "CAUTION"}, "n2": {"event_window": "CLEAR"},
                "n3": {"event_window": "CLEAR"}}
    ok = [{"node_id": "n1", "action": "lower_minor", "cite": "check:ex_dividend", "reason": "ex-div"}]
    assert G.critic_citation_problems(ok, verdicts, 2) == []
    bad = [{"node_id": "n1", "action": "lower_major", "cite": "check:event_window", "reason": "event"},
           {"node_id": "n2", "action": "veto", "cite": "lesson:3", "reason": "lesson"}]
    probs = G.critic_citation_problems(bad, verdicts, 2)
    assert any("passed" in p for p in probs) and any("lesson:3" in p for p in probs)
    generic = [{"node_id": n, "action": "lower_major", "cite": "lesson:1", "reason": "Landmine CAUTION is significant"}
               for n in ("n1", "n2", "n3")]
    assert any("same reason" in p for p in G.critic_citation_problems(generic, verdicts, 3))
    keeps = [{"node_id": "n1", "action": "keep", "cite": "check:event_window", "reason": ""}]
    assert G.critic_citation_problems(keeps, verdicts, 0) == []


# --- judge ---------------------------------------------------------------------------------------

def test_judge_flags_an_unfaithful_synthetic_brief():
    llm = JudgeLLM()
    facts = {"verifier_verdict": "CAUTION", "cautions": ["ex_dividend: ex-dividend inside the trade window"]}
    out = judge_brief("An iron condor fits. There are no risks to watch.", facts, llm)
    assert out["status"] == "ok" and out["flags"] and out["flags"][0]["issue"] == "contradicted"
    clean = judge_brief("An iron condor fits; the ex-dividend date is the main caution.", facts, llm)
    assert clean["flags"] == []


def test_judge_can_only_add_a_flag(pb):
    faithful, db1 = run(pb, DenseSources(), llm=JudgeLLM(prose="An iron condor fits a range-bound stock."))
    unfaithful, db2 = run(pb, DenseSources(), llm=JudgeLLM(prose="An iron condor fits. There are no risks here."))
    assert faithful.outcome == unfaithful.outcome == "SELECTED"
    assert not faithful.state["signals"].get("judge_flag") and unfaithful.state["signals"]["judge_flag"] == 1
    assert faithful.state["brief_id"] and unfaithful.state["brief_id"]   # a flag never removes the human decision
    b = db2.brief(unfaithful.state["brief_id"])
    assert b["status"] == "pending" and b["payload"]["judge"]["flags"]


def test_malformed_or_missing_judge_passes_nothing():
    class Bad:
        name = "bad"

        def chat(self, messages, **kw):
            return LLMResponse("not json", "bad")
    assert judge_brief("text", {}, Bad())["status"] == "malformed"
    assert judge_brief("text", {}, None)["status"] == "unavailable"


# --- evaluation metrics on known answers ----------------------------------------------------------

LABELS = {"a": {"expected_verdict": "CLEAR", "expected_outcome": "trade", "acceptable_strategies": ["iron_condor"]},
          "b": {"expected_verdict": "BLOCK", "expected_outcome": "BLOCK"},
          "c": {"expected_verdict": "CLEAR", "expected_outcome": "NO_TRADE"}}


def rec(case, verdict, outcome, strategy=None, conf=None, p_true=None, ms=1000, para=None, tier=None):
    return {"case": case, "verdict": verdict, "outcome": outcome, "strategy": strategy, "confidence": conf,
            "p_true": p_true, "latency_ms": ms, "llm_calls": 10, "builds": 5, "fallbacks": 0,
            "paraphrase_verdict": para, "tier": tier}


def test_eval_metrics_on_known_answers():
    records = [
        rec("a", "CLEAR", "SELECTED", "iron_condor", 0.9, 0.8, 1000, "CLEAR"),
        rec("a", "CLEAR", "SELECTED", "put_credit_spread", 0.9, 0.8, 2000),   # outcome wrong (strategy)
        rec("a", "CAUTION", "NO_TRADE", None, 0.5, 0.4, 3000),                  # false block, verdict wrong
        rec("b", "BLOCK", "BLOCK", None, 0.8, 0.9, 4000, "CAUTION"),
        rec("c", "CLEAR", "SELECTED", "iron_condor", 0.7, 0.6, 5000, "CLEAR"),  # unsafe pass
    ]
    s = E.summarize(records, LABELS)
    a = s["aggregate"]
    assert a["runs"] == 5 and a["cases"] == 3
    assert a["verdict_accuracy"] == 0.8 and a["outcome_accuracy"] == 0.4
    assert a["unsafe_passed"] == 1 and a["false_blocks"] == 1 and a["false_block_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["cases"]["a"]["verdict_consistency"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["cases"]["a"]["outcome_consistency"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["cases"]["b"]["paraphrase_agreement"] == 0.0 and s["cases"]["a"]["paraphrase_agreement"] == 1.0
    assert a["latency_p50_s"] == 3.0 and a["latency_p95_s"] == 5.0
    correct = [True, True, False, True, True]
    conf = [0.9, 0.9, 0.5, 0.8, 0.7]
    brier = sum((c - k) ** 2 for c, k in zip(conf, correct)) / 5
    assert s["calibration"]["confidence"]["brier"] == pytest.approx(brier, abs=1e-4)
    # bins of width 0.2: [0.4-0.6): 0.5 wrong -> |0.5-0|; [0.6-0.8): 0.7 right -> |0.7-1|;
    # [0.8-1.0]: 0.9, 0.9, 0.8 all right -> |0.8667-1|
    ece = (1 / 5) * 0.5 + (1 / 5) * 0.3 + (3 / 5) * abs((0.9 + 0.9 + 0.8) / 3 - 1)
    assert s["calibration"]["confidence"]["ece"] == pytest.approx(ece, abs=1e-4)
    rel = {r["bin"]: r["n"] for r in s["calibration"]["confidence"]["reliability"]}
    assert rel == {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 1, "0.6-0.8": 1, "0.8-1.0": 3}


def test_ece_and_brier_edges():
    assert E.ece([1.0, 1.0], [True, True]) == 0.0 and E.brier([1.0, 0.0], [True, False]) == 0.0
    assert E.ece([None], [True]) is None and E.brier([], []) is None
    assert E.consistency(["A", "A", "B", "A"]) == 0.75 and E.consistency([]) == 1.0
    assert E.percentile([5, 1, 3], 50) == 3 and E.percentile([], 50) is None


def test_judge_accuracy():
    res = [{"faithful": False, "flagged": True}, {"faithful": True, "flagged": False},
           {"faithful": True, "flagged": True}, {"faithful": False, "flagged": False}]
    j = E.judge_accuracy(res)
    assert (j["tp"], j["fp"], j["fn"], j["tn"]) == (1, 1, 1, 1) and j["accuracy"] == 0.5 and j["precision"] == 0.5


# --- sandbox and ablation ------------------------------------------------------------------------

class RecordingStore:
    def __init__(self):
        self.added = []

    def add_reflection(self, r):
        G.forbid_in_sandbox("writing a lesson")
        self.added.append(r)


@pytest.mark.parametrize("disabled", [frozenset(), frozenset({"verifier"}), frozenset({"judge", "critic"})])
def test_ablation_sandbox_cannot_brief_trade_or_learn(pb, disabled):
    store = RecordingStore()
    for sources in (DenseSources(), FakeSources(insider=__import__("options_copilot.verifier",
                                                                   fromlist=["InsiderActivity"]).InsiderActivity(
                                                                       100, 10_000, 5))):
        r, db = run(pb, sources, llm=RoleLLM(critic=veto_first), store=store, sandbox=True, disabled=disabled)
        assert r.state["brief_id"] is None and db.briefs() == [] and db.tickets() == []
        assert r.state["lesson"] is None
    assert store.added == []


def test_sandbox_forbids_every_side_effect(pb):
    from options_copilot.briefs import make_brief
    from options_copilot.broker import PaperBroker
    from options_copilot.react import AgentResult
    from options_copilot.reflexion import write_lesson
    db = Database(":memory:")
    with G.run_config(sandbox=True):
        with pytest.raises(G.SandboxViolation):
            PaperBroker(db, FakeSources(), AS_OF)
        with pytest.raises(G.SandboxViolation):
            db.save_brief("b1", None, "entry", TICKER, "pending", "CLEAR", {})
        with pytest.raises(G.SandboxViolation):
            db.add_ticket("b1", {})
        with pytest.raises(G.SandboxViolation):
            write_lesson(ReflectionTrigger.VERIFIER, TICKER, {}, llm=None, store=RecordingStore(), masker=Masker(PV))

        class Box:
            ticker = TICKER

            def accepted_brief_parts(self):
                return {"verdict": "CLEAR", "verifier": []}
        with pytest.raises(G.SandboxViolation):
            make_brief(AgentResult("r", "accepted", "x"), Box(), db, AS_OF)
    PaperBroker(db, FakeSources(), AS_OF)   # outside the sandbox the broker is usable again


def test_guardrails_can_only_be_switched_off_in_the_sandbox():
    with pytest.raises(ValueError):
        with G.run_config(disabled={"verifier"}):
            pass
    with pytest.raises(ValueError):
        with G.run_config(sandbox=True, disabled={"budgets"}):   # not ablatable
            pass
    assert not G.disabled("verifier") and not G.sandboxed()


def test_verifier_off_only_changes_results_inside_the_sandbox(pb):
    from options_copilot.verifier import InsiderActivity
    blocked = FakeSources(insider=InsiderActivity(100, 10_000, 5))
    normal, _ = run(pb, blocked)
    assert normal.outcome == "NO_TRADE"
    ablated, db = run(pb, blocked, sandbox=True, disabled=frozenset({"verifier"}))
    assert ablated.outcome == "SELECTED" and ablated.state["brief_id"] is None and db.briefs() == []


# --- harness, cases and snapshots ----------------------------------------------------------------

def test_cases_need_owner_labels(tmp_path):
    from options_copilot.eval_harness import CaseError, load_cases
    p = tmp_path / "cases.local.yaml"
    p.write_text(yaml.safe_dump({"cases": [{"id": "x", "ticker": "META", "label": {"expected_verdict": None,
                                                                                     "expected_outcome": None}}]}))
    with pytest.raises(CaseError, match="labels are yours"):
        load_cases(p)
    p.write_text(yaml.safe_dump({"cases": [{"id": "x", "ticker": "META", "label": {"expected_verdict": "BLOCK",
                                                                                     "expected_outcome": "BLOCK"}}],
                                 "judge_labels": [{"brief_id": "b1", "faithful": None}, {"brief_id": "b2",
                                                                                          "faithful": True}]}))
    cases, judges = load_cases(p)
    assert cases[0]["id"] == "x" and [j["brief_id"] for j in judges] == ["b2"]


def test_harness_on_synthetic_cases(pb):
    from options_copilot.eval_harness import Harness
    db = Database(":memory:")
    srcs = {"ok": lambda: DenseSources(), "blk": lambda: EvidenceSources(first="BLOCK")}
    h = Harness(lambda c: (srcs[c["id"]](), AS_OF, []), RoleLLM(), db, pb, Masker(PV), PV)
    cases = [{"id": "ok", "ticker": TICKER, "label": {"expected_verdict": "CLEAR", "expected_outcome": "trade"}},
             {"id": "blk", "ticker": TICKER, "label": {"expected_verdict": "BLOCK", "expected_outcome": "BLOCK"}}]
    res = h.evaluate(cases, repeats=2)
    a = res["summary"]["aggregate"]
    assert a["runs"] == 4 and a["unsafe_passed"] == 0 and a["outcome_accuracy"] == 1.0
    assert db.briefs() == []
    abl = h.ablation(cases, configs=("verifier", "judge"))
    assert [r["config"] for r in abl["rows"]] == ["full", "-verifier", "-judge"]
    assert db.briefs() == []


def test_snapshot_round_trip_replays_identical_retrieval(tmp_path, store, pb):
    from options_copilot.eval_harness import capture_snapshot, load_snapshot, snapshot_store
    text = "merger agreement acquisition material definitive agreement tender offer results of operations"
    store.add_chunks([filing("f1", text, ticker=TICKER), filing("f2", text + " closing", "Item 8.01", ticker=TICKER),
                      filing("old", text, filed=20250101, ticker=TICKER)])
    store.add_reflection(Reflection("Synthetic lesson: spreads near earnings.", ReflectionTrigger.VERIFIER, TICKER))
    db = Database(":memory:")
    db.add_iv_snapshot(TICKER, "2026-10-29", 0.31, None, None)
    out = capture_snapshot("case1", TICKER, AS_OF, live_sources=FakeSources(), store=store, db=db, pb=pb, root=tmp_path)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["chunks"] == 2 and meta["lessons"] == 1 and meta["chains"] >= 1   # out-of-window chunk excluded
    snap = load_snapshot("case1", tmp_path)
    replay = snapshot_store(snap, store.cfg, embedding_function=BagOfWordsEmbedding())
    q = f"{TICKER} material definitive agreement merger"
    a, b = store.search_filings(TICKER, q, AS_OF), replay.search_filings(TICKER, q, AS_OF)
    assert [c.source_url for c in a.chunks] == [c.source_url for c in b.chunks] and a.coverage == b.coverage
    assert snap.market["ticker_data"].ticker == TICKER and snap.meta["iv_history"] == [["2026-10-29", 0.31]]


def test_case_file_and_snapshots_are_gitignored_and_audited(repo_root):
    import subprocess
    import sys
    sys.path.insert(0, str(repo_root / "scripts"))
    from privacy_audit import check_path
    assert check_path("eval/cases.local.yaml", []) and check_path("data/eval_snapshots/m/market.pkl", [])
    assert check_path("eval_snapshots/m/chunks.json", [])
    ignored = subprocess.run(["git", "check-ignore", "eval/cases.local.yaml", "data/eval_snapshots/m/chunks.json"],
                             cwd=repo_root, capture_output=True, text=True)
    assert ignored.returncode == 0 and len(ignored.stdout.split()) == 2


# --- review queue --------------------------------------------------------------------------------

def test_risk_score_is_deterministic_and_tiers(pb):
    cfg = pb["guardrails.review"]
    assert G.risk_score({}, cfg)[:2] == (0, "standard")
    expected = {"verifier_cautions": "standard", "critic_citation_unsupported": "standard",
                "revisions_all_duplicates": "standard", "landmine_caution": "look_closely",
                "low_confidence": "look_closely", "grader_zero_relevant": "look_closely",
                "budget_exhausted": "look_closely", "judge_flag": "look_closely"}
    assert set(expected) == set(G.SIGNALS)
    for sig, tier in expected.items():
        assert G.risk_score({sig: 1}, cfg)[1] == tier, sig
    assert G.risk_score({"verifier_cautions": 9}, cfg)[0] == 3   # capped
    meta = {"verifier_cautions": 1, "landmine_caution": 1, "critic_citation_unsupported": 1,
            "revisions_all_duplicates": 1}
    score, tier, parts = G.risk_score(meta, cfg)
    assert (score, tier) == (5, "escalated") and parts[0] == ("landmine_caution", 2)
    assert G.risk_score(dict(meta), cfg) == G.risk_score(dict(reversed(list(meta.items()))), cfg)


def _brief(db, bid, ticker="MRVL", risks=(), landmine="CLEAR", signals=None, run_id=None, judge=None):
    db.save_brief(bid, run_id, "entry", ticker, "pending", "CAUTION" if risks else "CLEAR",
                  {"proposal": {"strategy": "iron_condor", "legs": [], "ticker": ticker}, "numbers": {},
                   "risks": list(risks), "landmine_verdict": landmine, "signals": signals or {}, "judge": judge,
                   "citations": [], "alternatives": []})


def test_review_queue_orders_by_tier_and_never_approves(pb):
    from options_copilot.review import queue, render_queue, render_view
    db = Database(":memory:")
    _brief(db, "b-std", risks=["x"])
    _brief(db, "b-esc", risks=["a", "b"], landmine="CAUTION", judge={"flags": [{"claim": "c"}]})
    _brief(db, "b-look", signals={"grader_zero_relevant": 1})
    rid = db.start_run("multi_agent", "META", "fake", "sha")
    db.finish_run(rid, "BLOCK")
    db.add_guardrail_event(rid, 1, None, "coverage_guard", "pre", "static", "pass", None, "ok")
    q = queue(db, pb)
    assert [i["brief_id"] for i in q["trades"]] == ["b-esc", "b-look", "b-std"]
    assert [i["tier"] for i in q["trades"]] == ["escalated", "look_closely", "standard"]
    assert q["info"][0]["outcome"] == "BLOCK" and "no trade" in q["info"][0]["next"]
    render_queue(q)
    view = render_view(db, pb, "b-esc")
    assert "approve   python -m options_copilot.cli approve b-esc" in view and "--code <" in view and "defer" in view
    assert len(view.splitlines()) <= 15
    assert {b["status"] for b in db.briefs()} == {"pending"} and db.tickets() == []
    db.defer("b-esc")
    assert queue(db, pb)["trades"][-1]["brief_id"] == "b-esc"


def test_no_auto_approve_path_exists(repo_root):
    """Nothing in the Phase 6 modules can approve a brief or write a ticket."""
    for mod in ("guardrails", "review", "observability", "evaluation", "eval_harness", "judge"):
        src = (repo_root / "options_copilot" / f"{mod}.py").read_text()
        assert "approve(" not in src and "add_ticket" not in src and "decide(" not in src, mod
    review_src = (repo_root / "options_copilot" / "review.py").read_text()
    assert "import approve" not in review_src and "briefs import" not in review_src


def test_reject_code_is_validated_and_stored(pb):
    from options_copilot.briefs import reject
    db = Database(":memory:")
    _brief(db, "b1")
    _brief(db, "b2")
    codes = tuple(pb["guardrails.reject_codes"])
    with pytest.raises(ValueError, match="unknown reject code"):
        reject(db, "b1", "too close", llm=None, store=None, masker=Masker(PV), code="vibes", codes=codes)
    assert db.brief("b1")["status"] == "pending"
    lesson = reject(db, "b2", "short call too near resistance", llm=None, store=None, masker=Masker(PV),
                    code="too_close_to_resistance", codes=codes)
    assert db.brief("b2")["decision_code"] == "too_close_to_resistance"
    assert "code:too_close_to_resistance" in lesson.tags


def test_playbook_reject_codes_validation(tmp_path, repo_root):
    from options_copilot.playbook import PlaybookError, load_playbook
    raw = yaml.safe_load((repo_root / "playbook.yaml").read_text())
    raw["guardrails"]["reject_codes"] = ["event_risk", "event_risk"]
    p = tmp_path / "playbook.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(PlaybookError, match="reject_codes"):
        load_playbook(p)


# --- metrics -------------------------------------------------------------------------------------

def test_metrics_match_hand_counted_fixtures(pb):
    from options_copilot.observability import metrics, render
    db = Database(":memory:")
    r1 = db.start_run("multi_agent", "META", "fake", "sha")
    db.finish_run(r1, "SELECTED")
    r2 = db.start_run("multi_agent", "NVDA", "fake", "sha")
    db.finish_run(r2, "NO_TRADE")
    r3 = db.start_run("multi_agent", "AAPL", "fake", "sha")
    db.finish_run(r3, "INSUFFICIENT_EVIDENCE")
    for rid, agent, ms, calls, conf in [(r1, "researcher", 1000, 2, 0.9), (r1, "strategist", 3000, 5, None),
                                        (r2, "researcher", 2000, 1, 0.5), (r2, "strategist", 5000, 3, None)]:
        db.add_span(rid, f"{agent}-{rid}", None, agent, "one-way", "2026-11-15T00:00:00", "", ms, calls, {}, {},
                    confidence=conf)
    db.add_span(r1, "session", None, "session", "supervisor", "", "", 0, 0, {}, None, fallbacks=1)
    db.add_span(r2, "err", None, "learner", "one-way", "", "", 10, 0, {}, None, error="boom")
    for rid, seq, g, dec, sig in [(r1, 1, "coverage_guard", "pass", None), (r1, 2, "judge", "flag", "judge_flag"),
                                  (r2, 1, "coverage_guard", "pass", None), (r2, 2, "judge", "pass", None),
                                  (r3, 1, "coverage_guard", "block", None)]:
        db.add_guardrail_event(rid, seq, None, g, G.BY_NAME[g].stage, G.BY_NAME[g].kind, dec, sig, "x")
    _brief(db, "b1", risks=["a"])
    _brief(db, "b2", risks=["a", "b"], landmine="CAUTION")
    from options_copilot.briefs import reject
    reject(db, "b1", "event soon", llm=None, store=None, masker=Masker(PV), code="event_risk",
           codes=tuple(pb["guardrails.reject_codes"]))
    db.add_eval("eval", {"unsafe_passed": 0})
    m = metrics(db, pb)
    assert m["runs_by_outcome"]["multi_agent"] == {"SELECTED": 1, "NO_TRADE": 1, "INSUFFICIENT_EVIDENCE": 1}
    assert m["guardrails"]["coverage_guard"] == {"pass": 2, "block": 1}
    assert m["judge_flag_rate"] == 0.5
    assert m["latency_per_agent"]["strategist"] == {"n": 2, "p50_s": 3.0, "p95_s": 5.0}
    assert m["llm_calls_per_run"]["mean"] == 5.5
    assert m["claude_fallbacks"] == 1 and m["span_errors"] == 1
    assert m["reject_codes"] == {"event_risk": 1}
    assert m["latest_eval_unsafe_passed"]["unsafe_passed"] == 0
    assert m["landmine_confidence"]["bins"]["0.8-1.0"] == 1 and m["landmine_confidence"]["bins"]["0.4-0.6"] == 1
    assert m["review_items"] == 3 and m["escalation_rate"]["standard"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["escalation_rate"]["info_block_or_insufficient"] == pytest.approx(1 / 3, abs=1e-4)
    assert "unsafe trades passed 0" in render(m)


def test_new_span_fields_are_recorded(pb):
    r, db = run(pb, DenseSources(), llm=RoleLLM(critic=veto_first))
    spans = {s["agent"]: s for s in db.spans(r.run_id)}
    assert spans["researcher"]["decision"] == "CLEAR" and spans["researcher"]["ticker"] == TICKER
    assert spans["researcher"]["as_of"] == AS_OF.isoformat() and spans["verifier_gate"]["decision"] == "SELECTED"
    assert spans["session"]["ticker"] == TICKER


# --- liquidity floor (added after the 2026-11-16 eval: every "trade" case was blocked on thin weeklies) ---

class ThinSources(FakeSources):
    """Two expiries in the DTE window: EXP is liquid, a later one is a thin weekly that fits the DTE
    target better (as Dec 31 did for META, MSFT and BAC)."""
    def __init__(self, thin_oi=3, **kw):
        super().__init__(**kw)
        self.thin_oi = thin_oi

    def expiries(self, ticker):
        from datetime import timedelta
        return [EXP_LIQUID, AS_OF + timedelta(days=46)]

    def chain(self, ticker, expiry):
        from dataclasses import replace
        from agent_fakes import chain
        q = chain()
        if expiry == EXP_LIQUID:
            return [replace(x, expiry=expiry) for x in q], 1.2
        return [replace(x, expiry=expiry, open_interest=self.thin_oi) for x in q], 1.2


from datetime import timedelta as _td   # noqa: E402
EXP_LIQUID = AS_OF + _td(days=38)


def _tot(pb, sources):
    from options_copilot.agent_tools import ToolBox
    from options_copilot.tot import explore
    db = Database(":memory:")
    box = ToolBox(TICKER, sources, pb, AS_OF, PV)
    return explore(box, pb, None, db, Masker(PV)), box


def test_thin_expiry_is_withheld_and_a_liquid_one_is_chosen(pb):
    result, _ = _tot(pb, ThinSources(thin_oi=3))
    expiries = {n.spec.expiry for n in result.tree.nodes if n.depth == 2}
    assert expiries == {EXP_LIQUID.isoformat()}
    thin = [t for n in result.tree.nodes for t in n.detail.get("thin_expiries", [])]
    assert thin and "below 500" in thin[0]
    assert result.outcome == "SELECTED" and result.selected.spec.expiry == EXP_LIQUID.isoformat()


def test_floor_uses_the_verifier_open_interest_minimum(pb):
    at_floor, _ = _tot(pb, ThinSources(thin_oi=500))
    assert len({n.spec.expiry for n in at_floor.tree.nodes if n.depth == 2}) == 2   # 500 is not thin
    below, _ = _tot(pb, ThinSources(thin_oi=499))
    assert len({n.spec.expiry for n in below.tree.nodes if n.depth == 2}) == 1


def test_all_thin_expiries_give_no_trade_with_the_reason(pb):
    class AllThin(ThinSources):
        def chain(self, ticker, expiry):
            from dataclasses import replace
            from agent_fakes import chain
            return [replace(x, expiry=expiry, open_interest=10) for x in chain()], 1.2
    result, _ = _tot(pb, AllThin())
    assert result.outcome == "NO_TRADE" and result.stats["builds"] == 0
    assert any("too thinly traded" in r for r in result.reasons)


def test_depth2_prunes_a_thin_expiry_even_if_offered(pb):
    from options_copilot.agent_tools import ToolBox
    from options_copilot.tot import Explorer, Node, Spec, Status
    box = ToolBox(TICKER, ThinSources(thin_oi=3), pb, AS_OF, PV)
    ex = Explorer(box, pb, None, Database(":memory:"), Masker(PV))
    node = Node("n9", "n1", 2, Spec("put_credit_spread", (AS_OF + _td(days=46)).isoformat(), None), source="test")
    ex._score_expiry(node)
    assert node.status is Status.PRUNED and node.prune_reason.startswith("thin expiry")


def test_floor_is_reported_as_a_pre_stage_event(pb):
    r, db = run(pb, ThinSources(thin_oi=3))
    ev = next(e for e in db.guardrail_events(r.run_id) if e["guardrail"] == "playbook_bands")
    assert ev["decision"] == "block" and "liquidity floor" in ev["reason"] and ev["stage"] == "pre"
    assert r.outcome == "SELECTED"
