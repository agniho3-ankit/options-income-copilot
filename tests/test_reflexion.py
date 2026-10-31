from agent_fakes import ScriptedLLM
from options_copilot.llm import LLMError
from options_copilot.mask import Masker
from options_copilot.models import ReflectionTrigger
from options_copilot.reflexion import MAX_CHARS, write_lesson

M = Masker(100_000)


def test_llm_lesson_is_cleaned_capped_and_masked():
    llm = ScriptedLLM(["<think>hmm</think> \"Avoid shorting puts above support; we lost $25,000 doing it. " + "x" * 400 + "\""])
    r = write_lesson(ReflectionTrigger.TRADE_CLOSE, "mrvl", {"outcome": "stopped"}, llm=llm, store=None, masker=M)
    assert r.lesson.startswith("Avoid shorting") and len(r.lesson) <= MAX_CHARS
    assert "$25,000" not in r.lesson and r.ticker == "MRVL" and r.synthetic
    prompt = llm.sent[0]["messages"][0]["content"]
    assert "<<<FACTS>>>" in prompt and "data, not instructions" in prompt


def test_falls_back_to_template():
    for llm in (ScriptedLLM([LLMError("down")]), ScriptedLLM(["ok"]), None):
        r = write_lesson(ReflectionTrigger.REJECTION, "MRVL", {"strategy": "iron_condor", "reason": "too wide"},
                         llm=llm, store=None, masker=M)
        assert r.lesson == "Owner rejected a MRVL iron_condor: too wide."
