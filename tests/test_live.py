"""Optional live checks, skipped by default. Run with `pytest -m llm` or `pytest -m network`."""
from datetime import date

import pytest

from options_copilot.agent import landmine_check
from options_copilot.config import load_settings


@pytest.mark.llm
def test_ollama_returns_a_valid_verdict_without_rag():
    from options_copilot.llm import build_llm

    out = landmine_check("MSFT", "put credit spread", use_retrieval=False, store=None,
                         llm=build_llm(load_settings()), as_of=date.today())
    assert out["llm_verdict"] in {"CLEAR", "CAUTION", "BLOCK", "INSUFFICIENT_EVIDENCE"}
    assert out["verdict"] != "CLEAR"


@pytest.mark.network
def test_edgar_lists_recent_filings(tmp_path):
    from options_copilot.edgar import EdgarClient

    settings = load_settings()
    if not settings.sec_user_agent:
        pytest.skip("SEC_USER_AGENT not set")
    client = EdgarClient(settings.sec_user_agent, tmp_path)
    assert client.cik_for("MSFT") > 0
