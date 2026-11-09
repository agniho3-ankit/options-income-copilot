"""CrewAI <-> our LLMClient. Every CrewAI agent talks to Qwen through this adapter: the same
local Ollama client as the rest of the copilot (qwen3:8b, think off), with our Claude fallback
only when Ollama fails and a key is set. There is no litellm and no OpenAI path; building an
agent with any other LLM raises (see agents.make_agent)."""
from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic import PrivateAttr

from . import privacy  # noqa: F401 - telemetry off before crewai loads
from crewai.llms.base_llm import BaseLLM

from ..llm import LLMClient

MODEL = "ollama/qwen3:8b"
# CrewAI logs a warning per call that it estimates tokens for custom LLMs; that is expected here
logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)


class CopilotLLM(BaseLLM):
    llm_type: str = "copilot"
    model: str = MODEL
    provider: str = "ollama"
    _client: Any = PrivateAttr()
    _on_call: Any = PrivateAttr(default=None)

    def __init__(self, client: LLMClient, on_call: Callable[[str, int, str], None] | None = None, **data: Any):
        super().__init__(model=MODEL, provider="ollama", **data)
        self._client, self._on_call = client, on_call

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None,
             from_agent=None, response_model=None) -> str:
        if isinstance(messages, str):
            msgs = [{"role": "user", "content": messages}]
        else:
            msgs = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in messages]
        resp = self._client.chat(msgs)
        text = resp.content or ""
        for stop in self.stop or []:  # CrewAI's ReAct stop words ("\nObservation:"), applied here
            if stop and stop in text:
                text = text[:text.index(stop)]
        if self._on_call is not None:
            self._on_call(getattr(from_agent, "role", "agent"), resp.eval_tokens, text)
        return text

    def supports_function_calling(self) -> bool:
        return False  # CrewAI uses its text tool protocol; our client handles plain chat

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 16384
