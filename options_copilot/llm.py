"""One chat interface over Ollama (primary) and the Claude API (fallback).

The fallback is used only when Ollama fails and ANTHROPIC_API_KEY is set. Neither client
logs prompts or responses, and the API key never appears in repr() or error messages.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class LLMClient(Protocol):
    name: str

    def chat(self, messages: list[dict[str, Any]], *, json_schema: dict[str, Any] | None = None,
             tools: list[dict[str, Any]] | None = None) -> LLMResponse: ...


def _session(session: Any):
    if session is None:
        import requests
        return requests.Session()
    return session


class OllamaClient:
    def __init__(self, url: str, model: str, *, think: bool = False, temperature: float = 0.2,
                 num_ctx: int = 8192, timeout: float = 180, session: Any = None):
        self.url, self.model, self.think = url, model, think
        self.temperature, self.num_ctx, self.timeout = temperature, num_ctx, timeout
        self._session = session
        self.name = f"ollama:{model}"

    def payload(self, messages, json_schema=None, tools=None) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                "think": self.think,
                                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx}}
        if json_schema is not None:
            body["format"] = json_schema
        if tools:
            body["tools"] = tools
        return body

    def chat(self, messages, *, json_schema=None, tools=None) -> LLMResponse:
        self._session = _session(self._session)
        try:
            resp = self._session.post(self.url, json=self.payload(messages, json_schema, tools),
                                      timeout=self.timeout)
        except Exception as exc:
            raise LLMError(f"Ollama unreachable: {type(exc).__name__}") from None
        if resp.status_code != 200:
            raise LLMError(f"Ollama returned HTTP {resp.status_code}")
        msg = resp.json().get("message", {})
        return LLMResponse(msg.get("content", ""), self.name, msg.get("tool_calls") or [])


class AnthropicClient:
    def __init__(self, api_key: str, model: str, *, max_tokens: int = 1500, timeout: float = 120,
                 session: Any = None):
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self._api_key = api_key
        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout
        self._session = session
        self.name = f"anthropic:{model}"

    def __repr__(self) -> str:
        return f"AnthropicClient(model={self.model!r})"

    def payload(self, messages, json_schema=None) -> dict[str, Any]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        if json_schema is not None:
            system += ("\n\nReply with a single JSON object that matches this JSON Schema, and nothing "
                       "else:\n" + json.dumps(json_schema))
        convo = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        # No temperature: newer Claude models manage sampling themselves.
        body: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "messages": convo}
        if system.strip():
            body["system"] = system.strip()
        return body

    def chat(self, messages, *, json_schema=None, tools=None) -> LLMResponse:
        if tools:
            raise LLMError("tool calling through the Claude fallback arrives in Phase 3")
        self._session = _session(self._session)
        headers = {"x-api-key": self._api_key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        try:
            resp = self._session.post(ANTHROPIC_URL, json=self.payload(messages, json_schema),
                                      headers=headers, timeout=self.timeout)
        except Exception as exc:
            raise LLMError(f"Claude API unreachable: {type(exc).__name__}") from None
        if resp.status_code != 200:
            raise LLMError(f"Claude API returned HTTP {resp.status_code}")
        blocks = resp.json().get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return LLMResponse(text, self.name)


class FallbackClient:
    """Try the primary client; on LLMError use the fallback if one is configured."""

    def __init__(self, primary: LLMClient, fallback: LLMClient | None = None):
        self.primary, self.fallback = primary, fallback
        self.name = primary.name + (f" (fallback {fallback.name})" if fallback else "")

    def chat(self, messages, *, json_schema=None, tools=None) -> LLMResponse:
        try:
            return self.primary.chat(messages, json_schema=json_schema, tools=tools)
        except LLMError as first:
            if self.fallback is None:
                raise
            try:
                return self.fallback.chat(messages, json_schema=json_schema, tools=tools)
            except LLMError as second:
                raise LLMError(f"{first}; fallback failed too: {second}") from None


def build_llm(settings) -> FallbackClient:
    primary = OllamaClient(settings.ollama_url, settings.ollama_model, think=settings.ollama_think,
                           temperature=settings.llm_temperature)
    fallback = (AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)
                if settings.has_anthropic_fallback else None)
    return FallbackClient(primary, fallback)
