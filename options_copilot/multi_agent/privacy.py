"""Privacy and safety switches for the agent frameworks. Applied before CrewAI, LangGraph or
LangChain are imported (options_copilot.multi_agent imports this first).

- CrewAI anonymous telemetry and CrewAI cloud tracing: off.
- OpenTelemetry SDK (any exporter): disabled.
- LangSmith / LangChain tracing: forced off, and no endpoint or key is read.
- OpenAI: nothing in this project uses it. A stray default-LLM call is pointed at a closed local
  port, so it fails loudly on this machine instead of reaching a cloud API.
"""
from __future__ import annotations

import os

FORCED = {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "OTEL_SDK_DISABLED": "true",
    "OTEL_TRACES_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_LOGS_EXPORTER": "none",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING": "false",
    "OPENAI_BASE_URL": "http://127.0.0.1:9/openai-disabled",   # port 9 (discard): never a real API
    "OPENAI_API_BASE": "http://127.0.0.1:9/openai-disabled",
}
REMOVED = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT",
           "OPENAI_API_KEY", "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")


def configure() -> None:
    os.environ.update(FORCED)
    for key in REMOVED:
        os.environ.pop(key, None)


def problems() -> list[str]:
    """Settings that are not as required (empty list = all good)."""
    out = [f"{k} should be {v!r}, is {os.environ.get(k)!r}" for k, v in FORCED.items() if os.environ.get(k) != v]
    out += [f"{k} must not be set" for k in REMOVED if os.environ.get(k)]
    return out
