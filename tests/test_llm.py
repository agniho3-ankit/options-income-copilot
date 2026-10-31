import pytest

from options_copilot.llm import AnthropicClient, FallbackClient, LLMError, LLMResponse, OllamaClient

SECRET = "test-key-not-real"


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class Session:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self.exc:
            raise self.exc
        return self.resp


MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def test_ollama_payload_and_reply():
    s = Session(Resp(200, {"message": {"content": "{\"verdict\": \"CLEAR\"}"}}))
    client = OllamaClient("http://localhost:11434/api/chat", "qwen3:8b", session=s)
    out = client.chat(MSGS, json_schema={"type": "object"})
    body = s.calls[0]["json"]
    assert body["stream"] is False and body["think"] is False and body["format"] == {"type": "object"}
    assert body["options"]["temperature"] == 0.2 and "tools" not in body
    assert out.content.startswith("{") and out.model == "ollama:qwen3:8b"


@pytest.mark.parametrize("session", [Session(exc=ConnectionError("down")), Session(Resp(500, {}))])
def test_ollama_failures_raise_llm_error(session):
    with pytest.raises(LLMError):
        OllamaClient("u", "m", session=session).chat(MSGS)


def test_anthropic_payload_moves_system_and_hides_key():
    s = Session(Resp(200, {"content": [{"type": "text", "text": "ok"}]}))
    client = AnthropicClient(SECRET, "claude-opus-5-5", session=s)
    assert client.chat(MSGS, json_schema={"type": "object"}).content == "ok"
    body = s.calls[0]["json"]
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["system"].startswith("sys") and "JSON Schema" in body["system"]
    assert s.calls[0]["headers"]["x-api-key"] == SECRET
    assert SECRET not in repr(client)
    failing = AnthropicClient(SECRET, "m", session=Session(exc=ConnectionError(SECRET)))
    with pytest.raises(LLMError) as err:
        failing.chat(MSGS)
    assert SECRET not in str(err.value)


def test_anthropic_needs_a_key():
    with pytest.raises(LLMError):
        AnthropicClient("", "m")


class Stub:
    def __init__(self, name, reply=None, fail=False):
        self.name, self.reply, self.fail, self.calls = name, reply, fail, 0

    def chat(self, messages, *, json_schema=None, tools=None):
        self.calls += 1
        if self.fail:
            raise LLMError(f"{self.name} down")
        return LLMResponse(self.reply, self.name)


def test_fallback_only_on_primary_failure():
    primary, backup = Stub("p", "from p"), Stub("b", "from b")
    assert FallbackClient(primary, backup).chat(MSGS).content == "from p" and backup.calls == 0
    assert FallbackClient(Stub("p", fail=True), backup).chat(MSGS).content == "from b"
    with pytest.raises(LLMError):
        FallbackClient(Stub("p", fail=True), None).chat(MSGS)
    with pytest.raises(LLMError, match="fallback failed"):
        FallbackClient(Stub("p", fail=True), Stub("b", fail=True)).chat(MSGS)
