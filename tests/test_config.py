from options_copilot.config import load_settings


def test_defaults_are_repo_relative_and_local(tmp_path):
    s = load_settings(env={}, root=tmp_path)
    assert s.playbook_path == tmp_path / "playbook.yaml"
    assert s.db_path == tmp_path / "data" / "copilot.db"
    assert s.chroma_dir == tmp_path / "data" / "chroma"
    assert s.ollama_model == "qwen3:8b" and not s.ollama_think
    assert not s.has_anthropic_fallback
    assert s.playbook_source is None


def test_secrets_never_appear_in_repr(tmp_path):
    agent = "Test Person " + "person" + "@" + "example.com"
    s = load_settings(env={"SEC_USER_AGENT": agent, "ANTHROPIC_API_KEY": "test-key-123"}, root=tmp_path)
    assert s.sec_user_agent == agent and s.has_anthropic_fallback
    assert agent not in repr(s) and "test-key-123" not in repr(s)


def test_env_overrides(tmp_path):
    s = load_settings(env={"OLLAMA_THINK": "true", "LLM_TEMPERATURE": "0.5", "DATA_DIR": "elsewhere",
                           "PLAYBOOK_SOURCE": "../private.md"}, root=tmp_path)
    assert s.ollama_think and s.llm_temperature == 0.5
    assert s.data_dir == tmp_path / "elsewhere"
    assert s.playbook_source == tmp_path / ".." / "private.md"
