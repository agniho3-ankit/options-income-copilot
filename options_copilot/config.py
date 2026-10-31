"""Runtime settings from the environment (.env), with repo-relative paths.

Secrets (SEC_USER_AGENT, ANTHROPIC_API_KEY) are excluded from repr() so they never
reach logs, traces, or demo output by accident.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional until dependencies are installed
    load_dotenv = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(value: str, root: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    data_dir: Path
    playbook_path: Path
    playbook_source: Path | None
    ollama_url: str
    ollama_model: str
    ollama_think: bool
    llm_temperature: float
    anthropic_model: str
    synthetic_portfolio_value: float
    sec_user_agent: str = field(default="", repr=False)
    anthropic_api_key: str = field(default="", repr=False)

    @property
    def has_anthropic_fallback(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "copilot.db"


def load_settings(env: dict[str, str] | None = None, root: Path = REPO_ROOT) -> Settings:
    """Build Settings from `env` (tests) or from os.environ after loading root/.env."""
    if env is None:
        if load_dotenv is not None:
            load_dotenv(root / ".env", override=False)
        env = dict(os.environ)
    get = lambda key, default="": (env.get(key) or default).strip()  # noqa: E731

    source = get("PLAYBOOK_SOURCE")
    return Settings(
        repo_root=root,
        data_dir=_path(get("DATA_DIR", "data"), root),
        playbook_path=_path(get("PLAYBOOK_PATH", "playbook.yaml"), root),
        playbook_source=_path(source, root) if source else None,
        ollama_url=get("OLLAMA_URL", "http://localhost:11434/api/chat"),
        ollama_model=get("OLLAMA_MODEL", "qwen3:8b"),
        ollama_think=_bool(get("OLLAMA_THINK", "false")),
        llm_temperature=float(get("LLM_TEMPERATURE", "0.2")),
        anthropic_model=get("ANTHROPIC_MODEL", "claude-opus-5-5"),
        synthetic_portfolio_value=float(get("SYNTHETIC_PORTFOLIO_VALUE", "100000")),
        sec_user_agent=get("SEC_USER_AGENT"),
        anthropic_api_key=get("ANTHROPIC_API_KEY"),
    )
