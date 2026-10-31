"""Shared fixtures. Unit tests must run with no network and no LLM."""
from __future__ import annotations

import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    """Fail any unit test that tries to open a socket (llm/network tests are exempt)."""
    if request.node.get_closest_marker("llm") or request.node.get_closest_marker("network"):
        return

    def guard(*args, **kwargs):
        raise RuntimeError("unit tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


class BagOfWordsEmbedding:
    """Deterministic, offline stand-in for MiniLM: hashed word counts in 64 dimensions."""

    def __init__(self) -> None:
        pass

    def __call__(self, input):  # noqa: A002 - Chroma's protocol names this `input`
        import hashlib
        import re

        import numpy as np

        out = []
        for text in input:
            vec = np.zeros(64, dtype=np.float32)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            vec[0] += 0.01  # never an all-zero vector
            out.append(vec)
        return out

    @staticmethod
    def name() -> str:
        return "bag-of-words-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config):
        return BagOfWordsEmbedding()

    def is_legacy(self) -> bool:
        return False

    @staticmethod
    def supported_spaces() -> list[str]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def default_space() -> str:
        return "cosine"


@pytest.fixture
def store(tmp_path):
    from options_copilot.store import Store

    return Store(tmp_path / "chroma", embedding_function=BagOfWordsEmbedding())
