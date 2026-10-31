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
