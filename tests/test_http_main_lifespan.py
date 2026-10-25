"""http_service/main.py lifespan: storage settings reach the memory system and the BM25 index is live."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "http_service"))


class FakeMemorySystem:
    last_kwargs = None

    def __init__(self, **kwargs):
        FakeMemorySystem.last_kwargs = kwargs
        self.memories = {"m1": SimpleNamespace(content="the zebra pipeline handoff", tags=["post"], category=None)}


def test_lifespan_passes_storage_settings_and_builds_bm25(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSIST_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("MEMORY_NAMESPACE", "testns")
    monkeypatch.setenv("PERSISTENT", "false")
    monkeypatch.setenv("EVO_THRESHOLD", "7")
    import main

    with patch("agentic_memory.memory_system.AgenticMemorySystem", FakeMemorySystem):
        with TestClient(main.app) as client:
            kw = FakeMemorySystem.last_kwargs
            assert kw["persist_directory"] == str(tmp_path)
            assert kw["namespace"] == "testns"
            assert kw["persistent"] is False
            assert kw["evo_threshold"] == 7

            assert main.bm25_retriever is not None
            r = client.post("/memory/search/bm25", json={"query": "zebra", "k": 3})
            assert r.status_code == 200, r.text
            assert r.json()["count"] == 1
