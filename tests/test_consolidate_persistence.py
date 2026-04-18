import os
import tempfile
import pytest
from agentic_memory.memory_system import AgenticMemorySystem
from agentic_memory.retrievers import PersistentChromaRetriever


def _make_ms(tmpdir, **kwargs):
    return AgenticMemorySystem(
        persistent=True, persist_directory=tmpdir,
        llm_backend="litellm", llm_base_url="http://localhost:9999",
        **kwargs
    )


def test_consolidate_keeps_persistent_retriever():
    """After consolidate_memories(), retriever must still be PersistentChromaRetriever."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ms = _make_ms(tmpdir, evo_threshold=1)
        ms.add_note("memory one", tags=["test"])
        ms.add_note("memory two", tags=["test"])  # triggers consolidate at threshold=1
        assert isinstance(ms.retriever, PersistentChromaRetriever), (
            f"Expected PersistentChromaRetriever after consolidate, got {type(ms.retriever)}"
        )
        assert ms._bm25.count == len(ms.memories), (
            f"BM25 count {ms._bm25.count} != memories count {len(ms.memories)} after consolidation"
        )


def test_consolidate_respects_namespace():
    """consolidate_memories() must use namespace-aware collection name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ms = _make_ms(tmpdir, namespace="test_ns", evo_threshold=1)
        ms.add_note("memory one", tags=["test"])
        ms.add_note("memory two", tags=["test"])
        assert ms.retriever.collection.name == "memories_test_ns", (
            f"Expected collection 'memories_test_ns', got {ms.retriever.collection.name}"
        )


def test_memories_searchable_after_consolidate():
    """Memories added before consolidation must still be searchable after."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ms = _make_ms(tmpdir, evo_threshold=1)
        ms.add_note("zebra stripes and savanna grasslands", tags=["test"])
        ms.add_note("unrelated memory about python code", tags=["test"])  # triggers consolidate
        results = ms.search_agentic("zebra stripes savanna", k=3)
        ids = [r["id"] for r in results]
        assert any("zebra" in ms.memories[i].content for i in ids if i in ms.memories), (
            "Memory added before consolidation not found after consolidation"
        )
