import tempfile
import time
from datetime import datetime
from agentic_memory.memory_system import AgenticMemorySystem


def _make_ms(tmpdir):
    return AgenticMemorySystem(
        persistent=True, persist_directory=tmpdir,
        llm_backend="litellm", llm_base_url="http://localhost:9999",
    )


def test_exact_phrase_ranks_first():
    """Memory whose first line IS the query must rank #1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ms = _make_ms(tmpdir)
        # Add many agentic-levels memories to create topic saturation
        for i in range(20):
            ms.add_note(f"Agentic Levels context memory {i}: course council findings phase {i}")
        # The target memory starts with the exact query phrase
        target_id = ms.add_note(
            "agentic levels next session: PR #4 blocked on GEMINI_API_KEY, smoke test needed"
        )
        results = ms.search_agentic("agentic levels next session", k=5)
        top_id = results[0]["id"] if results else None
        assert top_id == target_id, (
            f"Expected target memory at rank 1, got {top_id}. "
            f"Full results: {[r['id'] for r in results]}"
        )


def test_recent_memory_beats_older_same_topic():
    """A recent memory must outrank an older one with similar content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ms = _make_ms(tmpdir)
        old_id = ms.add_note("agentic levels project status update alpha version complete")
        ms.memories[old_id].timestamp = "202603010900"

        new_id = ms.add_note("agentic levels project status update sprint 3 complete today")
        ms.memories[new_id].timestamp = datetime.now().strftime("%Y%m%d%H%M")

        results = ms.search_agentic("agentic levels project status", k=5)
        ranked_ids = [r["id"] for r in results]
        assert ranked_ids.index(new_id) < ranked_ids.index(old_id), (
            f"Expected new memory ({new_id}) before old ({old_id}). Order: {ranked_ids}"
        )


def test_bm25_index_populated_on_add():
    """BM25 index count must grow when memories are added."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ms = _make_ms(tmpdir)
        assert ms._bm25.count == 0
        ms.add_note("hello world test memory")
        assert ms._bm25.count == 1, f"BM25 count should be 1 after add, got {ms._bm25.count}"
