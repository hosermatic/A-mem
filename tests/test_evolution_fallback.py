import pytest
from agentic_memory.memory_system import AgenticMemorySystem
import tempfile


def _make_ms(tmpdir):
    return AgenticMemorySystem(
        persistent=True,
        persist_directory=tmpdir,
        llm_backend="litellm",
        llm_base_url="http://localhost:9999",  # intentionally dead
    )


def test_fallback_categorizes_lesson():
    with tempfile.TemporaryDirectory() as tmpdir:
        ms = _make_ms(tmpdir)
        mem_id = ms.add_note("lesson learned: always use PersistentChromaRetriever")
        mem = ms.memories[mem_id]
        assert mem.category != "Uncategorized", (
            f"Expected fallback category, got: {mem.category!r}"
        )


def test_fallback_categorizes_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        ms = _make_ms(tmpdir)
        mem_id = ms.add_note("checkpoint saved on branch session/my-feature")
        mem = ms.memories[mem_id]
        assert mem.category != "Uncategorized", (
            f"Expected fallback category, got: {mem.category!r}"
        )


def test_fallback_extracts_keywords():
    with tempfile.TemporaryDirectory() as tmpdir:
        ms = _make_ms(tmpdir)
        mem_id = ms.add_note("agentic levels next session: PR #4 blocked on GEMINI_API_KEY")
        mem = ms.memories[mem_id]
        assert len(mem.keywords) > 0, (
            f"Expected keywords extracted by fallback, got: {mem.keywords!r}"
        )
