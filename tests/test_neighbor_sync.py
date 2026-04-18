import tempfile
from agentic_memory.memory_system import AgenticMemorySystem


def test_neighbor_tag_update_persists_to_chroma():
    """When process_memory updates a neighbor's tags, ChromaDB must reflect the change."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ms = AgenticMemorySystem(persistent=True, persist_directory=tmpdir,
                                 llm_backend="litellm", llm_base_url="http://localhost:9999")
        neighbor_id = ms.add_note("existing memory about topic X")

        # Simulate process_memory updating the neighbor's tags
        ms.memories[neighbor_id].tags = ["updated-tag", "evolved"]

        # Trigger sync to ChromaDB
        ms._sync_memory_to_chroma(neighbor_id)

        # Search and verify tag is present in results
        results = ms.search_agentic("topic X", k=3)
        matching = [r for r in results if r["id"] == neighbor_id]
        assert matching, f"Neighbor {neighbor_id} not in search results"
        assert "updated-tag" in matching[0].get("tags", []), (
            f"Updated tag not in search result tags: {matching[0].get('tags')}"
        )


def test_sync_memory_to_chroma_method_exists():
    """_sync_memory_to_chroma must be a callable method on AgenticMemorySystem."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ms = AgenticMemorySystem(persistent=True, persist_directory=tmpdir,
                                 llm_backend="litellm", llm_base_url="http://localhost:9999")
        assert callable(getattr(ms, "_sync_memory_to_chroma", None)), (
            "_sync_memory_to_chroma method does not exist on AgenticMemorySystem"
        )
