"""
Agentic Memory HTTP Service

REST API wrapper for the Agentic Memory system.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global memory system instance
memory_system = None
bm25_retriever = None


class AddMemoryRequest(BaseModel):
    """Request body for adding a memory."""

    content: str = Field(..., description="The content to store")
    tags: list[str] | None = Field(default=None, description="Optional tags")
    category: str | None = Field(default=None, description="Optional category")
    timestamp: str | None = Field(
        default=None, description="Optional timestamp (YYYYMMDDHHmm format)"
    )
    # Code context metadata for RAG enrichment
    file_path: str | None = Field(default=None, description="Source file path")
    class_name: str | None = Field(default=None, description="Containing class/struct/protocol")
    function_name: str | None = Field(default=None, description="Containing function")
    line_range: str | None = Field(default=None, description="Line range (e.g., '100-150')")
    language: str | None = Field(default=None, description="Programming language")
    metadata: dict | None = Field(default=None, description="Arbitrary key-value metadata")


class UpdateMemoryRequest(BaseModel):
    """Request body for updating a memory."""

    content: str | None = Field(default=None, description="New content")
    tags: list[str] | None = Field(default=None, description="New tags")
    category: str | None = Field(default=None, description="New category")


class SearchRequest(BaseModel):
    """Request body for searching memories."""

    query: str = Field(..., description="Search query")
    k: int = Field(default=5, description="Number of results to return")


class FilteredSearchRequest(BaseModel):
    """Request body for metadata-filtered search."""

    query: str = Field(..., description="Search query")
    k: int = Field(default=5, description="Number of results to return")
    file_path: str | None = Field(default=None, description="Filter by file path (substring match)")
    class_name: str | None = Field(default=None, description="Filter by exact class name")
    language: str | None = Field(default=None, description="Filter by programming language")
    category: str | None = Field(default=None, description="Filter by category")


class MemoryResponse(BaseModel):
    """Response containing a memory."""

    id: str
    content: str
    tags: list[str] | None = None
    context: str | None = None
    keywords: list[str] | None = None
    category: str | None = None
    timestamp: str | None = None
    file_path: str | None = None
    class_name: str | None = None
    function_name: str | None = None
    line_range: str | None = None
    language: str | None = None
    metadata: dict | None = None


class SearchResultResponse(BaseModel):
    """Response containing search results."""

    results: list[dict[str, Any]]
    query: str
    count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    memory_count: int | None = None
    model_name: str | None = None
    llm_backend: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and cleanup."""
    global memory_system

    # Initialize the memory system
    try:
        from agentic_memory.memory_system import AgenticMemorySystem
        from agentic_memory.retrievers import BM25Retriever

        llm_backend = os.getenv("LLM_BACKEND", "litellm")
        memory_system = AgenticMemorySystem(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            llm_backend=llm_backend,
            llm_model=os.getenv("LLM_MODEL", "memory-evolution"),
            llm_base_url=os.getenv("LITELLM_GATEWAY_URL", "http://localhost:4000") if llm_backend == "litellm" else None,
            api_key=os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )

        # Build BM25 index from existing memories
        bm25_retriever = BM25Retriever()
        for mem_id, mem in memory_system.memories.items():
            meta = {}
            for attr in ("tags", "category", "file_path", "class_name", "function_name", "line_range", "language"):
                if hasattr(mem, attr) and getattr(mem, attr) is not None:
                    meta[attr] = getattr(mem, attr)
            bm25_retriever.add_document(doc_id=mem_id, text=mem.content, metadata=meta)
        logger.info(f"Agentic Memory system initialized, BM25 indexed {bm25_retriever.count} docs")
    except Exception as e:
        logger.error(f"Failed to initialize memory system: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down Agentic Memory service")


app = FastAPI(
    title="Agentic Memory Service",
    description="HTTP API for the Agentic Memory system",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health and memory system status."""
    return HealthResponse(
        status="healthy" if memory_system else "unhealthy",
        timestamp=datetime.now().isoformat(),
        memory_count=len(memory_system.memories) if memory_system else None,
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        llm_backend=os.getenv("LLM_BACKEND", "litellm"),
    )


@app.post("/memory/add", response_model=dict, tags=["Memory"])
async def add_memory(request: AddMemoryRequest):
    """Add a new memory to the system."""
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    try:
        kwargs = {
            "tags": request.tags,
            "category": request.category,
            "timestamp": request.timestamp,
        }
        # Pass code context metadata if provided
        if request.file_path is not None:
            kwargs["file_path"] = request.file_path
        if request.class_name is not None:
            kwargs["class_name"] = request.class_name
        if request.function_name is not None:
            kwargs["function_name"] = request.function_name
        if request.line_range is not None:
            kwargs["line_range"] = request.line_range
        if request.language is not None:
            kwargs["language"] = request.language
        if request.metadata is not None:
            kwargs["metadata"] = request.metadata

        memory_id = memory_system.add_note(
            content=request.content,
            **kwargs,
        )
        # Keep BM25 index in sync
        if bm25_retriever is not None:
            bm25_meta = {k: v for k, v in kwargs.items() if v is not None and k != "timestamp"}
            bm25_retriever.add_document(doc_id=memory_id, text=request.content, metadata=bm25_meta)

        logger.info(f"Added memory with ID: {memory_id}")
        return {"id": memory_id, "status": "created"}
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{memory_id}", response_model=MemoryResponse, tags=["Memory"])
async def get_memory(memory_id: str):
    """Get a memory by ID."""
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    try:
        memory = memory_system.read(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

        return MemoryResponse(
            id=memory_id,
            content=memory.content,
            tags=memory.tags if hasattr(memory, "tags") else None,
            context=memory.context if hasattr(memory, "context") else None,
            keywords=memory.keywords if hasattr(memory, "keywords") else None,
            category=memory.category if hasattr(memory, "category") else None,
            file_path=memory.file_path if hasattr(memory, "file_path") else None,
            class_name=memory.class_name if hasattr(memory, "class_name") else None,
            function_name=memory.function_name if hasattr(memory, "function_name") else None,
            line_range=memory.line_range if hasattr(memory, "line_range") else None,
            language=memory.language if hasattr(memory, "language") else None,
            metadata=memory.metadata if hasattr(memory, "metadata") else None,
        )
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    except Exception as e:
        logger.error(f"Error reading memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memory/{memory_id}", response_model=dict, tags=["Memory"])
async def update_memory(memory_id: str, request: UpdateMemoryRequest):
    """Update an existing memory."""
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    try:
        # Check if memory exists
        memory = memory_system.read(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

        # Update with provided fields
        if request.content:
            memory_system.update(memory_id, content=request.content)

        logger.info(f"Updated memory: {memory_id}")
        return {"id": memory_id, "status": "updated"}
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{memory_id}", response_model=dict, tags=["Memory"])
async def delete_memory(memory_id: str):
    """Delete a memory."""
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    try:
        memory_system.delete(memory_id)
        logger.info(f"Deleted memory: {memory_id}")
        return {"id": memory_id, "status": "deleted"}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search", response_model=SearchResultResponse, tags=["Memory"])
async def search_memories(request: SearchRequest):
    """Search memories using semantic search."""
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    try:
        results = memory_system.search_agentic(request.query, k=request.k)
        return SearchResultResponse(
            results=results,
            query=request.query,
            count=len(results),
        )
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search/filtered", response_model=SearchResultResponse, tags=["Memory"])
async def search_memories_filtered(request: FilteredSearchRequest):
    """Search memories with metadata filtering (ChromaDB where clause)."""
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    try:
        # Build ChromaDB where clause
        where_clauses = []
        if request.class_name is not None:
            where_clauses.append({"class_name": request.class_name})
        if request.language is not None:
            where_clauses.append({"language": request.language})
        if request.category is not None:
            where_clauses.append({"category": request.category})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        # Over-fetch for post-filtering on file_path substring
        fetch_k = request.k * 2 if request.file_path else request.k
        results = memory_system.retriever.search_filtered(
            request.query, k=fetch_k, where=where
        )

        # Post-filter for file_path substring match
        if request.file_path is not None:
            results = [
                r for r in results
                if r.get("file_path", "") and request.file_path in r.get("file_path", "")
            ]

        return SearchResultResponse(
            results=results[: request.k],
            query=request.query,
            count=min(len(results), request.k),
        )
    except Exception as e:
        logger.error(f"Error in filtered search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search/bm25", response_model=SearchResultResponse, tags=["Memory"])
async def search_memories_bm25(request: SearchRequest):
    """Search memories using BM25 term-frequency matching (exact keyword search)."""
    if not bm25_retriever:
        raise HTTPException(status_code=503, detail="BM25 index not initialized")

    try:
        results = bm25_retriever.search(request.query, k=request.k)
        return SearchResultResponse(
            results=results,
            query=request.query,
            count=len(results),
        )
    except Exception as e:
        logger.error(f"Error in BM25 search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search/hybrid", response_model=SearchResultResponse, tags=["Memory"])
async def search_memories_hybrid(request: SearchRequest):
    """Hybrid search: merge vector (semantic) + BM25 (keyword) results via reciprocal rank fusion."""
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    try:
        # Get vector results
        vector_results = memory_system.retriever.search_filtered(request.query, k=request.k)
        # Get BM25 results
        bm25_results = bm25_retriever.search(request.query, k=request.k) if bm25_retriever else []

        # Reciprocal Rank Fusion (k=60 is standard)
        rrf_k = 60
        scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        for rank, r in enumerate(vector_results):
            doc_id = r.get("id", "")
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            result_map[doc_id] = r

        for rank, r in enumerate(bm25_results):
            doc_id = r.get("id", "")
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if doc_id not in result_map:
                result_map[doc_id] = r

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        merged = []
        for doc_id in sorted_ids[: request.k]:
            entry = result_map[doc_id]
            entry["rrf_score"] = round(scores[doc_id], 6)
            merged.append(entry)

        return SearchResultResponse(
            results=merged,
            query=request.query,
            count=len(merged),
        )
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/list", response_model=dict, tags=["Memory"])
async def list_memories(limit: int = 100, offset: int = 0):
    """List all memories with pagination."""
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    try:
        all_memories = list(memory_system.memories.keys())[offset : offset + limit]
        return {
            "ids": all_memories,
            "total": len(memory_system.memories),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Error listing memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8100"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port)
