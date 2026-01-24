"""FastAPI HTTP service wrapping the AgenticMemorySystem."""

import os
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from agentic_memory.memory_system import AgenticMemorySystem

from .models import (
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryUpdateRequest,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySearchResult,
    MemoryResponse,
    HealthResponse,
    ErrorResponse,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global memory system instance
memory_system: Optional[AgenticMemorySystem] = None

# Configuration from environment
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_BACKEND = os.getenv("LLM_BACKEND", "litellm")  # Default to LiteLLM gateway
LLM_MODEL = os.getenv("LLM_MODEL", "memory-evolution")  # LiteLLM alias
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EVO_THRESHOLD = int(os.getenv("EVO_THRESHOLD", "100"))

# Persistence settings
PERSISTENT = os.getenv("PERSISTENT", "true").lower() == "true"
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", os.path.expanduser("~/.agentic_memory"))
NAMESPACE = os.getenv("MEMORY_NAMESPACE", "default")

# LiteLLM Gateway settings
LITELLM_GATEWAY_URL = os.getenv("LITELLM_GATEWAY_URL", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the memory system."""
    global memory_system

    logger.info("Initializing AgenticMemorySystem...")
    logger.info(f"  Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"  LLM backend: {LLM_BACKEND}")
    logger.info(f"  LLM model: {LLM_MODEL}")
    logger.info(f"  Evolution threshold: {EVO_THRESHOLD}")
    logger.info(f"  Persistent storage: {PERSISTENT}")
    logger.info(f"  Persist directory: {PERSIST_DIRECTORY}")
    logger.info(f"  Namespace: {NAMESPACE}")

    try:
        memory_system = AgenticMemorySystem(
            model_name=EMBEDDING_MODEL,
            llm_backend=LLM_BACKEND,
            llm_model=LLM_MODEL,
            evo_threshold=EVO_THRESHOLD,
            api_key=OPENAI_API_KEY or LITELLM_API_KEY,
            persistent=PERSISTENT,
            persist_directory=PERSIST_DIRECTORY,
            namespace=NAMESPACE,
            llm_base_url=LITELLM_GATEWAY_URL if LLM_BACKEND == "litellm" else None,
        )
        logger.info("AgenticMemorySystem initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AgenticMemorySystem: {e}")
        raise

    yield

    logger.info("Shutting down AgenticMemorySystem...")
    memory_system = None


app = FastAPI(
    title="Agentic Memory HTTP Service",
    description="REST API wrapper for the Agentic Memory system",
    version="0.1.0",
    lifespan=lifespan,
)


def get_memory_system() -> AgenticMemorySystem:
    """Get the memory system instance or raise an error."""
    if memory_system is None:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    return memory_system


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health and return system info."""
    ms = get_memory_system()
    return HealthResponse(
        status="healthy",
        memory_count=len(ms.memories),
        model_name=EMBEDDING_MODEL,
        llm_backend=LLM_BACKEND,
        llm_model=LLM_MODEL,
        persistent=ms.persistent,
        namespace=ms.namespace,
    )


@app.post(
    "/memory/add",
    response_model=MemoryAddResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Memory"],
)
async def add_memory(request: MemoryAddRequest):
    """Add a new memory note to the system."""
    ms = get_memory_system()

    try:
        kwargs = {}
        if request.tags:
            kwargs["tags"] = request.tags
        if request.category:
            kwargs["category"] = request.category
        if request.timestamp:
            kwargs["timestamp"] = request.timestamp

        memory_id = ms.add_note(content=request.content, **kwargs)

        return MemoryAddResponse(
            id=memory_id,
            message="Memory added successfully",
        )
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/memory/{memory_id}",
    response_model=MemoryResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Memory"],
)
async def get_memory(memory_id: str):
    """Retrieve a memory by its ID."""
    ms = get_memory_system()

    note = ms.read(memory_id)
    if note is None:
        raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")

    return MemoryResponse(
        id=note.id,
        content=note.content,
        keywords=note.keywords,
        context=note.context,
        category=note.category,
        tags=note.tags,
        timestamp=note.timestamp,
        last_accessed=note.last_accessed,
        retrieval_count=note.retrieval_count,
        links=note.links if isinstance(note.links, list) else [],
    )


@app.put(
    "/memory/{memory_id}",
    response_model=MemoryResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Memory"],
)
async def update_memory(memory_id: str, request: MemoryUpdateRequest):
    """Update an existing memory."""
    ms = get_memory_system()

    # Check if memory exists
    note = ms.read(memory_id)
    if note is None:
        raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")

    # Build update kwargs
    kwargs = {}
    if request.content is not None:
        kwargs["content"] = request.content
    if request.tags is not None:
        kwargs["tags"] = request.tags
    if request.category is not None:
        kwargs["category"] = request.category
    if request.context is not None:
        kwargs["context"] = request.context
    if request.keywords is not None:
        kwargs["keywords"] = request.keywords

    if not kwargs:
        raise HTTPException(status_code=400, detail="No fields to update")

    success = ms.update(memory_id, **kwargs)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update memory")

    # Return updated memory
    updated_note = ms.read(memory_id)
    return MemoryResponse(
        id=updated_note.id,
        content=updated_note.content,
        keywords=updated_note.keywords,
        context=updated_note.context,
        category=updated_note.category,
        tags=updated_note.tags,
        timestamp=updated_note.timestamp,
        last_accessed=updated_note.last_accessed,
        retrieval_count=updated_note.retrieval_count,
        links=updated_note.links if isinstance(updated_note.links, list) else [],
    )


@app.delete(
    "/memory/{memory_id}",
    responses={404: {"model": ErrorResponse}},
    tags=["Memory"],
)
async def delete_memory(memory_id: str):
    """Delete a memory by its ID."""
    ms = get_memory_system()

    success = ms.delete(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")

    return {"message": f"Memory {memory_id} deleted successfully"}


@app.post(
    "/memory/search",
    response_model=MemorySearchResponse,
    tags=["Memory"],
)
async def search_memories(request: MemorySearchRequest):
    """Search for memories using semantic search."""
    ms = get_memory_system()

    try:
        results = ms.search_agentic(request.query, k=request.k)

        search_results = [
            MemorySearchResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                context=r.get("context", ""),
                keywords=r.get("keywords", []),
                tags=r.get("tags", []),
                timestamp=str(r.get("timestamp", "")),
                category=r.get("category", "Uncategorized"),
                score=r.get("score"),
                is_neighbor=r.get("is_neighbor", False),
            )
            for r in results
        ]

        return MemorySearchResponse(
            results=search_results,
            count=len(search_results),
        )
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/memory", tags=["Memory"])
async def list_memories(limit: int = 100, offset: int = 0):
    """List all memories with pagination."""
    ms = get_memory_system()

    memory_ids = list(ms.memories.keys())
    total = len(memory_ids)

    # Apply pagination
    paginated_ids = memory_ids[offset : offset + limit]

    memories = []
    for mid in paginated_ids:
        note = ms.memories.get(mid)
        if note:
            memories.append(
                MemoryResponse(
                    id=note.id,
                    content=note.content,
                    keywords=note.keywords,
                    context=note.context,
                    category=note.category,
                    tags=note.tags,
                    timestamp=note.timestamp,
                    last_accessed=note.last_accessed,
                    retrieval_count=note.retrieval_count,
                    links=note.links if isinstance(note.links, list) else [],
                )
            )

    return {
        "memories": memories,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
