"""Pydantic models for the HTTP service API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class MemoryAddRequest(BaseModel):
    """Request model for adding a new memory."""
    content: str = Field(..., description="The main text content of the memory")
    tags: Optional[List[str]] = Field(default=None, description="Classification tags")
    category: Optional[str] = Field(default=None, description="Category for the memory")
    timestamp: Optional[str] = Field(default=None, description="Creation time in YYYYMMDDHHmm format")


class MemoryUpdateRequest(BaseModel):
    """Request model for updating a memory."""
    content: Optional[str] = Field(default=None, description="Updated content")
    tags: Optional[List[str]] = Field(default=None, description="Updated tags")
    category: Optional[str] = Field(default=None, description="Updated category")
    context: Optional[str] = Field(default=None, description="Updated context")
    keywords: Optional[List[str]] = Field(default=None, description="Updated keywords")


class MemorySearchRequest(BaseModel):
    """Request model for searching memories."""
    query: str = Field(..., description="Search query text")
    k: int = Field(default=5, ge=1, le=100, description="Maximum number of results")


class MemoryResponse(BaseModel):
    """Response model for a single memory."""
    id: str
    content: str
    keywords: List[str]
    context: str
    category: str
    tags: List[str]
    timestamp: str
    last_accessed: str
    retrieval_count: int
    links: List[str]


class MemoryAddResponse(BaseModel):
    """Response model for adding a memory."""
    id: str
    message: str


class MemorySearchResult(BaseModel):
    """A single search result."""
    id: str
    content: str
    context: str
    keywords: List[str]
    tags: List[str]
    timestamp: str
    category: str
    score: Optional[float] = None
    is_neighbor: bool = False


class MemorySearchResponse(BaseModel):
    """Response model for search results."""
    results: List[MemorySearchResult]
    count: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    memory_count: int
    model_name: str
    llm_backend: str
    llm_model: str
    persistent: bool = False
    namespace: str = "default"


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None
