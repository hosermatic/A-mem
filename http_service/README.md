# Agentic Memory HTTP Service

REST API wrapper for the Agentic Memory system.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run the service
python run.py
```

The service will start on `http://localhost:8100` by default.

## API Endpoints

### Health Check

```
GET /health
```

Check service health and memory system status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-18T14:30:00.000Z",
  "memory_count": 42,
  "model_name": "all-MiniLM-L6-v2",
  "llm_backend": "openai"
}
```

### Add Memory

```
POST /memory/add
```

Add a new memory to the system.

**Request Body:**
```json
{
  "content": "Deep learning neural networks are powerful",
  "tags": ["ml", "ai"],
  "category": "Research",
  "timestamp": "202601181430"
}
```

**Response:**
```json
{
  "id": "mem_abc123",
  "status": "created"
}
```

### Get Memory

```
GET /memory/{memory_id}
```

Retrieve a memory by ID.

**Response:**
```json
{
  "id": "mem_abc123",
  "content": "Deep learning neural networks...",
  "tags": ["ml", "ai"],
  "context": "Generated context about the memory...",
  "keywords": ["deep learning", "neural networks"],
  "category": "Research"
}
```

### Update Memory

```
PUT /memory/{memory_id}
```

Update an existing memory.

**Request Body:**
```json
{
  "content": "Updated content about deep learning"
}
```

**Response:**
```json
{
  "id": "mem_abc123",
  "status": "updated"
}
```

### Delete Memory

```
DELETE /memory/{memory_id}
```

Delete a memory.

**Response:**
```json
{
  "id": "mem_abc123",
  "status": "deleted"
}
```

### Search Memories

```
POST /memory/search
```

Search memories using semantic search.

**Request Body:**
```json
{
  "query": "neural networks",
  "k": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "mem_abc123",
      "content": "Deep learning neural networks...",
      "tags": ["ml", "ai"]
    }
  ],
  "query": "neural networks",
  "count": 1
}
```

### List Memories

```
GET /memory/list?limit=100&offset=0
```

List all memory IDs with pagination.

**Response:**
```json
{
  "ids": ["mem_abc123", "mem_def456"],
  "total": 42,
  "limit": 100,
  "offset": 0
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8100 | HTTP server port |
| `HOST` | 0.0.0.0 | HTTP server host |
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `LLM_BACKEND` | openai | LLM backend (openai/ollama) |
| `LLM_MODEL` | gpt-4o-mini | LLM model name |

## OpenAPI Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8100/docs`
- ReDoc: `http://localhost:8100/redoc`

## Integration with Context Engine

This service provides the memory layer for the Claude AI Intelligence Layer.
The Context Engine can use these endpoints to store and retrieve session memories,
decision history, and preference learning.

```python
import httpx

# Add a memory
response = httpx.post("http://localhost:8100/memory/add", json={
    "content": "User prefers concise responses",
    "tags": ["preference", "style"],
    "category": "user_preference"
})

# Search relevant memories
response = httpx.post("http://localhost:8100/memory/search", json={
    "query": "response style preferences",
    "k": 5
})
```
