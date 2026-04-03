#!/usr/bin/env python3
"""
Run the Agentic Memory HTTP service.

Usage:
    python run.py
    # Or with environment variables:
    PORT=8100 LLM_BACKEND=openai python run.py
"""

import os
import sys

# Add parent directory to path for agentic_memory import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    """Run the HTTP service."""
    port = int(os.getenv("PORT", "8100"))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"

    print(f"Starting Agentic Memory HTTP service on {host}:{port}")
    print(f"OpenAPI docs: http://localhost:{port}/docs")
    print(f"Health check: http://localhost:{port}/health")

    uvicorn.run(
        "http_service.service:app",
        host=host,
        port=port,
        reload=reload,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
