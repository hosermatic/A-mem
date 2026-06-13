#!/usr/bin/env python3
"""
index_knowledge.py - Index all knowledge markdown files into A-MEM for BuckBrain RAG.

Scans shared/orchestration/knowledge/ and indexes by department.
Also indexes BuckBrain training data from projects/buckbrain/data/chat_format/.

Usage:
    python index_knowledge.py                    # Index everything
    python index_knowledge.py --dry-run          # Show what would be indexed
    python index_knowledge.py --department post   # Index only post-production
    python index_knowledge.py --skip-training     # Skip BuckBrain JSONL data
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path

AMEM_URL = "http://localhost:8101"

# Department detection from directory path
DEPT_DIR_MAP = {
    "2d": "design",
    "3d": "3d",
    "arg": "arg",
    "copy": "shared",
    "design": "design",
    "post": "post-production",
    "production": "production",
    "resourcing": "resourcing",
    "strategy": "strategy",
}

BUCKBRAIN_JSONL = "projects/buckbrain/data/chat_format/train.jsonl"


def chunk_markdown(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split markdown into overlapping word-based chunks for better retrieval."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def detect_department(filepath: str) -> str:
    """Detect department tag from file path."""
    parts = Path(filepath).parts
    for part in parts:
        part_lower = part.lower()
        if part_lower in DEPT_DIR_MAP:
            return DEPT_DIR_MAP[part_lower]
    return "shared"


def index_markdown_file(filepath: str, base_dir: str, dry_run: bool = False) -> int:
    """Index a single markdown file into A-MEM. Returns number of chunks indexed."""
    path = Path(filepath)
    content = path.read_text(encoding="utf-8")

    if len(content.strip()) < 50:
        return 0

    filename = path.stem
    relative_path = str(path.relative_to(base_dir))
    department = detect_department(relative_path)

    chunks = chunk_markdown(content)
    indexed = 0

    for i, chunk in enumerate(chunks):
        tagged_content = f"[Source: {filename}.md | Department: {department}]\n\n{chunk}"

        if dry_run:
            if i == 0:
                print(f"  [DRY RUN] {filename} -> {department} ({len(chunks)} chunks)")
            indexed += 1
            continue

        try:
            response = requests.post(f"{AMEM_URL}/memory/add", json={
                "content": tagged_content,
                "tags": [department, "shared", filename, "knowledge-base"],
                "category": department,
                "metadata": {
                    "source_file": relative_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "department": department,
                    "data_lineage": "knowledge-file",
                },
            }, timeout=30)

            if response.status_code == 200:
                indexed += 1
            else:
                print(f"  ERROR indexing {filename} chunk {i}: HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"  ERROR: Cannot connect to A-MEM at {AMEM_URL}")
            sys.exit(1)

    if not dry_run and indexed > 0:
        print(f"  Indexed {indexed}/{len(chunks)} chunks from {filename} -> {department}")

    return indexed


def index_buckbrain_jsonl(jsonl_path: str, dry_run: bool = False) -> int:
    """Index BuckBrain training Q&A pairs into A-MEM."""
    if not os.path.exists(jsonl_path):
        print(f"  BuckBrain JSONL not found at {jsonl_path}, skipping")
        return 0

    with open(jsonl_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"  Found {len(samples)} training samples")
    indexed = 0

    for i, sample in enumerate(samples):
        messages = sample.get("messages", [])
        question = ""
        answer = ""

        for msg in messages:
            if msg["role"] == "user":
                question = msg["content"]
            elif msg["role"] == "assistant":
                answer = msg["content"]

        if not answer or len(answer.strip()) < 20:
            continue

        content = f"[Source: buckbrain-training | Type: Q&A]\n\nQ: {question}\n\nA: {answer}"

        if dry_run:
            if i % 500 == 0:
                print(f"  [DRY RUN] Would index sample {i}/{len(samples)}")
            indexed += 1
            continue

        try:
            response = requests.post(f"{AMEM_URL}/memory/add", json={
                "content": content,
                "tags": ["post-production", "shared", "buckbrain", "training-data"],
                "category": "post-production",
                "metadata": {
                    "source_file": "buckbrain-training-data",
                    "sample_index": i,
                    "data_lineage": "training-data",
                },
            }, timeout=30)

            if response.status_code == 200:
                indexed += 1

            if i % 200 == 0:
                print(f"  Indexed {i}/{len(samples)} training samples...")
        except requests.exceptions.ConnectionError:
            print(f"  ERROR: Cannot connect to A-MEM at {AMEM_URL}")
            sys.exit(1)

    return indexed


def main():
    parser = argparse.ArgumentParser(description="Index knowledge files into A-MEM")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be indexed")
    parser.add_argument("--department", type=str, help="Only index specific department")
    parser.add_argument("--skip-training", action="store_true", help="Skip BuckBrain training data")
    args = parser.parse_args()

    # Find base directory (Developer/)
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent.parent.parent  # scripts -> a-mem -> llm-infrastructure -> shared -> Developer/

    # Verify A-MEM is running
    if not args.dry_run:
        try:
            health = requests.get(f"{AMEM_URL}/health", timeout=5).json()
            print(f"A-MEM status: {health.get('status', 'unknown')}, current memories: {health.get('memory_count', '?')}")
        except Exception:
            print("ERROR: A-MEM is not running at", AMEM_URL)
            sys.exit(1)

    # Index knowledge markdown files
    knowledge_base = base_dir / "shared" / "orchestration" / "knowledge"
    total_files = 0
    total_chunks = 0

    if knowledge_base.exists():
        md_files = sorted(knowledge_base.rglob("*.md"))
        # Skip meta files
        md_files = [f for f in md_files if f.name not in ("README.md", "DEDUP-REPORT.md", ".dedup-summary.txt")]
        print(f"\nFound {len(md_files)} knowledge files")

        for filepath in md_files:
            department = detect_department(str(filepath.relative_to(base_dir)))

            if args.department and args.department != department:
                continue

            chunks = index_markdown_file(str(filepath), str(base_dir), dry_run=args.dry_run)
            if chunks > 0:
                total_files += 1
                total_chunks += chunks

    # Index BuckBrain training data
    if not args.skip_training:
        jsonl_path = base_dir / BUCKBRAIN_JSONL
        print(f"\nIndexing BuckBrain training data...")
        training_chunks = index_buckbrain_jsonl(str(jsonl_path), dry_run=args.dry_run)
        total_chunks += training_chunks
        print(f"  Training samples indexed: {training_chunks}")

    # Summary
    mode = " (DRY RUN)" if args.dry_run else ""
    print(f"\n{'='*50}")
    print(f"INDEXING COMPLETE{mode}")
    print(f"  Knowledge files processed: {total_files}")
    print(f"  Total chunks indexed: {total_chunks}")

    if not args.dry_run:
        try:
            health = requests.get(f"{AMEM_URL}/health", timeout=5).json()
            print(f"  A-MEM total memories: {health.get('memory_count', '?')}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
