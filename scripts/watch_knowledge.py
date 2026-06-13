#!/usr/bin/env python3
"""
watch_knowledge.py - Auto-index new/modified knowledge files into A-MEM.

Uses macOS FSEvents via watchdog library to detect changes in the knowledge directory.
When a .md file is created or modified, it's automatically indexed into A-MEM.

Usage:
    python watch_knowledge.py              # Watch and auto-index
    python watch_knowledge.py --once       # Index all changed files since last run and exit

Requires: pip install watchdog requests
"""

import sys
import time
import hashlib
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Install watchdog: pip install watchdog")
    sys.exit(1)

AMEM_URL = "http://localhost:8101"
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "shared" / "orchestration" / "knowledge"
STATE_FILE = Path(__file__).resolve().parent / ".watch_state.json"


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def file_hash(filepath: str) -> str:
    content = Path(filepath).read_text(encoding="utf-8")
    return hashlib.md5(content.encode()).hexdigest()


def detect_departments(filepath: str) -> list[str]:
    """Detect department tags from file path."""
    path_lower = filepath.lower()
    dept_map = {
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
    for key, dept in dept_map.items():
        if f"/{key}/" in path_lower or f"/{key}\\" in path_lower:
            return [dept]
    return ["shared"]


def chunk_and_index(filepath: str) -> int:
    """Chunk a file and index into A-MEM. Returns number of chunks indexed."""
    path = Path(filepath)
    if not path.exists() or path.suffix != ".md":
        return 0

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return 0

    filename = path.stem
    departments = detect_departments(filepath)

    # Chunk by words with overlap
    words = content.split()
    chunk_size = 800
    overlap = 150
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    if not chunks:
        chunks = [content]

    indexed = 0
    for i, chunk in enumerate(chunks):
        tagged = (
            f"[Source: {filename}.md | Departments: {', '.join(departments)} "
            f"| Auto-indexed: {datetime.now().isoformat()}]\n\n{chunk}"
        )
        try:
            resp = requests.post(
                f"{AMEM_URL}/memory/add",
                json={
                    "content": tagged,
                    "tags": departments + [filename, "knowledge-base", "auto-indexed"],
                    "category": departments[0],
                },
                timeout=10,
            )
            if resp.status_code == 200:
                indexed += 1
            else:
                print(f"  Error indexing {filename} chunk {i}: HTTP {resp.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"  Error: Cannot connect to A-MEM at {AMEM_URL}")
            return indexed

    print(f"  Indexed {indexed} chunks from {filename}.md -> {departments}")
    return indexed


class KnowledgeHandler(FileSystemEventHandler):
    def __init__(self):
        self.state = load_state()

    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith(".md"):
            return
        self._process(event.src_path)

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".md"):
            return
        self._process(event.src_path)

    def _process(self, filepath):
        try:
            current_hash = file_hash(filepath)
        except (OSError, UnicodeDecodeError):
            return

        if self.state.get(filepath) == current_hash:
            return  # No actual content change

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Change detected: {Path(filepath).name}")
        chunk_and_index(filepath)

        self.state[filepath] = current_hash
        save_state(self.state)


def index_changed_files():
    """Index all files that changed since last run (--once mode)."""
    state = load_state()
    total_indexed = 0
    total_files = 0

    md_files = sorted(KNOWLEDGE_DIR.rglob("*.md"))
    print(f"Scanning {len(md_files)} markdown files...")

    for filepath in md_files:
        filepath_str = str(filepath)
        try:
            current_hash = file_hash(filepath_str)
        except (OSError, UnicodeDecodeError):
            continue

        if state.get(filepath_str) == current_hash:
            continue  # No change

        total_files += 1
        print(f"\nChanged: {filepath.name}")
        indexed = chunk_and_index(filepath_str)
        total_indexed += indexed
        state[filepath_str] = current_hash

    save_state(state)
    print(f"\nDone: {total_files} files changed, {total_indexed} chunks indexed")
    return total_indexed


def main():
    parser = argparse.ArgumentParser(description="Auto-index knowledge files into A-MEM")
    parser.add_argument("--once", action="store_true", help="Index changed files and exit")
    args = parser.parse_args()

    if not KNOWLEDGE_DIR.exists():
        print(f"Knowledge directory not found: {KNOWLEDGE_DIR}")
        sys.exit(1)

    # Verify A-MEM is running
    try:
        health = requests.get(f"{AMEM_URL}/health", timeout=5).json()
        print(f"A-MEM status: {health.get('status', 'unknown')}, memories: {health.get('memory_count', '?')}")
    except Exception:
        print(f"Error: A-MEM is not running at {AMEM_URL}. Start it first.")
        sys.exit(1)

    if args.once:
        index_changed_files()
        return

    print(f"Watching: {KNOWLEDGE_DIR}")
    print(f"A-MEM: {AMEM_URL}")
    print("Press Ctrl+C to stop\n")

    handler = KnowledgeHandler()
    observer = Observer()
    observer.schedule(handler, str(KNOWLEDGE_DIR), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
