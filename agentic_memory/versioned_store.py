"""
versioned_store.py — File versioning with retry loop (OpenCode history pattern).

Uses (path, session_id, version) UNIQUE constraint + 3-attempt retry loop
to handle concurrent version conflicts safely.
"""
import sqlite3
import hashlib
import time
import uuid
from typing import Optional


class VersionConflictError(Exception):
    pass


def _next_version(current: Optional[str]) -> str:
    if current is None:
        return 'initial'
    if current == 'initial':
        return 'v1'
    if current.startswith('v'):
        try:
            return f'v{int(current[1:]) + 1}'
        except ValueError:
            pass
    return f'v{time.time_ns()}'


class VersionedStore:
    MAX_RETRIES = 3

    def __init__(self, db: sqlite3.Connection, on_create=None):
        self.db = db
        self.db.row_factory = sqlite3.Row
        self._on_create = on_create

    def ensure_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS versioned_files (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                path TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                version TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(path, session_id, version)
            );
            CREATE INDEX IF NOT EXISTS idx_vf_path_created
            ON versioned_files(path, created_at DESC);
        """)
        self.db.commit()

    def get_latest_version(self, path: str) -> Optional[str]:
        row = self.db.execute("""
            SELECT version FROM versioned_files WHERE path = ?
            ORDER BY created_at DESC LIMIT 1
        """, (path,)).fetchone()
        return row['version'] if row else None

    def create_version(self, session_id: str, path: str, content: str) -> dict:
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        current_version = self.get_latest_version(path)
        next_ver = _next_version(current_version)

        for attempt in range(self.MAX_RETRIES):
            try:
                file_id = str(uuid.uuid4())
                created_at = time.time_ns()
                self.db.execute("""
                    INSERT INTO versioned_files
                    (id, session_id, path, content, content_hash, version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (file_id, session_id, path, content, content_hash, next_ver, created_at))
                self.db.commit()

                result = {
                    'id': file_id, 'session_id': session_id, 'path': path,
                    'version': next_ver, 'content_hash': content_hash, 'created_at': created_at
                }
                if self._on_create:
                    self._on_create(result)
                return result

            except sqlite3.IntegrityError as e:
                if 'UNIQUE constraint' in str(e) and attempt < self.MAX_RETRIES - 1:
                    current_version = self.get_latest_version(path)
                    next_ver = _next_version(current_version)
                    continue
                raise VersionConflictError(f"Version conflict for {path} after {self.MAX_RETRIES} retries") from e

        raise VersionConflictError(f"Exhausted retries for {path}")

    def get_history(self, path: str, limit: int = 20) -> list:
        rows = self.db.execute("""
            SELECT id, session_id, path, version, content_hash, created_at
            FROM versioned_files WHERE path = ?
            ORDER BY created_at DESC LIMIT ?
        """, (path, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_version(self, path: str, version: str) -> Optional[dict]:
        row = self.db.execute("""
            SELECT * FROM versioned_files WHERE path = ? AND version = ?
        """, (path, version)).fetchone()
        return dict(row) if row else None

    def get_snapshot(self, session_id: str) -> list:
        rows = self.db.execute("""
            SELECT vf.* FROM versioned_files vf
            INNER JOIN (
                SELECT path, MAX(created_at) as max_at
                FROM versioned_files WHERE session_id = ?
                GROUP BY path
            ) latest ON vf.path = latest.path AND vf.created_at = latest.max_at
            WHERE vf.session_id = ?
        """, (session_id, session_id)).fetchall()
        return [dict(r) for r in rows]
