"""Tests for versioned memory storage (OpenCode history pattern)."""
import pytest
import sqlite3
from agentic_memory.versioned_store import VersionedStore, VersionConflictError

class TestVersionedStore:
    def setup_method(self):
        self.db = sqlite3.connect(':memory:')
        self.store = VersionedStore(self.db)
        self.store.ensure_tables()

    def test_first_write_creates_initial_version(self):
        result = self.store.create_version('session-1', 'memory/foo.md', 'content v1')
        assert result['version'] == 'initial'

    def test_second_write_creates_v1(self):
        self.store.create_version('session-1', 'memory/foo.md', 'content v1')
        result = self.store.create_version('session-1', 'memory/foo.md', 'content v2')
        assert result['version'] == 'v1'

    def test_sequential_versions_increment(self):
        self.store.create_version('s', 'p', 'v0')
        self.store.create_version('s', 'p', 'v1')
        result = self.store.create_version('s', 'p', 'v2')
        assert result['version'] == 'v2'

    def test_get_history_returns_newest_first(self):
        self.store.create_version('s', 'path/test.md', 'v1 content')
        self.store.create_version('s', 'path/test.md', 'v2 content')
        history = self.store.get_history('path/test.md')
        assert len(history) == 2
        assert history[0]['version'] == 'v1'  # newest first
