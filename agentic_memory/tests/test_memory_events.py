"""Tests for A-MEM pub/sub events on memory changes."""
import pytest
import sqlite3
from agentic_memory.versioned_store import VersionedStore
from agentic_memory.memory_broker import MemoryBroker, MemoryEventType

class TestMemoryBroker:
    def setup_method(self):
        self.broker = MemoryBroker()

    def test_subscriber_receives_created_event(self):
        received = []
        self.broker.subscribe(lambda e: received.append(e))
        self.broker.publish(MemoryEventType.CREATED, {'path': 'foo.md', 'version': 'initial'})
        assert len(received) == 1
        assert received[0]['type'] == MemoryEventType.CREATED

    def test_slow_subscriber_does_not_block_publish(self):
        def bad_subscriber(e):
            raise RuntimeError("subscriber crashed")
        self.broker.subscribe(bad_subscriber)
        # Must not raise
        self.broker.publish(MemoryEventType.CREATED, {'path': 'foo.md'})

    def test_unsubscribe_stops_events(self):
        received = []
        unsub = self.broker.subscribe(lambda e: received.append(e))
        self.broker.publish(MemoryEventType.CREATED, {'path': 'a'})
        unsub()
        self.broker.publish(MemoryEventType.CREATED, {'path': 'b'})
        assert len(received) == 1

    def test_versioned_store_fires_event_on_create(self):
        db = sqlite3.connect(':memory:')
        received = []
        store = VersionedStore(db, on_create=lambda e: received.append(e))
        store.ensure_tables()
        store.create_version('s1', 'memory/test.md', 'hello world')
        assert len(received) == 1
        assert received[0]['version'] == 'initial'
        assert received[0]['path'] == 'memory/test.md'
