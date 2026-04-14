"""
memory_broker.py — Pub/Sub event broker for A-MEM memory changes.

Pattern: OpenCode internal/pubsub/broker.go (Go → Python port).

Key design:
- Non-blocking: subscriber exceptions caught, never crash publisher
- Snapshot: subscriber list copied before iterating (safe against concurrent unsub)
- No external deps: plain Python callbacks
"""
import logging
import time
import uuid
from typing import Callable

logger = logging.getLogger(__name__)


class MemoryEventType:
    CREATED = 'created'
    UPDATED = 'updated'
    DELETED = 'deleted'


class MemoryBroker:
    """
    Pub/sub broker for memory change events.

    Usage:
        broker = MemoryBroker()
        unsub = broker.subscribe(lambda e: print(e))
        broker.publish(MemoryEventType.CREATED, {'path': 'foo.md', 'version': 'v1'})
        unsub()
    """

    def __init__(self):
        self._subscribers: dict = {}

    def subscribe(self, handler: Callable[[dict], None]) -> Callable:
        """Register subscriber. Returns unsubscribe function."""
        sub_id = str(uuid.uuid4())
        self._subscribers[sub_id] = handler

        def unsubscribe():
            self._subscribers.pop(sub_id, None)

        return unsubscribe

    def publish(self, event_type: str, payload: dict) -> int:
        """
        Publish event to all subscribers.
        Non-blocking: subscriber exceptions are caught, not propagated.
        Returns count of successfully notified subscribers.
        """
        # Snapshot to avoid mutation during iteration (OpenCode pattern)
        snapshot = list(self._subscribers.values())
        event = {'type': event_type, 'payload': payload, 'at': time.time()}
        notified = 0

        for handler in snapshot:
            try:
                handler(event)
                notified += 1
            except Exception as e:
                logger.warning(f"MemoryBroker: subscriber raised {e!r}, continuing")

        return notified


# Global default broker
_default_broker = MemoryBroker()


def get_default_broker() -> MemoryBroker:
    return _default_broker


def subscribe(handler: Callable) -> Callable:
    """Subscribe to the global default broker."""
    return _default_broker.subscribe(handler)


def publish(event_type: str, payload: dict) -> int:
    """Publish to the global default broker."""
    return _default_broker.publish(event_type, payload)
