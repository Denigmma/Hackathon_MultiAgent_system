from __future__ import annotations

import threading
from queue import Queue
from typing import Any


class ChatEventHub:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: dict[str, list[Queue[dict[str, Any]]]] = {}

    def subscribe(self, chat_id: str) -> Queue[dict[str, Any]]:
        subscriber: Queue[dict[str, Any]] = Queue()
        with self._lock:
            self._subscribers.setdefault(chat_id, []).append(subscriber)
        return subscriber

    def unsubscribe(self, chat_id: str, subscriber: Queue[dict[str, Any]]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(chat_id)
            if not subscribers:
                return
            self._subscribers[chat_id] = [item for item in subscribers if item is not subscriber]
            if not self._subscribers[chat_id]:
                self._subscribers.pop(chat_id, None)

    def publish(self, chat_id: str, event: str, data: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(chat_id, []))

        for subscriber in subscribers:
            subscriber.put(
                {
                    "event": event,
                    "data": data,
                }
            )

