import threading
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """LRU + TTL بسيط. بيوفر حصة Groq لما نفس السؤال يتكرر."""

    def __init__(self, maxsize: int = 256, ttl: int = 3600):
        self.maxsize, self.ttl = maxsize, ttl
        self._d: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            ts, value = item
            if time.time() - ts > self.ttl:
                del self._d[key]
                return None
            self._d.move_to_end(key)
            return value

    def set(self, key, value):
        if self.maxsize <= 0:
            return
        with self._lock:
            self._d[key] = (time.time(), value)
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)
