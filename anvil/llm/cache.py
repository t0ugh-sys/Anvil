from __future__ import annotations

import hashlib
import threading
import time
from typing import Dict


class PromptCache:
    """LRU-style cache for LLM prompt responses.

    Caches responses keyed by (model, prompt_hash) to avoid redundant
    API calls for identical prompts. Useful for repeated system prompts,
    template rendering, and test/development scenarios.

    Usage::

        cache = PromptCache(max_size=100)
        key = cache.make_key('claude-3', 'Summarize this: ...')
        cached = cache.get(key)
        if cached is None:
            result = call_llm(prompt)
            cache.set(key, result)
        else:
            result = cached
    """

    def __init__(self, max_size: int = 128, ttl_seconds: float = 3600.0) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple] = {}  # key -> (result, timestamp)
        self._access_order: list[str] = []  # LRU tracking
        self._lock = threading.Lock()

    @staticmethod
    def make_key(model: str, prompt: str) -> str:
        """Create a cache key from model name and prompt text."""
        content = f'{model}:{prompt}'
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def get(self, key: str) -> str | None:
        """Get cached result if available and not expired."""
        with self._lock:
            if key not in self._cache:
                return None

            result, timestamp = self._cache[key]
            # Check TTL
            import time as _t
            if _t.time() - timestamp > self._ttl_seconds:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None

            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return result

    def set(self, key: str, result: str) -> None:
        """Store a result in the cache."""
        import time as _t
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_size and self._access_order:
                oldest = self._access_order.pop(0)
                self._cache.pop(oldest, None)

            self._cache[key] = (result, _t.time())
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'ttl_seconds': self._ttl_seconds,
        }
