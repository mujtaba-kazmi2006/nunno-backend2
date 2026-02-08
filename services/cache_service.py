import time
import json
from typing import Any, Optional

class CacheService:
    """
    A simple in-memory cache service with TTL.
    In production, this should be replaced with Redis.
    """
    def __init__(self):
        self._cache = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            data, expiry = self._cache[key]
            if expiry > time.time():
                return data
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        expiry = time.time() + ttl_seconds
        self._cache[key] = (value, expiry)

    def delete(self, key: str):
        if key in self._cache:
            del self._cache[key]

# Global singleton
cache_service = CacheService()
