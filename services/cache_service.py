import time
import json
import os
from typing import Any, Optional
try:
    import redis
except ImportError:
    redis = None

class CacheService:
    """
    Production-ready Cache Service.
    Uses Redis as primary storage with automatic in-memory fallback if Redis is unavailable.
    """
    def __init__(self):
        self._memory_cache = {}
        self.redis_client = None
        
        # Try to initialize Redis
        if redis:
            try:
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                # Test connection
                self.redis_client.ping()
                print("✅ Redis connection established successfully.")
            except Exception as e:
                print(f"⚠️ Redis connection failed: {e}. Falling back to in-memory cache.")
                self.redis_client = None

    def get(self, key: str) -> Optional[Any]:
        # 1. Try Redis
        if self.redis_client:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"❌ Redis GET error for {key}: {e}")
        
        # 2. Fallback to In-Memory
        if key in self._memory_cache:
            data, expiry = self._memory_cache[key]
            if expiry > time.time():
                return data
            else:
                del self._memory_cache[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        # 1. Try Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    ttl_seconds,
                    json.dumps(value)
                )
                return
            except Exception as e:
                print(f"❌ Redis SET error for {key}: {e}")
        
        # 2. Fallback to In-Memory
        expiry = time.time() + ttl_seconds
        self._memory_cache[key] = (value, expiry)

    def delete(self, key: str):
        # 1. Try Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                print(f"❌ Redis DELETE error for {key}: {e}")
        
        # 2. Fallback to In-Memory
        if key in self._memory_cache:
            del self._memory_cache[key]

# Global singleton
cache_service = CacheService()
