"""
Cache Manager - TTL-Based File Cache

Caches LLM responses to disk for reuse across sessions.

Key Learning Concepts:
- File-based caching (persists across restarts)
- TTL (Time-To-Live) expiration
- Hash-based cache keys
- Disk I/O optimization

Why This Matters:
- Saves API costs (don't call twice for same prompt)
- Faster development (instant responses)
- Essential for production apps
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Any, Dict


class CacheManager:
    """
    File-based cache with TTL (Time-To-Live).

    Caches data to disk, persists across app restarts.
    Automatically expires old entries.

    Example:
        cache = CacheManager(cache_dir="data/cache", ttl_seconds=86400)

        # Store
        cache.set("my_key", {"result": "some data"})

        # Retrieve
        data = cache.get("my_key")
        if data:
            print("Cache hit!")
        else:
            print("Cache miss - call API")
    """

    def __init__(self, cache_dir: str = "data/cache", ttl_seconds: int = 86400):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache files
            ttl_seconds: Time-to-live in seconds (default: 24 hours)
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        self.hits = 0
        self.misses = 0

    def _get_cache_path(self, key: str) -> Path:
        """
        Get file path for cache key.

        Uses hash to create filename (safe for any key content).

        Args:
            key: Cache key

        Returns:
            Path to cache file
        """
        # Hash the key to create a safe filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def set(self, key: str, value: Any, ttl_override: Optional[int] = None) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl_override: Override default TTL for this entry
        """
        cache_path = self._get_cache_path(key)

        ttl = ttl_override if ttl_override is not None else self.ttl_seconds

        cache_entry = {
            "key": key,  # Store original key for debugging
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl,
            "expires_at": time.time() + ttl
        }

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=2)
        except Exception as e:
            # Fail silently - caching is optional
            pass

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.

        Returns None if:
        - Key doesn't exist
        - Entry has expired
        - Error reading cache

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            self.misses += 1
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_entry = json.load(f)

            # Check if expired
            if time.time() > cache_entry["expires_at"]:
                # Expired - delete file
                cache_path.unlink()
                self.misses += 1
                return None

            # Cache hit!
            self.hits += 1
            return cache_entry["value"]

        except Exception as e:
            # Error reading cache - treat as miss
            self.misses += 1
            return None

    def delete(self, key: str) -> bool:
        """
        Delete cache entry.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            cache_path.unlink()
            return True

        return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        return count

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        count = 0
        current_time = time.time()

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_entry = json.load(f)

                if current_time > cache_entry["expires_at"]:
                    cache_file.unlink()
                    count += 1

            except Exception:
                # Invalid cache file - delete it
                cache_file.unlink()
                count += 1

        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_entries = len(list(self.cache_dir.glob("*.json")))
        total_size_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))

        hit_rate = (self.hits / (self.hits + self.misses) * 100) if (self.hits + self.misses) > 0 else 0.0

        return {
            "total_entries": total_entries,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_dir": str(self.cache_dir),
            "ttl_seconds": self.ttl_seconds,
        }


class NoOpCache:
    """
    No-op cache for testing/debugging.

    Always returns None (cache miss).
    Useful for disabling cache temporarily.
    """

    def set(self, key: str, value: Any, ttl_override: Optional[int] = None) -> None:
        """Do nothing."""
        pass

    def get(self, key: str) -> Optional[Any]:
        """Always return None (cache miss)."""
        return None

    def delete(self, key: str) -> bool:
        """Always return False."""
        return False

    def clear(self) -> int:
        """Always return 0."""
        return 0

    def cleanup_expired(self) -> int:
        """Always return 0."""
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Return empty stats."""
        return {
            "total_entries": 0,
            "total_size_mb": 0.0,
            "hits": 0,
            "misses": 0,
            "hit_rate_percent": 0.0,
            "cache_dir": "disabled",
            "ttl_seconds": 0,
        }
