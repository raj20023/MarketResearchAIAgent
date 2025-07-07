"""
Simple cache manager compatible with Streamlit and asyncio.
"""
import time
import json
import hashlib
import logging
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from functools import wraps
import threading

logger = logging.getLogger(__name__)

class SimpleCacheManager:
    """Thread-safe memory cache manager for Streamlit compatibility."""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, datetime] = {}
        self.lock = threading.RLock()  # Use threading lock instead of asyncio
        self.default_ttl = 3600  # 1 hour
        self.max_items = 1000
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key from function name and arguments."""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return f"mr_cache:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                timestamp = self.timestamps.get(key)
                if timestamp and datetime.now() - timestamp < timedelta(seconds=self.default_ttl):
                    logger.debug(f"Cache hit for key: {key[:20]}...")
                    return self.cache[key]
                else:
                    # Expired
                    self._remove_item(key)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self.lock:
            # Clean up if cache is too large
            if len(self.cache) >= self.max_items:
                self._cleanup_old_items()
            
            self.cache[key] = value
            self.timestamps[key] = datetime.now()
            logger.debug(f"Cached item with key: {key[:20]}...")
    
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        with self.lock:
            self._remove_item(key)
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_items = len(self.cache)
            expired_items = 0
            now = datetime.now()
            
            for key, timestamp in self.timestamps.items():
                if now - timestamp >= timedelta(seconds=self.default_ttl):
                    expired_items += 1
            
            return {
                'total_items': total_items,
                'expired_items': expired_items,
                'active_items': total_items - expired_items,
                'max_items': self.max_items,
                'default_ttl': self.default_ttl
            }
    
    def _remove_item(self, key: str) -> None:
        """Remove item from cache (internal method)."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def _cleanup_old_items(self) -> None:
        """Remove expired items from cache."""
        now = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.timestamps.items():
            if now - timestamp >= timedelta(seconds=self.default_ttl):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_item(key)
        
        # If still too many items, remove oldest
        if len(self.cache) >= self.max_items:
            oldest_keys = sorted(
                self.timestamps.keys(),
                key=lambda k: self.timestamps[k]
            )[:100]  # Remove oldest 100 items
            
            for key in oldest_keys:
                self._remove_item(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache items")

# Global cache instance
simple_cache = SimpleCacheManager()

def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """Simple caching decorator that works with Streamlit."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = simple_cache._generate_key(
                f"{key_prefix}{func.__name__}", args, kwargs
            )
            
            # Try to get from cache
            cached_result = simple_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache the result
            simple_cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# For backward compatibility with existing code
class CacheManagerCompat:
    """Compatibility wrapper for the original cache manager."""
    
    def __init__(self):
        self.cache = simple_cache
    
    async def initialize(self):
        """Initialize - no-op for compatibility."""
        pass
    
    async def get(self, key: str):
        """Async get wrapper."""
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Async set wrapper."""
        return self.cache.set(key, value, ttl)
    
    async def delete(self, key: str):
        """Async delete wrapper."""
        return self.cache.delete(key)
    
    async def clear_all(self):
        """Async clear wrapper."""
        return self.cache.clear()

# Create compatibility instance
cache_manager = CacheManagerCompat()