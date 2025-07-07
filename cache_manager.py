"""
Unified cache system that works with both Streamlit UI and backend application.
Replaces the original cache_manager.py to avoid asyncio loop conflicts.
"""
import json
import hashlib
import logging
import time
import threading
from typing import Any, Optional, Callable, Union, Dict
from functools import wraps
from datetime import datetime, timedelta
import asyncio
import concurrent.futures

logger = logging.getLogger(__name__)

class UnifiedCacheManager:
    """
    Thread-safe cache manager that works with both sync and async operations.
    Avoids asyncio event loop conflicts while maintaining full functionality.
    """
    
    def __init__(self):
        self.memory_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self._initialized = True  # Always initialized for simplicity
        self.default_ttl = 3600  # 1 hour
        self.max_items = 1000
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key from function name and arguments."""
        try:
            # Create a string representation of args and kwargs
            key_data = {
                'func': func_name,
                'args': [str(arg) for arg in args],  # Convert to strings to avoid serialization issues
                'kwargs': sorted([(str(k), str(v)) for k, v in kwargs.items()])
            }
            key_string = json.dumps(key_data, sort_keys=True, default=str)
            return f"market_research:{hashlib.md5(key_string.encode()).hexdigest()}"
        except Exception as e:
            # Fallback key generation
            logger.warning(f"Key generation error: {e}, using fallback")
            fallback_key = f"market_research:{hash(str(args) + str(kwargs)) % 1000000}"
            return fallback_key
    
    # Synchronous methods (primary interface)
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (synchronous)."""
        with self._lock:
            if key in self.memory_cache:
                timestamp = self.cache_timestamps.get(key)
                if timestamp and datetime.now() - timestamp < timedelta(seconds=self.default_ttl):
                    logger.debug(f"Cache hit for {key[:20]}...")
                    return self.memory_cache[key]
                else:
                    # Expired
                    self._remove_item(key)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache (synchronous)."""
        with self._lock:
            # Clean up if cache is too large
            if len(self.memory_cache) >= self.max_items:
                self._cleanup_old_items()
            
            self.memory_cache[key] = value
            self.cache_timestamps[key] = datetime.now()
            logger.debug(f"Cached item with key: {key[:20]}...")
    
    def delete(self, key: str) -> None:
        """Delete value from cache (synchronous)."""
        with self._lock:
            self._remove_item(key)
    
    def clear_all(self) -> None:
        """Clear all cached data (synchronous)."""
        with self._lock:
            self.memory_cache.clear()
            self.cache_timestamps.clear()
            logger.info("Cache cleared successfully")
    
    # Async wrapper methods (for backward compatibility)
    async def initialize(self) -> None:
        """Initialize the cache system (async compatibility)."""
        # Already initialized, this is just for compatibility
        pass
    
    async def get_async(self, key: str) -> Optional[Any]:
        """Async wrapper for get."""
        return self.get(key)
    
    async def set_async(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Async wrapper for set."""
        return self.set(key, value, ttl)
    
    async def delete_async(self, key: str) -> None:
        """Async wrapper for delete."""
        return self.delete(key)
    
    async def clear_all_async(self) -> None:
        """Async wrapper for clear_all."""
        return self.clear_all()
    
    # Internal methods
    def _remove_item(self, key: str) -> None:
        """Remove item from cache (internal)."""
        self.memory_cache.pop(key, None)
        self.cache_timestamps.pop(key, None)
    
    def _cleanup_old_items(self) -> None:
        """Remove expired items from cache."""
        now = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if now - timestamp >= timedelta(seconds=self.default_ttl):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_item(key)
        
        # If still too many items, remove oldest
        if len(self.memory_cache) >= self.max_items:
            oldest_keys = sorted(
                self.cache_timestamps.keys(),
                key=lambda k: self.cache_timestamps[k]
            )[:100]  # Remove oldest 100 items
            
            for key in oldest_keys:
                self._remove_item(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache items")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_items = len(self.memory_cache)
            expired_items = 0
            now = datetime.now()
            
            for key, timestamp in self.cache_timestamps.items():
                if now - timestamp >= timedelta(seconds=self.default_ttl):
                    expired_items += 1
            
            return {
                'total_items': total_items,
                'expired_items': expired_items,
                'active_items': total_items - expired_items,
                'max_items': self.max_items,
                'default_ttl': self.default_ttl
            }

# Global cache manager instance
cache_manager = UnifiedCacheManager()

def run_in_executor(func, *args, **kwargs):
    """
    Helper function to run sync functions in an executor to avoid blocking.
    """
    try:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return loop.run_in_executor(executor, func, *args, **kwargs)
    except RuntimeError:
        # No event loop running, just call the function directly
        return func(*args, **kwargs)

def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Unified caching decorator that works with both sync and async functions.
    Avoids asyncio loop conflicts by handling event loops properly.
    """
    def decorator(func: Callable) -> Callable:
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Synchronous wrapper for caching."""
            cache_key = cache_manager._generate_key(
                f"{key_prefix}{func.__name__}", args, kwargs
            )
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache the result
            cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Asynchronous wrapper for caching."""
            cache_key = cache_manager._generate_key(
                f"{key_prefix}{func.__name__}", args, kwargs
            )
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)  # Use sync method
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Run sync function in executor to avoid blocking
                result = await run_in_executor(func, *args, **kwargs)
            
            # Cache the result
            cache_manager.set(cache_key, result, ttl)  # Use sync method
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Helper function for running async operations safely
def safe_run_async(coro):
    """
    Safely run async operations, handling event loop conflicts.
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're already in an event loop, run in executor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No event loop running, we can run directly
        return asyncio.run(coro)
    except Exception as e:
        logger.error(f"Error running async operation: {e}")
        raise

# Complete backward compatibility - cache_manager matches original interface exactly
class AsyncCacheManagerWrapper:
    """
    Full compatibility wrapper that provides both sync and async interfaces.
    This ensures your existing code works exactly as before.
    """
    
    def __init__(self):
        self._unified_cache = cache_manager
        self._initialized = True
    
    # Async methods (for compatibility with original async code)
    async def initialize(self) -> None:
        """Initialize the cache system (async compatibility)."""
        self._initialized = True
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (async)."""
        return self._unified_cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache (async)."""
        return self._unified_cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> None:
        """Delete value from cache (async)."""
        return self._unified_cache.delete(key)
    
    async def clear_all(self) -> None:
        """Clear all cached data (async)."""
        return self._unified_cache.clear_all()
    
    # Sync methods (direct access to unified cache)
    def get_sync(self, key: str) -> Optional[Any]:
        """Get value from cache (sync)."""
        return self._unified_cache.get(key)
    
    def set_sync(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache (sync)."""
        return self._unified_cache.set(key, value, ttl)
    
    def delete_sync(self, key: str) -> None:
        """Delete value from cache (sync)."""
        return self._unified_cache.delete(key)
    
    def clear_all_sync(self) -> None:
        """Clear all cached data (sync)."""
        return self._unified_cache.clear_all()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._unified_cache.get_stats()
    
    # Internal methods for compatibility
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key (compatibility)."""
        return self._unified_cache._generate_key(func_name, args, kwargs)

# Replace the global cache_manager with the compatibility wrapper
# This ensures all existing code continues to work exactly as before
cache_manager = AsyncCacheManagerWrapper()