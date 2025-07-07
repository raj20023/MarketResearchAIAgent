"""
Fixed HTTP client with proper connection management and error handling.
"""
import asyncio
import logging
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta
import aiohttp
import time
from dataclasses import dataclass, field
import random
import ssl

from config import config

logger = logging.getLogger(__name__)

@dataclass
class RateLimiter:
    """Simple rate limiter implementation."""
    max_requests: int
    time_window: int  # seconds
    requests: list = field(default_factory=list)
    
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        now = time.time()
        
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        # Check if we can make a request
        if len(self.requests) >= self.max_requests:
            # Calculate how long to wait
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request) + 0.1  # Small buffer
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                return await self.acquire()  # Recursive call after waiting
        
        # Record this request
        self.requests.append(now)

class HTTPClient:
    """Fixed HTTP client with proper connection management."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiters: Dict[str, RateLimiter] = {
            'default': RateLimiter(max_requests=10, time_window=60),  # Conservative default
            'google': RateLimiter(max_requests=20, time_window=60),   # Google searches
            'news_api': RateLimiter(max_requests=100, time_window=3600),  # NewsAPI
            'serpapi': RateLimiter(max_requests=100, time_window=3600),   # SerpAPI
        }
        self._session_lock = asyncio.Lock()
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def start(self) -> None:
        """Initialize the HTTP session with proper configuration."""
        async with self._session_lock:
            if self.session is None or self.session.closed:
                # Create SSL context
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                # Configure timeouts
                timeout = aiohttp.ClientTimeout(
                    total=config.scraping.timeout,
                    connect=10,
                    sock_read=config.scraping.timeout
                )
                
                # Configure connector
                connector = aiohttp.TCPConnector(
                    limit=50,  # Reduced from 100
                    limit_per_host=10,  # Reduced from 30
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    ssl=ssl_context,
                    enable_cleanup_closed=True,
                    keepalive_timeout=30,
                    force_close=True  # Force close connections to avoid hanging
                )
                
                self.session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                    headers=self._get_base_headers(),
                    raise_for_status=False  # Handle status codes manually
                )
                
                logger.debug("HTTP session initialized")
    
    async def close(self) -> None:
        """Close the HTTP session properly."""
        async with self._session_lock:
            if self.session and not self.session.closed:
                await self.session.close()
                # Wait for connections to close
                await asyncio.sleep(0.1)
                self.session = None
                logger.debug("HTTP session closed")
    
    def _get_base_headers(self) -> Dict[str, str]:
        """Get base headers for requests."""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        return {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def _determine_rate_limiter(self, url: str) -> RateLimiter:
        """Determine which rate limiter to use based on the URL."""
        if 'google.com' in url:
            return self.rate_limiters['google']
        elif 'newsapi.org' in url:
            return self.rate_limiters['news_api']
        elif 'serpapi.com' in url:
            return self.rate_limiters['serpapi']
        else:
            return self.rate_limiters['default']
    
    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None
    ) -> Optional[aiohttp.ClientResponse]:
        """Make a GET request with retry logic and proper error handling."""
        if not self.session or self.session.closed:
            await self.start()
        
        max_retries = max_retries or config.scraping.max_retries
        timeout = timeout or config.scraping.timeout
        
        # Apply rate limiting
        rate_limiter = self._determine_rate_limiter(url)
        await rate_limiter.acquire()
        
        # Merge headers
        request_headers = self._get_base_headers()
        if headers:
            request_headers.update(headers)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Create timeout for this specific request
                request_timeout = aiohttp.ClientTimeout(total=timeout)
                
                async with self.session.get(
                    url,
                    params=params,
                    headers=request_headers,
                    timeout=request_timeout,
                    allow_redirects=True,
                    max_redirects=3
                ) as response:
                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited, waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    # Handle server errors
                    if response.status >= 500:
                        logger.warning(f"Server error {response.status} for {url}, attempt {attempt + 1}")
                        if attempt < max_retries:
                            await asyncio.sleep(config.scraping.retry_delay * (2 ** attempt))
                            continue
                    
                    # For client errors (4xx), don't retry
                    if 400 <= response.status < 500 and response.status != 429:
                        logger.warning(f"Client error {response.status} for {url}")
                        return None
                    
                    # Success or handled error
                    if response.status < 400 or response.status == 429:
                        # Read the response content while connection is still open
                        try:
                            content = await response.read()
                            
                            # Create a new response object with the content
                            class ResponseWrapper:
                                def __init__(self, original_response, content):
                                    self.status = original_response.status
                                    self.headers = original_response.headers
                                    self.url = original_response.url
                                    self._content = content
                                    self._text = None
                                    self._json = None
                                
                                async def text(self, encoding='utf-8', errors='strict'):
                                    if self._text is None:
                                        self._text = self._content.decode(encoding, errors)
                                    return self._text
                                
                                async def json(self, **kwargs):
                                    if self._json is None:
                                        import json
                                        text_content = await self.text()
                                        self._json = json.loads(text_content, **kwargs)
                                    return self._json
                                
                                def raise_for_status(self):
                                    if self.status >= 400:
                                        raise aiohttp.ClientResponseError(
                                            request_info=None,
                                            history=(),
                                            status=self.status,
                                            message=f"HTTP {self.status}"
                                        )
                            
                            wrapped_response = ResponseWrapper(response, content)
                            
                            # Check for successful status
                            if response.status < 400:
                                return wrapped_response
                            else:
                                logger.warning(f"HTTP {response.status} for {url}")
                                return None
                                
                        except Exception as e:
                            logger.error(f"Error reading response content from {url}: {e}")
                            last_exception = e
                            if attempt < max_retries:
                                await asyncio.sleep(config.scraping.retry_delay * (2 ** attempt))
                                continue
                            return None
                    
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last_exception = e
                logger.warning(f"Request failed for {url}, attempt {attempt + 1}: {e}")
                
                # For connection errors, recreate session
                if isinstance(e, (aiohttp.ClientConnectionError, OSError)):
                    logger.debug("Recreating session due to connection error")
                    await self.close()
                    await self.start()
                
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    wait_time = config.scraping.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                    continue
            
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error for {url}: {e}")
                break
        
        logger.error(f"All retry attempts failed for {url}: {last_exception}")
        return None
    
    async def get_text(
        self,
        url: str,
        **kwargs
    ) -> Optional[str]:
        """Get response text from a URL."""
        try:
            response = await self.get(url, **kwargs)
            if response:
                return await response.text()
        except Exception as e:
            logger.error(f"Error getting text from {url}: {e}")
        return None
    
    async def get_json(
        self,
        url: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Get JSON response from a URL."""
        try:
            response = await self.get(url, **kwargs)
            if response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error getting JSON from {url}: {e}")
        return None

# Global HTTP client instance with singleton pattern
_http_client_instance = None
_client_lock = asyncio.Lock()

async def get_http_client() -> HTTPClient:
    """Get singleton HTTP client instance."""
    global _http_client_instance
    
    async with _client_lock:
        if _http_client_instance is None:
            _http_client_instance = HTTPClient()
            await _http_client_instance.start()
    
    return _http_client_instance

async def close_http_client():
    """Close the global HTTP client."""
    global _http_client_instance
    
    async with _client_lock:
        if _http_client_instance:
            await _http_client_instance.close()
            _http_client_instance = None

# For backward compatibility
http_client = HTTPClient()