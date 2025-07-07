"""
Centralized configuration management for the market research system.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class APIConfig:
    """Configuration for external APIs."""
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    news_api_key: Optional[str] = field(default_factory=lambda: os.getenv("NEWS_API_KEY"))
    google_maps_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_MAPS_API_KEY"))
    serp_api_key: Optional[str] = field(default_factory=lambda: os.getenv("SERP_API_KEY"))
    redis_url: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))

@dataclass
class ModelConfig:
    """Configuration for AI models."""
    model_name: str = "gpt-4o"
    temperature: float = 0.2
    analytical_temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: int = 30

@dataclass
class ScrapingConfig:
    """Configuration for web scraping."""
    timeout: int = 30  # Increased timeout
    max_retries: int = 3
    retry_delay: float = 2.0  # Increased delay
    use_simple_client: bool = True  # Use simple synchronous client
    enable_web_scraping: bool = True  # Can be disabled to avoid issues
    min_request_delay: float = 1.0  # Minimum delay between requests
    user_agents: list = field(default_factory=lambda: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ])

@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour
    max_memory: str = "100mb"

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_path: Optional[str] = None

@dataclass
class Config:
    """Main configuration class."""
    api: APIConfig = field(default_factory=APIConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    def get_headers(self) -> Dict[str, str]:
        """Get randomized headers for web requests."""
        import random
        return {
            'User-Agent': random.choice(self.scraping.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

# Global configuration instance
config = Config()