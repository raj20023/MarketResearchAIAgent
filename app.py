"""
Optimized main application with enhanced error handling, monitoring, and user interface.
"""
import asyncio
import logging
import sys
import time
from typing import Optional, Dict, Any
from datetime import datetime
import argparse

from config import config
from supervisor import Supervisor
from cache_manager import cache_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format=config.logging.format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Optionally add file handler
        *([logging.FileHandler(config.logging.file_path)] if config.logging.file_path else [])
    ]
)

logger = logging.getLogger(__name__)

class MarketResearchApp:
    """
    Main application class for the market research system.
    Provides both CLI and programmatic interfaces.
    """
    
    def __init__(self, enable_cache: bool = True, debug: bool = False):
        self.enable_cache = enable_cache
        self.debug = debug
        self.supervisor: Optional[Supervisor] = None
        self.start_time = time.time()
        
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
    
    async def initialize(self) -> bool:
        """Initialize the application and all components."""
        try:
            logger.info("🚀 Initializing Market Research System...")
            
            # Initialize cache if enabled
            if self.enable_cache:
                await cache_manager.initialize()
                logger.info("✅ Cache system initialized")
            else:
                logger.info("⚠️  Cache system disabled")
            
            # Initialize supervisor
            self.supervisor = Supervisor()
            logger.info("✅ Supervisor initialized")
            
            # Validate system components
            if hasattr(self.supervisor, 'get_agent_status'):
                status = await self.supervisor.get_agent_status()
                logger.info(f"✅ System validation complete: {len(status.get('agents', {}))} agents available")
            
            initialization_time = time.time() - self.start_time
            logger.info(f"🎉 System ready in {initialization_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def run_analysis(self, query: str) -> Dict[str, Any]:
        """
        Run market research analysis with comprehensive error handling.
        
        Args:
            query: The market research question
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        if not self.supervisor:
            raise RuntimeError("Application not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            logger.info(f"📊 Starting analysis: {query[:100]}...")
            
            # Run the analysis
            result = self.supervisor.run(query)
            
            execution_time = time.time() - start_time
            
            # Prepare response with metadata
            response = {
                "success": True,
                "query": query,
                "result": result,
                "metadata": {
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "system_version": "optimized_v2.0",
                    "cache_enabled": self.enable_cache,
                    "result_length": len(result),
                }
            }
            
            logger.info(f"✅ Analysis completed in {execution_time:.2f} seconds")
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_response = {
                "success": False,
                "query": query,
                "error": str(e),
                "error_type": type(e).__name__,
                "metadata": {
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "system_version": "optimized_v2.0",
                }
            }
            
            logger.error(f"❌ Analysis failed after {execution_time:.2f} seconds: {e}")
            return error_response
    
    async def run_analysis_async(self, query: str) -> Dict[str, Any]:
        """Async version of run_analysis."""
        if not self.supervisor:
            raise RuntimeError("Application not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            logger.info(f"📊 Starting async analysis: {query[:100]}...")
            
            # Use async method if available
            if hasattr(self.supervisor, 'run_async'):
                result = await self.supervisor.run_async(query)
            else:
                # Fallback to sync method
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.supervisor.run, query)
            
            execution_time = time.time() - start_time
            
            response = {
                "success": True,
                "query": query,
                "result": result,
                "metadata": {
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "system_version": "optimized_v2.0",
                    "cache_enabled": self.enable_cache,
                    "result_length": len(result),
                    "async_execution": True
                }
            }
            
            logger.info(f"✅ Async analysis completed in {execution_time:.2f} seconds")
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_response = {
                "success": False,
                "query": query,
                "error": str(e),
                "error_type": type(e).__name__,
                "metadata": {
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "system_version": "optimized_v2.0",
                    "async_execution": True
                }
            }
            
            logger.error(f"❌ Async analysis failed after {execution_time:.2f} seconds: {e}")
            return error_response
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                "system": {
                    "initialized": self.supervisor is not None,
                    "uptime": time.time() - self.start_time,
                    "cache_enabled": self.enable_cache,
                    "debug_mode": self.debug,
                    "version": "optimized_v2.0"
                },
                "config": {
                    "model": config.models.model_name,
                    "temperature": config.models.temperature,
                    "timeout": config.models.timeout,
                    "cache_ttl": config.cache.ttl_seconds if config.cache.enabled else None
                }
            }
            
            # Add agent status if available
            if self.supervisor and hasattr(self.supervisor, 'get_agent_status'):
                agent_status = await self.supervisor.get_agent_status()
                status["agents"] = agent_status
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def clear_cache(self) -> bool:
        """Clear system cache."""
        try:
            if self.supervisor and hasattr(self.supervisor, 'clear_cache'):
                result = await self.supervisor.clear_cache()
            else:
                await cache_manager.clear_all()
                result = True
            
            logger.info("🗑️  Cache cleared successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def shutdown(self):
        """Gracefully shutdown the application."""
        logger.info("🛑 Shutting down Market Research System...")
        
        # Close executor if available
        if hasattr(self.supervisor, 'executor'):
            self.supervisor.executor.shutdown(wait=True)
        
        logger.info("✅ Shutdown complete")

async def main():
    """Main application entry point with CLI interface."""
    parser = argparse.ArgumentParser(description="Market Research Analysis System")
    parser.add_argument("query", nargs="?", 
                       default="I want to start an IT class in Ahmedabad, what is the market size and how to start it?",
                       help="Market research query")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache and exit")
    parser.add_argument("--async", dest="async_mode", action="store_true", help="Use async execution")
    
    args = parser.parse_args()
    
    # Create and initialize app
    app = MarketResearchApp(
        enable_cache=not args.no_cache,
        debug=args.debug
    )
    
    try:
        # Initialize system
        if not await app.initialize():
            logger.error("Failed to initialize system")
            return 1
        
        # Handle special commands
        if args.clear_cache:
            await app.clear_cache()
            return 0
        
        if args.status:
            status = await app.get_system_status()
            print("\n📊 System Status:")
            print("=" * 50)
            
            # Format status output
            for section, data in status.items():
                print(f"\n{section.upper()}:")
                if isinstance(data, dict):
                    for key, value in data.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {data}")
            
            return 0
        
        # Run analysis
        print(f"\n🔍 Query: {args.query}")
        print("=" * 80)
        
        if args.async_mode:
            result = await app.run_analysis_async(args.query)
        else:
            result = app.run_analysis(args.query)
        
        # Display results
        if result["success"]:
            print("\n📊 MARKET RESEARCH REPORT")
            print("=" * 80)
            print(result["result"])
            print("\n" + "=" * 80)
            print(f"✅ Analysis completed in {result['metadata']['execution_time']:.2f} seconds")
        else:
            print(f"\n❌ Analysis failed: {result['error']}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        app.shutdown()

def sync_main():
    """Synchronous entry point for backward compatibility."""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        return 130

if __name__ == "__main__":
    # Default example for testing
    async def run_example():
        app = MarketResearchApp(debug=True)
        
        if await app.initialize():
            query = "I want to start an IT class in Ahmedabad, what is the market size and how to start it?"
            
            print(f"\n🔍 Example Query: {query}")
            print("=" * 80)
            
            result = await app.run_analysis_async(query)
            
            if result["success"]:
                print("\n📊 MARKET RESEARCH REPORT")
                print("=" * 80)
                print(result["result"])
                print("\n" + "=" * 80)
                print(f"✅ Example completed in {result['metadata']['execution_time']:.2f} seconds")
            else:
                print(f"\n❌ Example failed: {result['error']}")
        
        app.shutdown()
    
    if len(sys.argv) > 1:
        # CLI mode
        exit_code = sync_main()
        sys.exit(exit_code)
    else:
        # Example mode
        print("🚀 Running Market Research System Example...")
        asyncio.run(run_example())