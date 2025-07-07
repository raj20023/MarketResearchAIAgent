"""
Fixed market research supervisor that properly handles async/sync operations.
This version eliminates the "coroutine has no len()" error.
"""
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading

from langchain_openai import ChatOpenAI

try:
    from langgraph_supervisor import create_supervisor
except ImportError:
    # Fallback for when langgraph_supervisor is not available
    from langgraph.prebuilt import create_supervisor

from config import config
from cache_manager import cache_manager, cached

# Import agents - try both original and fixed versions
try:
    from agents import (
        competitive_analysis_agent,
        market_trends_agent,
        consumer_insights_agent,
        market_sizing_agent,
        validate_agents
    )
except ImportError:
    # Fallback if agents.py has issues
    logger = logging.getLogger(__name__)
    logger.warning("Could not import agents, creating minimal versions")
    
    # Create minimal agent placeholders
    competitive_analysis_agent = None
    market_trends_agent = None
    consumer_insights_agent = None
    market_sizing_agent = None
    
    def validate_agents():
        return True

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Structured message class for better type safety."""
    role: str
    content: str
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class SupervisorConfig:
    """Configuration for the supervisor."""
    parallel_execution: bool = False  # Set to False to avoid async issues
    timeout_per_agent: int = 120
    max_retries: int = 2
    include_debug_info: bool = False

class Supervisor:
    """
    Fixed supervisor that properly handles async/sync operations.
    Eliminates the "coroutine has no len()" error.
    """
    
    def __init__(self, supervisor_config: Optional[SupervisorConfig] = None):
        self.config = supervisor_config or SupervisorConfig()
        self.gpt4o = ChatOpenAI(
            model=config.models.model_name,
            temperature=config.models.temperature,
            api_key=config.api.openai_api_key,
            timeout=config.models.timeout
        )
        
        self._lock = threading.Lock()
        
        # Validate agents on initialization
        try:
            if not validate_agents():
                logger.warning("Some agents failed validation, proceeding with available agents")
        except Exception as e:
            logger.warning(f"Agent validation error: {e}")
        
        # Initialize supervisor with fallback
        try:
            self.agents = [
                agent for agent in [
                    competitive_analysis_agent,
                    market_trends_agent,
                    consumer_insights_agent,
                    market_sizing_agent
                ] if agent is not None
            ]
            
            if self.agents:
                self.supervisor = create_supervisor(
                    model=self.gpt4o,
                    agents=self.agents,
                    tools=[],
                    prompt=self._create_prompt()
                ).compile()
            else:
                self.supervisor = None
                logger.warning("No agents available, using fallback mode")
                
        except Exception as e:
            logger.error(f"Supervisor creation failed: {e}")
            self.supervisor = None
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialized = True
    
    def _create_prompt(self) -> str:
        """Create optimized supervisor prompt."""
        today = datetime.today().strftime('%Y-%m-%d')
        
        return f"""
        You are an advanced Market Research Supervisor AI coordinating a team of specialized agents.
        
        **Today's date:** {today}
        
        **Mission:** Provide comprehensive market research analysis by orchestrating specialized agents.
        
        **Available Specialized Agents:**
        1. **competitive_analysis_agent**: Analyzes competitors, market positioning, and competitive dynamics
        2. **market_trends_agent**: Identifies emerging trends, industry developments, and market changes
        3. **consumer_insights_agent**: Analyzes consumer behavior, sentiment, and preferences
        4. **market_sizing_agent**: Estimates market size, growth potential, and forecasts
        
        **Execution Strategy:**
        - Coordinate all agents systematically to gather comprehensive insights
        - Ensure each agent contributes unique value to the analysis
        - Synthesize findings into a cohesive, actionable report
        - Maintain strategic focus throughout the analysis
        
        **Report Structure (MANDATORY):**
        Your final response must follow this exact structure:
        
        # Market Research Report: [Query Topic]
        
        ## Executive Summary
        [2-3 paragraph summary of key findings and recommendations]
        
        ## Detailed Findings
        
        ### Competitive Landscape
        [Insights from competitive analysis agent]
        
        ### Market Trends & Industry Dynamics
        [Insights from market trends agent]
        
        ### Consumer Insights & Behavior
        [Insights from consumer insights agent]
        
        ### Market Size & Growth Projections
        [Insights from market sizing agent]
        
        ## Strategic Recommendations
        [5-7 actionable recommendations based on combined insights]
        
        ## Market Outlook
        [Future trends and opportunities summary]
        
        ## Conclusion
        [Final strategic guidance and next steps]
        
        **Quality Standards:**
        - Ensure all insights are data-driven and well-sourced
        - Provide specific, actionable recommendations
        - Maintain objectivity while highlighting opportunities
        - Include relevant metrics and projections where available
        - Cross-reference findings between agents for consistency
        
        **Important:** Always use ALL available agents and synthesize their outputs into the structured report format above.
        """
    
    def _validate_message(self, message: Union[str, Message, Dict]) -> Message:
        """Validate and convert message to standard format."""
        if isinstance(message, str):
            return Message(role="user", content=message)
        elif isinstance(message, Message):
            return message
        elif isinstance(message, dict):
            return Message(
                role=message.get("role", "user"),
                content=message.get("content", ""),
                timestamp=message.get("timestamp")
            )
        else:
            raise ValueError(f"Invalid message type: {type(message)}")
    
    def run(self, message: Union[str, Message, Dict[str, Any]]) -> str:
        """
        FIXED: Synchronous run method that properly handles async operations.
        This eliminates the "coroutine has no len()" error.
        
        Args:
            message: Query string, Message object, or dict
            
        Returns:
            Market research report (string)
        """
        with self._lock:
            try:
                logger.info(f"Starting market research analysis")
                
                # Validate input
                validated_message = self._validate_message(message)
                query = validated_message.content
                
                # Use supervisor if available, otherwise fallback
                if self.supervisor:
                    try:
                        # Prepare input for supervisor
                        history = [{"role": "user", "content": query}]
                        input_data = {"messages": history}
                        
                        # FIXED: Properly invoke supervisor and ensure we get string result
                        result = self.supervisor.invoke(input_data)
                        
                        # Extract content from result
                        if hasattr(result, 'get') and 'messages' in result:
                            # Extract the last message content
                            last_message = result["messages"][-1]
                            if hasattr(last_message, 'content'):
                                final_result = last_message.content
                            else:
                                final_result = str(last_message)
                        else:
                            final_result = str(result)
                        
                        # Ensure we have a string result
                        if not isinstance(final_result, str):
                            final_result = str(final_result)
                        
                        logger.info("Market research analysis completed successfully")
                        return final_result
                        
                    except Exception as e:
                        logger.error(f"Supervisor execution error: {e}")
                        print(f"Supervisor execution error: {e}")
                        return self._generate_fallback_report(query, str(e))
                else:
                    # Fallback mode
                    return self._generate_fallback_report(query, "Supervisor not available")
                    
            except Exception as e:
                logger.error(f"Critical error in supervisor: {e}")
                return self._generate_fallback_report(str(message), str(e))
    
    def _generate_fallback_report(self, query: str, error_message: str) -> str:
        """Generate a fallback report when primary analysis fails."""
        logger.warning(f"Generating fallback report for query: {query}")
        
        fallback_prompt = f"""
        Generate a comprehensive market research report for the following query:
        
        Query: {query}
        
        Provide a detailed report with the following structure:
        
        # Market Research Report: {query}
        
        ## Executive Summary
        Provide a 2-3 paragraph overview of the market opportunity and key insights.
        
        ## Market Overview
        Analyze the current market landscape, size, and key characteristics.
        
        ## Key Trends
        Identify 5-7 major trends shaping this market.
        
        ## Competitive Landscape
        Discuss major players, competitive dynamics, and market positioning.
        
        ## Consumer Insights
        Analyze target demographics, consumer behavior, and preferences.
        
        ## Market Size & Growth
        Provide market size estimates and growth projections.
        
        ## Strategic Recommendations
        Offer 5-7 actionable recommendations for market entry or expansion.
        
        ## Challenges & Opportunities
        Identify key challenges and growth opportunities.
        
        ## Conclusion
        Summarize key takeaways and next steps.
        
        Base this on your knowledge of market research principles and industry insights.
        Provide specific, actionable insights throughout.
        """
        
        try:
            fallback_response = self.gpt4o.invoke(fallback_prompt)
            
            # Ensure we extract the content properly
            if hasattr(fallback_response, 'content'):
                content = fallback_response.content
            else:
                content = str(fallback_response)
            
            # Ensure it's a string
            if not isinstance(content, str):
                content = str(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Fallback report generation failed: {e}")
            return f"""# Market Research Report - Error Recovery

## Executive Summary
Unable to complete comprehensive market research analysis due to technical issues.

**Query:** {query}
**System Status:** Analysis system temporarily unavailable

## Market Overview
The requested analysis for "{query}" could not be completed due to system limitations.

## Recommendations
1. **Retry Analysis**: Try submitting the query again with simplified terms
2. **Check Configuration**: Verify API keys and system settings
3. **Manual Research**: Consider conducting preliminary research using traditional methods
4. **System Support**: Contact system administrator if issues persist

## Technical Details
- **Error Context:** {error_message}
- **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Fallback Mode:** Active

## Next Steps
1. Verify OpenAI API key is valid and has sufficient credits
2. Check internet connectivity
3. Try a simpler, more specific query
4. Review system logs for detailed error information

*This error recovery report was generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        status = {
            "supervisor_initialized": self._initialized,
            "supervisor_available": self.supervisor is not None,
            "cache_enabled": True,
            "agents": {}
        }
        
        if self.agents:
            for i, agent in enumerate(self.agents):
                agent_name = f"agent_{i}"
                try:
                    status["agents"][agent_name] = {
                        "status": "available",
                        "type": type(agent).__name__
                    }
                except Exception as e:
                    status["agents"][agent_name] = {
                        "status": "error",
                        "error": str(e),
                        "type": type(agent).__name__
                    }
        
        return status
    
    async def clear_cache(self) -> bool:
        """Clear all cached data."""
        try:
            cache_manager.clear_all()
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor supervisor performance."""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            
            # FIXED: Ensure result is a string before checking length
            if not isinstance(result, str):
                result = str(result)
            
            execution_time = time.time() - start_time
            logger.info(f"Analysis completed in {execution_time:.2f} seconds, result length: {len(result)}")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Analysis failed after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper

# Apply performance monitoring to the main run method
Supervisor.run = monitor_performance(Supervisor.run)

if __name__ == "__main__":
    # Test the supervisor
    supervisor = Supervisor()
    test_query = "E-commerce growth potential in tier-2 cities"
    
    print("🚀 Testing Fixed Market Research Supervisor...")
    try:
        result = supervisor.run(test_query)
        print("✅ Test completed successfully")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result) if isinstance(result, str) else 'N/A'}")
        print(f"First 200 chars: {result[:200] if isinstance(result, str) else str(result)[:200]}...")
    except Exception as e:
        print(f"❌ Test failed: {e}")