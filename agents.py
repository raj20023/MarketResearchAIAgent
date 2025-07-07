"""
Optimized agent definitions with better modularity and configuration.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import config
from agent_tools import (
    analyze_competitors,
    search_news,
    consumer_sentiment_analysis,
    market_size_estimation,
    swot_analysis,
)

logger = logging.getLogger(__name__)

# Shared model instances
gpt4o_analytical = ChatOpenAI(
    model=config.models.model_name,
    temperature=config.models.analytical_temperature,
    api_key=config.api.openai_api_key,
    timeout=config.models.timeout
)

class BaseAgent:
    """Base class for all market research agents."""
    
    def __init__(self, name: str, tools: List, description: str, tasks: List[str]):
        self.name = name
        self.tools = tools
        self.description = description
        self.tasks = tasks
        self.agent = None
        self._create_agent()
    
    def _create_agent(self):
        """Create the agent with optimized prompt."""
        prompt = self._build_prompt()
        self.agent = create_react_agent(
            model=gpt4o_analytical,
            tools=self.tools,
            prompt=prompt,
            name=self.name
        )
    
    def _build_prompt(self) -> str:
        """Build the agent prompt with current date and standardized structure."""
        today = datetime.today().strftime('%Y-%m-%d')
        
        # Build tasks section
        tasks_section = "**Your tasks include:** (execute each task systematically)\n"
        for i, task in enumerate(self.tasks, 1):
            tasks_section += f"{i}. {task}\n"
        
        # Build tools section
        tools_section = "**Available Tools:**\n"
        for tool in self.tools:
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
            tool_desc = tool.description if hasattr(tool, 'description') else f"Tool: {tool_name}"
            tools_section += f"- `{tool_name}`: {tool_desc}\n"
        
        prompt = f"""
        Today's date is: {today}
        
        You are a {self.description}
        
        {tools_section}
        
        {tasks_section}
        
        **Guidelines:**
        - Execute tasks systematically and thoroughly
        - Use tools efficiently to gather comprehensive data
        - Provide clear, actionable insights
        - Maintain objectivity in your analysis
        - Include data sources and methodology when relevant
        - Format responses with clear structure and headings
        - Summarize key findings at the end before returning to supervisor
        
        Focus on delivering high-quality, data-driven insights that inform strategic decisions.
        """
        
        return prompt

class CompetitiveAnalysisAgent(BaseAgent):
    """Specialized agent for competitive analysis."""
    
    def __init__(self):
        super().__init__(
            name="competitive_analysis_agent",
            tools=[analyze_competitors, swot_analysis],
            description="competitive analysis specialist focused on market positioning, competitor research, and strategic recommendations.",
            tasks=[
                "Analyze competitors using the `analyze_competitors` tool to understand competitive landscape",
                "Perform SWOT analysis using the `swot_analysis` tool to identify strategic positioning",
                "Synthesize findings into actionable competitive intelligence"
            ]
        )

class MarketTrendsAgent(BaseAgent):
    """Specialized agent for market trends analysis."""
    
    def __init__(self):
        super().__init__(
            name="market_trends_agent",
            tools=[search_news],
            description="market trends research specialist focused on identifying emerging trends, industry developments, and market dynamics.",
            tasks=[
                "Search for latest news and trends using the `search_news` tool",
                "Identify emerging market patterns and developments",
                "Analyze industry news for strategic implications",
                "Provide forward-looking market insights"
            ]
        )

class ConsumerInsightsAgent(BaseAgent):
    """Specialized agent for consumer insights."""
    
    def __init__(self):
        super().__init__(
            name="consumer_insights_agent",
            tools=[consumer_sentiment_analysis],
            description="consumer insights specialist focused on understanding consumer behavior, sentiment, and preferences.",
            tasks=[
                "Conduct consumer sentiment analysis using the `consumer_sentiment_analysis` tool",
                "Analyze consumer behavior patterns and preferences",
                "Identify key factors driving consumer decisions",
                "Provide insights on consumer satisfaction and pain points"
            ]
        )

class MarketSizingAgent(BaseAgent):
    """Specialized agent for market sizing and forecasting."""
    
    def __init__(self):
        super().__init__(
            name="market_sizing_agent",
            tools=[market_size_estimation],
            description="market sizing and forecasting specialist focused on quantifying market opportunities and growth projections.",
            tasks=[
                "Estimate market size using the `market_size_estimation` tool",
                "Analyze market growth trajectories and projections",
                "Identify market segments and their relative sizes",
                "Assess factors influencing market expansion or contraction"
            ]
        )

# Agent factory for easy instantiation
class AgentFactory:
    """Factory class for creating optimized agents."""
    
    _agent_classes = {
        'competitive_analysis': CompetitiveAnalysisAgent,
        'market_trends': MarketTrendsAgent,
        'consumer_insights': ConsumerInsightsAgent,
        'market_sizing': MarketSizingAgent,
    }
    
    @classmethod
    def create_agent(cls, agent_type: str) -> BaseAgent:
        """Create an agent of the specified type."""
        if agent_type not in cls._agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(cls._agent_classes.keys())}")
        
        agent_class = cls._agent_classes[agent_type]
        return agent_class()
    
    @classmethod
    def create_all_agents(cls) -> List[BaseAgent]:
        """Create all available agents."""
        return [cls.create_agent(agent_type) for agent_type in cls._agent_classes.keys()]
    
    @classmethod
    def get_agent_names(cls) -> List[str]:
        """Get names of all available agents."""
        return list(cls._agent_classes.keys())

# Pre-instantiated agents for backward compatibility
competitive_analysis_agent = AgentFactory.create_agent('competitive_analysis').agent
market_trends_agent = AgentFactory.create_agent('market_trends').agent
consumer_insights_agent = AgentFactory.create_agent('consumer_insights').agent
market_sizing_agent = AgentFactory.create_agent('market_sizing').agent

# Export list of all agents
all_agents = [
    competitive_analysis_agent,
    market_trends_agent,
    consumer_insights_agent,
    market_sizing_agent,
]

def get_agent_info() -> Dict[str, Any]:
    """Get information about all available agents."""
    return {
        'available_agents': AgentFactory.get_agent_names(),
        'total_agents': len(AgentFactory._agent_classes),
        'capabilities': {
            'competitive_analysis': 'Competitor analysis, SWOT analysis, market positioning',
            'market_trends': 'News analysis, trend identification, industry developments',
            'consumer_insights': 'Sentiment analysis, consumer behavior, preferences',
            'market_sizing': 'Market size estimation, growth projections, forecasting'
        }
    }

def validate_agents() -> bool:
    """Validate that all agents are properly configured."""
    try:
        for agent_type in AgentFactory.get_agent_names():
            agent = AgentFactory.create_agent(agent_type)
            if not agent.agent:
                logger.error(f"Agent {agent_type} failed to initialize")
                return False
        
        logger.info("All agents validated successfully")
        return True
    except Exception as e:
        logger.error(f"Agent validation failed: {e}")
        return False

# Initialize and validate agents on import
if __name__ == "__main__":
    # Run validation when script is executed directly
    if validate_agents():
        print("✅ All agents initialized and validated successfully")
        agent_info = get_agent_info()
        print(f"📊 {agent_info['total_agents']} agents available: {', '.join(agent_info['available_agents'])}")
    else:
        print("❌ Agent validation failed")