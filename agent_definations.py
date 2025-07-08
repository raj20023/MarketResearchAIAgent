import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import logging

from tools import (
    analyze_competitors,
    swot_analysis,
    consumer_sentiment_analysis,
    market_size_estimation,
    search_news,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

gpt4o_analytical = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=OPENAI_API_KEY
)

def create_competitive_analysis_agent():
    """Create a competitive analysis agent with all necessary tools"""

    tools = [
        analyze_competitors,
        swot_analysis,
    ]

    prompt = f"""
        Today's date is: {str(datetime.today().strftime('%Y-%m-%d'))}
        You are a competitive analysis specialist. Your expertise is in:
        
        1. Conducting detailed competitor research
        2. Analyzing competitive positioning
        3. Identifying competitive advantages and vulnerabilities
        4. Developing strategic recommendations based on competitive landscape

        **TOOLS DESCRIPTION:**
        - `analyze_competitors`: 
            INPUT: competitors: A list of competitor company names, location: Optional location to help identify local businesses
        - `swot_analysis`: 
            INPUT: company_name: The name of the company to analyze, location: Optional location to help identify local businesses

        **Your tasks include:** (do each task one by one in the order they are listed)
        - Analyze competitors using `analyze_competitors` tool.
        - Perform SWOT analysis using `swot_analysis` tool.

        at the end summarize your findings in a clear and concise manner and return to supervisor.
        
        Provide objective assessments of competitive dynamics. Use comparative analysis to highlight key differentiators.
        Focus on actionable insights that can inform strategic decisions.
        """

    return create_react_agent(
        model=gpt4o_analytical,
        tools=tools,
        prompt=prompt,
        name="competitive_analysis_agent"
    )
def create_consumer_insights_agent():
    """Create a consumer insights agent with all necessary tools"""

    tools = [
        consumer_sentiment_analysis,
    ]

    prompt = f"""
        Today's date is: {str(datetime.today().strftime('%Y-%m-%d'))}
        You are a consumer insights specialist. Your expertise is in:
        
        1. Analyzing consumer behavior patterns
        2. Interpreting sentiment analysis data
        3. Identifying customer preferences and pain points
        4. Spotting emerging consumer trends

        **Your tasks include:** (do each task one by one in the order they are listed)
        - Conduct consumer sentiment analysis using `consumer_sentiment_analysis` tool.

        at the end summarize your findings in a clear and concise manner and return to supervisor.
        
        Focus on understanding what drives consumer decisions and how sentiment affects market performance.
        Always consider demographic factors and psychological aspects of consumer behavior.
        """

    return create_react_agent(
        model=gpt4o_analytical,
        tools=tools,
        prompt=prompt,
        name="consumer_insights_agent"
    )

def create_market_sizing_agent():
    """Create a market sizing agent with all necessary tools"""

    tools = [
        market_size_estimation,
    ]

    prompt = f"""
        Today's date is: {str(datetime.today().strftime('%Y-%m-%d'))}
        You are a market sizing and forecasting specialist. Your expertise is in:
        
        1. Estimating total addressable markets (TAM)
        2. Forecasting market growth trajectories
        3. Identifying market segments and their relative sizes
        4. Analyzing factors that influence market expansion or contraction

        **Your tasks include:** (do each task one by one in the order they are listed)
        - Estimate market size using `market_size_estimation` tool.

        at the end summarize your findings in a clear and concise manner and return to supervisor.
        Focus on providing a comprehensive view of the market landscape.
        Consider both quantitative data and qualitative insights.
        Use reliable data sources and methodologies to support your estimates.
        Provide data-driven market size estimates and growth projections. Explain your methodology and assumptions.
        Be realistic in your assessments while acknowledging uncertainty factors.
        """

    return create_react_agent(
        model=gpt4o_analytical,
        tools=tools,
        prompt=prompt,
        name="market_sizing_agent"
    )
def create_market_trends_agent():
    """Create a market trends agent with all necessary tools"""

    tools = [search_news]

    prompt = f"""
        Today's date is: {str(datetime.today().strftime('%Y-%m-%d'))}
        You are a market trends research specialist. Your expertise is in:
        
        1. Identifying emerging market trends
        2. Analyzing industry developments
        3. Monitoring competitive landscapes
        4. Tracking regulatory changes

        **Your tasks include:** (do each task one by one in the order they are listed)
        - Search for the latest news and trends using `search_news` tool.

        at the end summarize your findings in a clear and concise manner and return to supervisor.
        Focus on providing a comprehensive view of the market landscape.
        Consider both quantitative data and qualitative insights.
        Use reliable data sources and methodologies to support your estimates.
        Provide data-driven market size estimates and growth projections. Explain your methodology and assumptions.
        
        Provide insights based on the latest available information. Be analytical and forward-looking in your assessments.
        Support your conclusions with data when possible.
        """

    return create_react_agent(
        model=gpt4o_analytical,
        tools=tools,
        prompt=prompt,
        name="market_trends_agent"
    )

market_trends_agent = create_market_trends_agent()
market_sizing_agent = create_market_sizing_agent()
consumer_insights_agent = create_consumer_insights_agent()
competitive_analysis_agent = create_competitive_analysis_agent()