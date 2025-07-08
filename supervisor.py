import io
import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

from agent_definations import (
    competitive_analysis_agent,
    market_trends_agent,
    consumer_insights_agent,
    market_sizing_agent
)

import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Setup the GPT-4o model
gpt4o = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

@dataclass
class Message:
    role: str
    content: str

class Supervisor:
    def __init__(self):

        agents = [
            competitive_analysis_agent,
            market_trends_agent,
            consumer_insights_agent,
            market_sizing_agent
        ]
        self.supervisor = create_supervisor(
            model=gpt4o,
            agents=agents,
            tools=[],
            prompt=self._create_prompt()
        ).compile()

    def _create_prompt(self):
        return f"""
        You are a Market Research Supervisor AI that coordinates a team of specialized market research agents.

        Today's date is: {str(datetime.today().strftime('%Y-%m-%d'))}
        Your job is to:
        
        1. Understand the user's market research questions
        2. Determine which specialized agents to consult
        3. Synthesize the information from different agents into a cohesive response
        4. Present findings in a clear, actionable format

        **You have access to the following specialized agents:** (Please use them all one by one in the order they are listed)
        - **competitive_analysis_agent**: Analyzes competitors' strengths, weaknesses, and market positioning.
        - **market_trends_agent**: Identifies and analyzes current and emerging market trends.
        - **consumer_insights_agent**: Gathers and interprets data on consumer preferences and behaviors.
        - **market_sizing_agent**: Estimates the size and growth potential of specific market segments.

        Maintain a strategic perspective and ensure that the insights provided are relevant to business decision-making.
        return to the user with a comprehensive market research report that includes:
        - Key findings from each agent
        - Strategic recommendations based on the combined insights
        
        **Your response should be structured as follows:**
        1. **Executive Summary**: A brief overview of the key findings and recommendations.
        2. **Detailed Findings**: In-depth analysis from each agent, including data and insights.
        3. **Strategic Recommendations**: Actionable steps based on the combined insights.
        4. **Market Landscape Overview**: A summary of the current market conditions, including trends and consumer behavior.
        5. **Competitive Analysis**: Insights into the competitive landscape, including key players and their strategies.
        6. **Conclusion**: A summary of the overall market landscape and future outlook.

        **Important Notes:**
        - Always ensure that the information is accurate and up-to-date.
        - For any research use today's date as a reference point.
        - If any agent is unable to provide information, note it and suggest alternative approaches.
        - Use clear and concise language to communicate findings.
        - Focus on providing actionable insights that can inform strategic decisions.
        
        """
    def run(self, messages: Optional[Dict[str, Any]] = None) -> str:
        if isinstance(messages, str):
            messages = Message(role="user", content=messages)

        history = []
        history.append({"role": messages.role, "content": messages.content})
        input_data = {
            "messages": history,
        }

        result = self.supervisor.invoke(input_data)
        return result["messages"][-1].content