# Market Research AI Platform

![Market Research AI Banner](https://img.icons8.com/fluency/96/000000/economic-improvement.png)

A comprehensive, AI-powered market research platform that combines specialized agents to deliver in-depth analysis of markets, competitors, consumer sentiment, and industry trends.

## 🌟 Features

- **Multi-Agent Architecture**: Specialized AI agents working together to provide comprehensive market insights
- **Competitive Analysis**: Detailed analysis of competitors, their strengths, weaknesses, and market positioning
- **Consumer Insights**: Analysis of consumer sentiment, preferences, and behavior patterns
- **Market Sizing**: Data-driven market size estimation and growth projections
- **Trend Identification**: Recognition of emerging market trends and developments
- **SWOT Analysis**: Automated SWOT analysis for companies based on real data
- **Interactive UI**: User-friendly Streamlit interface for easy interaction

## 📊 Key Capabilities

- **Web Data Integration**: Extracts and analyzes information from multiple web sources
- **API Integration**: Connects with external APIs for enhanced data collection (News API, Google Maps, SerpAPI)
- **Sentiment Analysis**: Uses NLP to analyze consumer sentiment about products and services
- **Market Estimation**: Provides data-driven market size estimates with regional breakdowns
- **Automated Research**: Conducts complex market research tasks with minimal user input

## 🏗️ Architecture

The platform uses a hierarchical agent structure:

```
Supervisor
├── Competitive Analysis Agent
├── Consumer Insights Agent
├── Market Sizing Agent
└── Market Trends Agent
```

- **Supervisor**: Coordinates specialized agents and synthesizes their findings
- **Competitive Analysis Agent**: Analyzes competitive landscape using real data
- **Consumer Insights Agent**: Extracts and analyzes consumer sentiment and behavior
- **Market Sizing Agent**: Estimates market size, growth rates, and projections
- **Market Trends Agent**: Identifies current and emerging trends in the market

## 📂 Project Structure

```
market_research/
├── tools/                      # Tool implementations
│   ├── __init__.py
│   └── agent_tools.py          # Implementation of research tools
├── app.py                      # Entry point for running the application
├── agent_definations.py        # All agents definations
├── supervisor.py               # Agent orchestration and coordination
├── market_research_app.py      # Streamlit UI implementation
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) News API key
- (Optional) Google Maps API key
- (Optional) SerpAPI key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/market-research-ai.git
   cd market-research-ai
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   NEWS_API_KEY=your_news_api_key
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   SERP_API_KEY=your_serp_api_key
   ```

### Running the Application

Launch the Streamlit web interface:

```bash
streamlit run market_research_app.py
```

The web interface will be available at http://localhost:8501.

## 🔧 Usage Examples

### Basic Market Research

To conduct a basic market research:

```python
from supervisor import Supervisor

supervisor = Supervisor()
query = "What is the market size for electric vehicles in Europe?"
response = supervisor.run(query)
print(response)
```

### Comprehensive Market Analysis

For a more detailed analysis with specific components:

```python
query = """
Analyze the fitness app market with focus on:
1. Key competitors and their strengths
2. Current market size and growth projections
3. Consumer sentiment about existing solutions
4. Emerging trends in the industry
"""
response = supervisor.run(query)
```

### Localized Market Research

For location-specific market research:

```python
query = "I want to start an IT training center in Ahmedabad, what is the market size and how to start it?"
response = supervisor.run(query)
```

## 📊 Sample Outputs

### Market Size Analysis

```
# Market Size Analysis: IT Training in Ahmedabad

## Current Market Size
**$45.2 million**

## Annual Growth Rate (CAGR)
**12.5%**

## Projected Market Size (5 Years)
**$81.4 million**

## Key Market Drivers
- Rising demand for technical skills in the IT sector
- Growth of IT and ITES companies in Gujarat
- Government initiatives promoting digital literacy
- Increasing adoption of cloud and AI technologies

## Major Market Challenges
- Intense competition from established training centers
- Rapidly changing technology landscape requiring frequent curriculum updates
- Price sensitivity in the local market

## Leading Companies
- NIIT
- Jetking
- Aptech Computer Education
- Seed Infotech
- AkashTechnoLabs
```

### Competitive Analysis

```
# Competitive Analysis: Coffee Shop XYZ vs. Competitors

## Coffee Shop XYZ
A specialty coffee shop focusing on premium single-origin beans and artisanal brewing methods.

## Competitor Profiles

### Starbucks
**Competitive Position**: Starbucks is significantly larger with global brand recognition but offers a more standardized experience compared to Coffee Shop XYZ.

### Blue Bottle Coffee
**Competitive Position**: Blue Bottle operates in a similar premium segment with comparable quality but has stronger brand recognition and more locations.

### Philz Coffee
**Competitive Position**: Philz has a strong local presence with a loyal customer base and unique pour-over coffee approach.

## SWOT Analysis for Coffee Shop XYZ

### Strengths
- Specialty coffee expertise with high-quality beans
- Personalized customer experience
- Unique brewing methods not available at chain stores
- Strong relationships with ethical bean suppliers

### Weaknesses
- Limited physical presence
- Higher price points than mainstream competitors
- Lower brand recognition
- Limited marketing budget

### Opportunities
- Growing consumer interest in specialty coffee
- Potential for coffee subscription service
- Educational workshops on coffee brewing
- Partnerships with local businesses

### Threats
- Expansion of premium coffee chains into the local market
- Economic downturns affecting discretionary spending
- Rising costs of quality coffee beans
- Increasing competition in the specialty coffee segment
```

## 📚 API Documentation

### Supervisor

The `Supervisor` class coordinates multiple specialized agents to provide comprehensive market research.

```python
supervisor = Supervisor()
response = supervisor.run(query)
```

### Specialized Agents

The platform includes the following specialized agents:

- **competitive_analysis_agent**: Analyzes competitive positioning
- **consumer_insights_agent**: Analyzes consumer behavior and sentiment
- **market_sizing_agent**: Estimates market size and growth
- **market_trends_agent**: Identifies emerging trends

### Research Tools

The platform provides various research tools:

- **search_news**: Searches for recent news articles
- **analyze_competitors**: Analyzes competitive landscape
- **consumer_sentiment_analysis**: Analyzes sentiment about products/services
- **market_size_estimation**: Estimates market size and growth
- **swot_analysis**: Performs SWOT analysis for companies

## 📋 Dependencies

Major dependencies include:

- **langchain**: For creating and managing AI agents
- **openai**: For accessing GPT-4o and other OpenAI models
- **streamlit**: For the web interface
- **nltk**: For sentiment analysis
- **beautifulsoup4**: For web scraping
- **matplotlib**: For data visualization
- **pandas**: For data manipulation

Optional dependencies:

- **yfinance**: For financial data on public companies
- **googlemaps**: For local business data
- **serpapi**: For enhanced web search

## ⚠️ Troubleshooting

### API Key Issues

If you encounter errors related to API keys:

1. Verify that your `.env` file contains the correct API keys
2. Ensure the environment variables are being loaded properly
3. Check that your API keys are active and have sufficient quota

### Rate Limiting

If you encounter rate limiting issues:

1. Reduce the number of consecutive requests
2. Implement retry logic with exponential backoff
3. Consider upgrading your API plans for higher rate limits

### Missing Dependencies

If you encounter missing dependencies:

```bash
pip install -r requirements.txt
```

For optional dependencies:

```bash
pip install yfinance googlemaps google-search-results
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- OpenAI for GPT-4o
- LangChain for the agent framework
- All the data providers and API services that make this platform possible