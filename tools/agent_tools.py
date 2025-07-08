"""
Market Research Agent Tools - Optimized version with smaller, modular functions.
"""
import os
import re
import json
import time
import logging
import base64
import requests
from typing import List, Dict, Any, Optional, Union, Tuple
from io import BytesIO

# Third-party imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Configure services if available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import googlemaps
    GOOGLEMAPS_AVAILABLE = True
except ImportError:
    GOOGLEMAPS_AVAILABLE = False

try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

# Configure matplotlib for non-interactive environments
import matplotlib
matplotlib.use('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langgraph").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Download NLTK data for sentiment analysis
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Setup OpenAI models
gpt4o = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

gpt4o_analytical = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=OPENAI_API_KEY
)

# Initialize Google Maps client if API key is available
gmaps = None
if GOOGLE_MAPS_API_KEY and GOOGLEMAPS_AVAILABLE:
    try:
        gmaps = googlemaps.Client(
            key=GOOGLE_MAPS_API_KEY, 
            requests_kwargs={'timeout': 20}
        )
    except Exception as e:
        logger.error(f"Error initializing Google Maps client: {e}")

# Helper functions for web requests
def make_web_request(url: str) -> Optional[str]:
    """Make a web request with proper error handling and user agent."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.text
    except requests.RequestException as e:
        logger.warning(f"Web request failed: {e}")
        return None

def parse_html(html_content: str) -> Optional[BeautifulSoup]:
    """Parse HTML content into BeautifulSoup object with error handling."""
    if not html_content:
        return None
    
    try:
        return BeautifulSoup(html_content, 'html.parser')
    except Exception as e:
        logger.warning(f"Failed to parse HTML: {e}")
        return None

# Company data extraction functions
def get_company_data_from_yahoo_finance(company_name: str) -> Optional[Dict[str, Any]]:
    """Attempt to get company data from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        ticker_search = yf.Tickers(company_name)
        if not ticker_search.tickers:
            return None
            
        ticker = list(ticker_search.tickers.keys())[0]
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if we got valid data
        if 'longName' in info or 'shortName' in info:
            return {
                'data_source': 'yahoo_finance',
                'is_public': True,
                'ticker': ticker,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'description': info.get('longBusinessSummary', ''),
                'website': info.get('website', ''),
                'market_cap': info.get('marketCap', None),
                'revenue': info.get('totalRevenue', None),
                'profit_margin': info.get('profitMargin', None),
                'employees': info.get('fullTimeEmployees', None)
            }
    except Exception as e:
        logger.warning(f"Error getting Yahoo Finance data for {company_name}: {e}")
    
    return None

def get_company_data_from_google_maps(company_name: str, location: str = None) -> Optional[Dict[str, Any]]:
    """Attempt to get company data from Google Maps."""
    if not gmaps:
        return None
    
    try:
        input_text = f"{company_name} {location}" if location else company_name
        
        # Search for the company using Google Places API
        places_result = gmaps.find_place(
            input=input_text,
            input_type="textquery",
            fields=["place_id", "name", "formatted_address", "rating", "user_ratings_total"]
        )

        if places_result['status'] != 'OK' or not places_result['candidates']:
            return None
            
        place_id = places_result['candidates'][0]['place_id']
        
        # Get detailed place information
        place_details = gmaps.place(place_id=place_id,
                                fields=["name", "rating", "formatted_address", 
                                        "formatted_phone_number", "website",
                                        "opening_hours", "user_ratings_total"])
        
        if place_details['status'] != 'OK':
            return None
            
        details = place_details['result']
        
        # Determine business size based on review count
        estimated_size = 'Small local business'
        if 'user_ratings_total' in details:
            if details['user_ratings_total'] > 1000:
                estimated_size = 'Large local business'
            elif details['user_ratings_total'] > 200:
                estimated_size = 'Medium local business'
        
        # Extract relevant information
        return {
            'data_source': 'google_maps',
            'is_public': False,
            'address': details.get('formatted_address', ''),
            'phone': details.get('formatted_phone_number', ''),
            'rating': details.get('rating', None),
            'reviews_count': details.get('user_ratings_total', None),
            'website': details.get('website', ''),
            'hours': details.get('opening_hours', {}).get('weekday_text', []),
            'place_id': place_id,
            'estimated_size': estimated_size
        }
    except Exception as e:
        logger.warning(f"Error getting Google Maps data for {company_name}: {e}")
    
    return None

def get_company_data_from_serp_api(company_name: str, location: str = None) -> Optional[Dict[str, Any]]:
    """Attempt to get company data from SERP API."""
    if not SERP_API_KEY or not SERPAPI_AVAILABLE:
        return None
    
    try:
        search_query = f"{company_name} company"
        if location:
            search_query += f" {location}"
            
        search_params = {
            "engine": "google",
            "q": search_query,
            "api_key": SERP_API_KEY
        }
        
        search_results = GoogleSearch(search_params).get_dict()
        
        # Check if we have knowledge graph data
        if 'knowledge_graph' in search_results:
            kg = search_results['knowledge_graph']
            
            company_data = {
                'data_source': 'knowledge_graph',
                'is_public': False,
                'description': kg.get('description', ''),
                'website': kg.get('website', ''),
                'type': kg.get('type', ''),
                'founded': kg.get('founded', '')
            }
            
            # Extract structured attributes
            if 'attributes' in kg:
                for key, value in kg['attributes'].items():
                    if key.lower() == 'headquarters':
                        company_data['headquarters'] = value
                    elif key.lower() == 'ceo' or key.lower() == 'founder':
                        company_data['leadership'] = value
                    elif key.lower() == 'revenue':
                        company_data['revenue_text'] = value
                    elif key.lower() == 'employees':
                        company_data['employees_text'] = value
            
            return company_data
        
        # If no knowledge graph, try to extract from organic results
        if 'organic_results' in search_results and search_results['organic_results']:
            # Get the first result that seems to be about the company
            for result in search_results['organic_results']:
                if company_name.lower() in result.get('title', '').lower():
                    return {
                        'data_source': 'web_search',
                        'is_public': False,
                        'description': result.get('snippet', ''),
                        'website': result.get('link', '')
                    }
    except Exception as e:
        logger.warning(f"Error getting web search data for {company_name}: {e}")
    
    return None

def get_company_data_from_web_scraping(company_name: str, location: str = None) -> Optional[Dict[str, Any]]:
    """Attempt to get company data through web scraping."""
    try:
        # Format search query
        search_query = f"{company_name} company"
        if location:
            search_query += f" {location}"
            
        search_query = search_query.replace(' ', '+')
        url = f"https://www.google.com/search?q={search_query}"
        
        html_content = make_web_request(url)
        if not html_content:
            return None
            
        soup = parse_html(html_content)
        if not soup:
            return None
        
        # Try to extract information from Google's knowledge panel
        knowledge_panel = soup.select_one('div.kp-wholepage')
        if not knowledge_panel:
            return None
            
        # Try to get company description
        description = None
        desc_div = soup.select_one('div.kno-rdesc')
        if desc_div:
            description = desc_div.text.replace('Description', '').strip()
        
        # Try to get website
        website = None
        website_div = soup.select_one('a.ab_button')
        if website_div and 'href' in website_div.attrs:
            website = website_div['href']
        
        if description or website:
            return {
                'data_source': 'web_scraping',
                'is_public': False,
                'description': description,
                'website': website
            }
    except Exception as e:
        logger.warning(f"Error scraping web data for {company_name}: {e}")
    
    return None

def generate_company_estimation(company_name: str, location: str = None) -> Dict[str, Any]:
    """Generate an estimation of company details using GPT-4o when other sources fail."""
    try:
        # Generate company description and details
        prompt = f"""Generate reasonable information about a company named "{company_name}"
        {f"located in {location}" if location else ""}.
        
        Provide the following:
        1. A brief description of what the company likely does based on its name
        2. An estimated industry/sector
        3. An estimated company size (small, medium, large)
        
        Format the response as a JSON with keys: description, industry, sector, size
        """
        
        estimation_response = gpt4o.invoke(prompt)
        
        try:
            estimation = json.loads(estimation_response.content)
            return {
                'data_source': 'gpt4o_estimation',
                'is_public': False,
                **estimation
            }
        except json.JSONDecodeError:
            # If JSON parsing fails, just extract some basic info
            return {
                'data_source': 'gpt4o_estimation',
                'is_public': False,
                'description': estimation_response.content
            }
    except Exception as e:
        logger.error(f"Error generating company estimation: {e}")
        return {
            'data_source': 'gpt4o_estimation',
            'is_public': False,
            'description': f"No detailed information available for {company_name}."
        }

def get_company_data(company_name: str, location: str = None) -> Dict[str, Any]:
    """
    Get company data from multiple sources, prioritizing reliable data.
    
    Args:
        company_name: The name of the company
        location: Optional location to narrow down search
    
    Returns:
        Dictionary with company information
    """
    company_data = {
        'name': company_name,
        'data_source': None
    }
    
    # Try each data source in order of reliability
    data_sources = [
        # Uncomment when using yfinance
        # get_company_data_from_yahoo_finance,
        get_company_data_from_google_maps,
        get_company_data_from_serp_api,
        get_company_data_from_web_scraping
    ]
    
    for source_func in data_sources:
        source_data = source_func(company_name, location)
        if source_data:
            company_data.update(source_data)
            return company_data
    
    # If all sources fail, use GPT-4o estimation
    estimation_data = generate_company_estimation(company_name, location)
    company_data.update(estimation_data)
    
    return company_data

# News search functions
def get_news_from_news_api(query: str) -> Optional[str]:
    """Get news articles from News API."""
    if not NEWS_API_KEY:
        return None
    
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=relevancy&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return None
            
        news_data = response.json()
        articles = news_data.get('articles', [])[:5]  # Get top 5 articles
        
        if not articles:
            return None
            
        summary = f"Found {len(articles)} recent articles about {query}:\n\n"
        for i, article in enumerate(articles):
            title = article.get('title', 'No title')
            source = article.get('source', {}).get('name', 'Unknown')
            description = article.get('description', 'No description')
            url = article.get('url', '')
            published = article.get('publishedAt', '').split('T')[0]
            
            summary += f"{i+1}. {title} ({source}, {published})\n"
            summary += f"   {description}\n"
            summary += f"   URL: {url}\n\n"
        
        return summary
    except Exception as e:
        logger.error(f"Error with NewsAPI: {e}")
        return None

def get_news_from_serp_api(query: str) -> Optional[str]:
    """Get news articles from SERP API."""
    if not SERP_API_KEY or not SERPAPI_AVAILABLE:
        return None
    
    try:
        logger.info(f"Searching news using SERP API for query: {query}")
        search_params = {
            "engine": "google_news",
            "q": query,
            "api_key": SERP_API_KEY
        }
        
        search_results = GoogleSearch(search_params).get_dict()
        
        if 'news_results' not in search_results or not search_results['news_results']:
            return None
            
        articles = search_results['news_results']
        
        summary = f"Found {len(articles)} recent articles about {query}:\n\n"
        for i, article in enumerate(articles[:5]):
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown')
            snippet = article.get('snippet', 'No description')
            url = article.get('link', '')
            date = article.get('date', '')
            
            summary += f"{i+1}. {title} ({source}, {date})\n"
            summary += f"   {snippet}\n"
            summary += f"   URL: {url}\n\n"
        
        return summary
    except Exception as e:
        logger.error(f"Error with SERP API: {e}")
        return None

def get_news_from_web_scraping(query: str) -> Optional[str]:
    """Get news articles through web scraping."""
    try:
        # Format query for Google News
        search_query = query.replace(' ', '+')
        url = f"https://www.google.com/search?q={search_query}&tbm=nws"
        
        html_content = make_web_request(url)
        if not html_content:
            return None
            
        soup = parse_html(html_content)
        if not soup:
            return None
        
        # Extract news articles
        articles = []
        for g in soup.find_all('div', class_='SoaBEf'):
            title_elem = g.find('div', class_='mCBkyc')
            source_elem = g.find('div', class_='CEMjEf')
            snippet_elem = g.find('div', class_='GI74Re')
            
            if title_elem and source_elem and snippet_elem:
                title = title_elem.text
                source = source_elem.text
                snippet = snippet_elem.text
                
                articles.append({
                    'title': title,
                    'source': source,
                    'snippet': snippet
                })
        
        if not articles:
            return None
            
        summary = f"Found {len(articles)} recent articles about {query}:\n\n"
        for i, article in enumerate(articles[:5]):  # Limit to top 5
            summary += f"{i+1}. {article['title']} ({article['source']})\n"
            summary += f"   {article['snippet']}\n\n"
        
        return summary
    except Exception as e:
        logger.error(f"Error scraping news: {e}")
        return None

def extract_trends_from_articles(query: str, summary: str) -> str:
    """Extract key trends from news article summaries using GPT-4o."""
    trend_prompt = f"Based on these news articles about {query}, identify 3-5 key market trends:\n\n{summary}"
    trend_response = gpt4o.invoke(trend_prompt)
    return trend_response.content

def generate_news_estimate(query: str) -> str:
    """Generate an estimated news summary when all other sources fail."""
    prompt = f"""Create a summary of what might be in recent news about {query}.
    Include:
    1. Types of articles that might be written
    2. Potential key trends
    3. Possible market developments
    
    Format this as if it were an actual news summary.
    """
    
    response = gpt4o.invoke(prompt)
    return response.content

@tool
def search_news(query: str) -> str:
    """
    Search for recent news articles related to a market research query.
    
    Args:
        query: The search query for news articles
    
    Returns:
        A summary of relevant news articles
    """
    logger.info(f"Searching news for: {query}")
    
    # Try each news source in order
    news_sources = [
        get_news_from_news_api,
        get_news_from_serp_api,
        get_news_from_web_scraping
    ]
    
    for source_func in news_sources:
        summary = source_func(query)
        if summary:
            # Extract key trends from the articles
            trends = extract_trends_from_articles(query, summary)
            return f"{summary}Key trends:\n{trends}"
    
    # If all sources fail, generate an estimated summary
    return generate_news_estimate(query)

# Competitor analysis functions
def format_competitor_main_info(company_name: str, company_data: Dict[str, Any]) -> str:
    """Format the main company information section for competitor analysis."""
    analysis = f"## {company_name}\n\n"
    
    if company_data.get('description'):
        analysis += f"{company_data['description']}\n\n"
    
    if company_data.get('sector') and company_data.get('industry'):
        analysis += f"Sector: {company_data['sector']}\n"
        analysis += f"Industry: {company_data['industry']}\n\n"
    
    if company_data.get('website'):
        analysis += f"Website: {company_data['website']}\n\n"
    
    return analysis

def format_serp_company_info(company_name: str, company_data: Dict[str, Any]) -> str:
    """Format SERP API company information."""
    if company_data.get('data_source') not in ['knowledge_graph', 'web_search']:
        return ""
        
    analysis = "### Company Information\n\n"
    
    fields = [
        ('type', 'Type'),
        ('founded', 'Founded'),
        ('headquarters', 'Headquarters'),
        ('leadership', 'Leadership'),
        ('revenue_text', 'Revenue'),
        ('employees_text', 'Employees')
    ]
    
    for field, label in fields:
        if company_data.get(field):
            analysis += f"{label}: {company_data[field]}\n"
            
    analysis += "\n"
    return analysis

def format_financial_metrics(company_name: str, company_data: Dict[str, Any]) -> str:
    """Format financial metrics for public companies."""
    if not company_data.get('is_public', False) or company_data.get('data_source') != 'yahoo_finance':
        return ""
        
    analysis = "### Financial Metrics\n\n"
    
    if company_data.get('market_cap'):
        market_cap_billions = company_data['market_cap'] / 1_000_000_000
        analysis += f"Market Cap: ${market_cap_billions:.2f} billion\n"
    
    if company_data.get('revenue'):
        revenue_billions = company_data['revenue'] / 1_000_000_000
        analysis += f"Revenue: ${revenue_billions:.2f} billion\n"
    
    if company_data.get('profit_margin'):
        analysis += f"Profit Margin: {company_data['profit_margin']:.2%}\n"
    
    if company_data.get('employees'):
        analysis += f"Employees: {company_data['employees']:,}\n"
        
    analysis += "\n"
    return analysis

def format_local_business_metrics(company_name: str, company_data: Dict[str, Any]) -> str:
    """Format metrics for local businesses."""
    if company_data.get('data_source') != 'google_maps':
        return ""
        
    analysis = "### Business Metrics\n\n"
    
    if company_data.get('rating') and company_data.get('reviews_count'):
        analysis += f"Rating: {company_data['rating']}/5 ({company_data['reviews_count']} reviews)\n"
    
    if company_data.get('address'):
        analysis += f"Address: {company_data['address']}\n"
    
    if company_data.get('phone'):
        analysis += f"Phone: {company_data['phone']}\n"
        
    if company_data.get('estimated_size'):
        analysis += f"Estimated Size: {company_data['estimated_size']}\n"
        
    analysis += "\n"
    return analysis

def create_financial_comparison_table(company_name: str, main_company_data: Dict[str, Any], 
                                    competitors: List[str], competitor_data: Dict[str, Dict[str, Any]]) -> str:
    """Create a financial comparison table for public companies."""
    # Check if we have public companies to compare
    is_public_analysis = main_company_data.get('is_public', False) and main_company_data.get('data_source') == 'yahoo_finance'
    any_public_competitors = any(data.get('is_public', False) and data.get('data_source') == 'yahoo_finance' 
                               for data in competitor_data.values())
    
    if not (is_public_analysis or any_public_competitors):
        return ""
        
    # Get public companies
    public_companies = [company_name] if main_company_data.get('is_public', False) else []
    public_companies += [comp for comp in competitors if competitor_data[comp].get('is_public', False)]
    
    if not public_companies:
        return ""
        
    table = "### Financial Comparison (Public Companies)\n\n"
    
    # Create a table header
    table += "| Company | Market Cap | Revenue | Profit Margin |\n"
    table += "| --- | --- | --- | --- |\n"
    
    # Add main company if public
    if main_company_data.get('is_public', False):
        market_cap = f"${main_company_data.get('market_cap', 0) / 1_000_000_000:.2f}B" if main_company_data.get('market_cap') else "N/A"
        revenue = f"${main_company_data.get('revenue', 0) / 1_000_000_000:.2f}B" if main_company_data.get('revenue') else "N/A"
        profit_margin = f"{main_company_data.get('profit_margin', 0):.2%}" if main_company_data.get('profit_margin') else "N/A"
        
        table += f"| {company_name} | {market_cap} | {revenue} | {profit_margin} |\n"
    
    # Add competitors if public
    for comp in competitors:
        data = competitor_data[comp]
        if data.get('is_public', False):
            market_cap = f"${data.get('market_cap', 0) / 1_000_000_000:.2f}B" if data.get('market_cap') else "N/A"
            revenue = f"${data.get('revenue', 0) / 1_000_000_000:.2f}B" if data.get('revenue') else "N/A"
            profit_margin = f"{data.get('profit_margin', 0):.2%}" if data.get('profit_margin') else "N/A"
            
            table += f"| {comp} | {market_cap} | {revenue} | {profit_margin} |\n"
    
    table += "\n"
    return table

def create_company_info_comparison(company_name: str, main_company_data: Dict[str, Any],
                                 competitors: List[str], competitor_data: Dict[str, Dict[str, Any]]) -> str:
    """Create a company information comparison table for companies with SERP data."""
    # Check for SerpAPI data sources (knowledge_graph or web_search)
    has_serp_main = main_company_data.get('data_source') in ['knowledge_graph', 'web_search']
    has_serp_competitors = any(data.get('data_source') in ['knowledge_graph', 'web_search'] 
                             for data in competitor_data.values())
    
    if not (has_serp_main or has_serp_competitors):
        return ""
        
    serp_companies = [company_name] if main_company_data.get('data_source') in ['knowledge_graph', 'web_search'] else []
    serp_companies += [comp for comp in competitors if competitor_data[comp].get('data_source') in ['knowledge_graph', 'web_search']]
    
    if not serp_companies:
        return ""
        
    table = "### Company Information Comparison\n\n"
    
    # Create a table with available information
    table += "| Company | Type | Founded | Location | Website |\n"
    table += "| --- | --- | --- | --- | --- |\n"
    
    # Add main company if it has SerpAPI data
    if main_company_data.get('data_source') in ['knowledge_graph', 'web_search']:
        company_type = main_company_data.get('type', 'N/A')
        founded = main_company_data.get('founded', 'N/A')
        location = main_company_data.get('headquarters', 'N/A')
        website = main_company_data.get('website', 'N/A')
        
        table += f"| {company_name} | {company_type} | {founded} | {location} | {website} |\n"
    
    # Add competitors with SerpAPI data
    for comp in competitors:
        data = competitor_data[comp]
        if data.get('data_source') in ['knowledge_graph', 'web_search']:
            company_type = data.get('type', 'N/A')
            founded = data.get('founded', 'N/A')
            location = data.get('headquarters', 'N/A')
            website = data.get('website', 'N/A')
            
            table += f"| {comp} | {company_type} | {founded} | {location} | {website} |\n"
    
    table += "\n"
    return table

def create_local_business_comparison(company_name: str, main_company_data: Dict[str, Any],
                                   competitors: List[str], competitor_data: Dict[str, Dict[str, Any]]) -> str:
    """Create a comparison table for local businesses."""
    # Check for local businesses
    local_businesses = []
    if main_company_data.get('data_source') == 'google_maps':
        local_businesses.append(company_name)
    
    local_businesses += [comp for comp in competitors if competitor_data[comp].get('data_source') == 'google_maps']
    
    if not local_businesses:
        return ""
        
    table = "### Local Business Comparison\n\n"
    
    # Create a table header
    table += "| Company | Rating | Reviews | Estimated Size |\n"
    table += "| --- | --- | --- | --- |\n"
    
    # Add main company if local
    if main_company_data.get('data_source') == 'google_maps':
        rating = f"{main_company_data.get('rating', 'N/A')}/5" if main_company_data.get('rating') else "N/A"
        reviews = f"{main_company_data.get('reviews_count', 'N/A'):,}" if main_company_data.get('reviews_count') else "N/A"
        size = main_company_data.get('estimated_size', 'Unknown')
        
        table += f"| {company_name} | {rating} | {reviews} | {size} |\n"
    
    # Add competitors if local
    for comp in competitors:
        data = competitor_data[comp]
        if data.get('data_source') == 'google_maps':
            rating = f"{data.get('rating', 'N/A')}/5" if data.get('rating') else "N/A"
            reviews = f"{data.get('reviews_count', 'N/A'):,}" if data.get('reviews_count') else "N/A"
            size = data.get('estimated_size', 'Unknown')
            
            table += f"| {comp} | {rating} | {reviews} | {size} |\n"
    
    table += "\n"
    return table

def generate_competitive_position(competitor: str, company_name: str, 
                                competitor_data: Dict[str, Any], 
                                main_company_data: Dict[str, Any]) -> str:
    """Generate a competitive position statement based on available data."""
    # For public companies, compare market caps
    if (competitor_data.get('data_source') == 'yahoo_finance' and 
        main_company_data.get('data_source') == 'yahoo_finance'):
        
        if competitor_data.get('market_cap') and main_company_data.get('market_cap'):
            ratio = competitor_data['market_cap'] / main_company_data['market_cap']
            
            if ratio > 2:
                return f"**Competitive Position**: {competitor} is significantly larger than {company_name} with {ratio:.1f}x the market cap.\n\n"
            elif ratio > 1.2:
                return f"**Competitive Position**: {competitor} is somewhat larger than {company_name} with {ratio:.1f}x the market cap.\n\n"
            elif ratio > 0.8:
                return f"**Competitive Position**: {competitor} is comparable in size to {company_name}.\n\n"
            elif ratio > 0.5:
                return f"**Competitive Position**: {competitor} is somewhat smaller than {company_name} with {1/ratio:.1f}x smaller market cap.\n\n"
            else:
                return f"**Competitive Position**: {competitor} is significantly smaller than {company_name} with {1/ratio:.1f}x smaller market cap.\n\n"
    
    # For local businesses, compare ratings
    elif (competitor_data.get('data_source') == 'google_maps' and 
          main_company_data.get('data_source') == 'google_maps'):
          
        if competitor_data.get('rating') and main_company_data.get('rating'):
            rating_diff = competitor_data['rating'] - main_company_data['rating']
            
            if rating_diff > 0.5:
                return f"**Competitive Position**: {competitor} has significantly better ratings than {company_name}.\n\n"
            elif rating_diff > 0.2:
                return f"**Competitive Position**: {competitor} has somewhat better ratings than {company_name}.\n\n"
            elif rating_diff > -0.2:
                return f"**Competitive Position**: {competitor} has comparable ratings to {company_name}.\n\n"
            elif rating_diff > -0.5:
                return f"**Competitive Position**: {competitor} has somewhat lower ratings than {company_name}.\n\n"
            else:
                return f"**Competitive Position**: {competitor} has significantly lower ratings than {company_name}.\n\n"
    
    # For companies with employee data from SerpAPI
    elif (competitor_data.get('data_source') in ['knowledge_graph', 'web_search'] and 
          main_company_data.get('data_source') in ['knowledge_graph', 'web_search']):
        
        # Compare employee counts if available
        if competitor_data.get('employees_text') and main_company_data.get('employees_text'):
            try:
                # Try to extract numbers from employee text
                data_emp = re.search(r'(\d[\d,]*)', competitor_data.get('employees_text'))
                main_emp = re.search(r'(\d[\d,]*)', main_company_data.get('employees_text'))
                
                if data_emp and main_emp:
                    data_count = int(data_emp.group(1).replace(',', ''))
                    main_count = int(main_emp.group(1).replace(',', ''))
                    
                    ratio = data_count / main_count
                    
                    if ratio > 2:
                        return f"**Competitive Position**: {competitor} is significantly larger than {company_name} with approximately {ratio:.1f}x more employees.\n\n"
                    elif ratio > 1.2:
                        return f"**Competitive Position**: {competitor} is somewhat larger than {company_name} with approximately {ratio:.1f}x more employees.\n\n"
                    elif ratio > 0.8:
                        return f"**Competitive Position**: {competitor} has a comparable employee count to {company_name}.\n\n"
                    elif ratio > 0.5:
                        return f"**Competitive Position**: {competitor} is somewhat smaller than {company_name} with approximately {1/ratio:.1f}x fewer employees.\n\n"
                    else:
                        return f"**Competitive Position**: {competitor} is significantly smaller than {company_name} with approximately {1/ratio:.1f}x fewer employees.\n\n"
            except Exception:
                pass
        
        # Compare founding dates if available
        if competitor_data.get('founded') and main_company_data.get('founded'):
            try:
                data_year = re.search(r'(\d{4})', competitor_data.get('founded'))
                main_year = re.search(r'(\d{4})', main_company_data.get('founded'))
                
                if data_year and main_year:
                    data_founding = int(data_year.group(1))
                    main_founding = int(main_year.group(1))
                    
                    years_diff = data_founding - main_founding
                    
                    if years_diff < -20:
                        return f"**Competitive Position**: {competitor} is a more established company than {company_name}, founded approximately {abs(years_diff)} years earlier.\n\n"
                    elif years_diff < -5:
                        return f"**Competitive Position**: {competitor} was founded before {company_name} by approximately {abs(years_diff)} years.\n\n"
                    elif years_diff <= 5:
                        return f"**Competitive Position**: {competitor} and {company_name} were founded around the same time period.\n\n"
                    elif years_diff <= 20:
                        return f"**Competitive Position**: {competitor} is a newer company than {company_name}, founded approximately {years_diff} years later.\n\n"
                    else:
                        return f"**Competitive Position**: {competitor} is a much newer company than {company_name}, founded approximately {years_diff} years later.\n\n"
            except Exception:
                pass
    
    # Fall back to GPT-4o comparison for all other cases
    prompt = f"""
    Generate a brief competitive position statement comparing {competitor} to {company_name}.
    Base this on the following information:
    
    {company_name} information:
    {json.dumps(main_company_data, indent=2)}
    
    {competitor} information:
    {json.dumps(competitor_data, indent=2)}
    
    Keep the statement to 1-2 sentences focusing on their relative market position.
    """
    
    position_response = gpt4o.invoke(prompt)
    return f"**Competitive Position**: {position_response.content}\n\n"

def format_competitor_profile(competitor: str, company_name: str, 
                            competitor_data: Dict[str, Any], 
                            main_company_data: Dict[str, Any]) -> str:
    """Format a competitor profile with details and position."""
    profile = f"### {competitor}\n\n"
    
    if competitor_data.get('description'):
        profile += f"{competitor_data['description']}\n\n"
    
    if competitor_data.get('sector') and competitor_data.get('industry'):
        profile += f"Sector: {competitor_data['sector']}\n"
        profile += f"Industry: {competitor_data['industry']}\n\n"
    
    if competitor_data.get('website'):
        profile += f"Website: {competitor_data['website']}\n\n"
        
    # Add SerpAPI specific information if available
    if competitor_data.get('data_source') in ['knowledge_graph', 'web_search']:
        fields = [
            ('type', 'Type'),
            ('founded', 'Founded'),
            ('headquarters', 'Headquarters'),
            ('leadership', 'Leadership'),
            ('revenue_text', 'Revenue'),
            ('employees_text', 'Employees')
        ]
        
        for field, label in fields:
            if competitor_data.get(field):
                profile += f"{label}: {competitor_data[field]}\n"
        
        profile += "\n"
    
    # Add competitive position statement
    profile += generate_competitive_position(competitor, company_name, competitor_data, main_company_data)
    
    return profile

def generate_swot_analysis(company_name: str, company_data: Dict[str, Any], 
                         competitor_data: Dict[str, Dict[str, Any]]) -> str:
    """Generate a SWOT analysis based on company and competitor data."""
    swot_prompt = f"""
    Create a SWOT analysis for {company_name} based on the following competitive information:
    
    Company information:
    {json.dumps(company_data, indent=2)}
    
    Competitor information:
    {json.dumps(competitor_data, indent=2)}
    
    Provide 3-5 bullet points for each of: Strengths, Weaknesses, Opportunities, and Threats.
    Format as a markdown list with proper headings for each SWOT category.
    """
    
    swot_response = gpt4o.invoke(swot_prompt)
    return swot_response.content

@tool
def analyze_competitors(company_name: str, competitors: List[str], location: str = None) -> str:
    """
    Analyze the competitive landscape for a company using real data including local business data.
    
    Args:
        company_name: The name of the main company
        competitors: A list of competitor company names
        location: Optional location to help identify local businesses
    
    Returns:
        Competitive analysis summary
    """
    logger.info(f"Analyzing competitors for {company_name}: {competitors} in {location if location else 'unknown location'}")
    
    # Get data for main company
    main_company_data = get_company_data(company_name, location)
    
    # Get data for competitors
    competitor_data = {}
    for competitor in competitors:
        competitor_data[competitor] = get_company_data(competitor, location)
    
    # Format analysis
    analysis = f"# Competitive Analysis: {company_name} vs. Competitors\n\n"
    
    # Add information about the main company
    analysis += format_competitor_main_info(company_name, main_company_data)
    
    # Add additional information for SerpAPI data
    analysis += format_serp_company_info(company_name, main_company_data)
    
    # Add financial metrics if available (for public companies)
    analysis += format_financial_metrics(company_name, main_company_data)
    
    # Add local business metrics if available
    analysis += format_local_business_metrics(company_name, main_company_data)
    
    # Add competitor analysis section
    analysis += "## Competitor Analysis\n\n"
    
    # Add comparison tables
    analysis += create_financial_comparison_table(company_name, main_company_data, competitors, competitor_data)
    analysis += create_company_info_comparison(company_name, main_company_data, competitors, competitor_data)
    analysis += create_local_business_comparison(company_name, main_company_data, competitors, competitor_data)
    
    # Add individual competitor profiles
    analysis += "## Competitor Profiles\n\n"
    
    for competitor in competitors:
        analysis += format_competitor_profile(
            competitor, 
            company_name, 
            competitor_data[competitor], 
            main_company_data
        )
    
    # Generate SWOT for the main company based on the competitive landscape
    analysis += f"## SWOT Analysis for {company_name}\n\n"
    analysis += generate_swot_analysis(company_name, main_company_data, competitor_data)
    
    return analysis

# Consumer sentiment analysis functions
def get_reviews_from_google_maps(product: str, location: str = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Get reviews from Google Maps."""
    reviews = []
    source = None
    
    if not gmaps or not location:
        return reviews, source
    
    try:
        # Search for the product/service using Google Places API
        places_result = gmaps.places(f"{product} {location}")
        
        if places_result['status'] != 'OK' or not places_result['results']:
            return reviews, source
            
        source = "Google Maps"
        
        # Get reviews for top 3 places
        for place in places_result['results'][:3]:
            place_id = place['place_id']
            place_name = place.get('name', 'Unknown business')
            
            # Get place details including reviews
            place_details = gmaps.place(place_id=place_id)
            
            if place_details['status'] == 'OK' and 'reviews' in place_details['result']:
                for review in place_details['result']['reviews']:
                    reviews.append({
                        'text': review.get('text', ''),
                        'rating': review.get('rating', 0),
                        'time': review.get('time', 0),
                        'source': place_name
                    })
    except Exception as e:
        logger.warning(f"Error getting Google Maps reviews: {e}")
    
    return reviews, source

def get_reviews_from_amazon(product: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Get product reviews from Amazon."""
    reviews = []
    source = None
    
    try:
        # Try to scrape product reviews from Amazon
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Search for product on Amazon
        search_query = product.replace(' ', '+')
        url = f"https://www.amazon.com/s?k={search_query}"
        
        html_content = make_web_request(url)
        if not html_content:
            return reviews, source
            
        soup = parse_html(html_content)
        if not soup:
            return reviews, source
        
        # Try to find the first product link
        product_link = None
        for a in soup.find_all('a', class_='a-link-normal s-no-outline'):
            if 'href' in a.attrs and '/dp/' in a['href']:
                product_link = f"https://www.amazon.com{a['href']}"
                break
        
        if not product_link:
            return reviews, source
            
        source = "Amazon"
        
        # Get the product page
        html_content = make_web_request(product_link)
        if not html_content:
            return reviews, source
            
        soup = parse_html(html_content)
        if not soup:
            return reviews, source
        
        # Find the reviews section
        reviews_link = None
        for a in soup.find_all('a'):
            if 'href' in a.attrs and 'customerReviews' in a['href']:
                reviews_link = f"https://www.amazon.com{a['href']}"
                break
        
        if not reviews_link:
            return reviews, source
            
        # Get the reviews page
        html_content = make_web_request(reviews_link)
        if not html_content:
            return reviews, source
            
        soup = parse_html(html_content)
        if not soup:
            return reviews, source
        
        # Extract reviews
        for review_div in soup.find_all('div', {'data-hook': 'review'}):
            rating_span = review_div.find('i', {'data-hook': 'review-star-rating'})
            if rating_span:
                rating_text = rating_span.text.strip()
                rating = float(rating_text.split(' out of')[0])
            else:
                rating = None
            
            body_span = review_div.find('span', {'data-hook': 'review-body'})
            if body_span:
                review_text = body_span.text.strip()
            else:
                review_text = ""
            
            reviews.append({
                'text': review_text,
                'rating': rating,
                'source': 'Amazon'
            })
    except Exception as e:
        logger.warning(f"Error scraping Amazon reviews: {e}")
    
    return reviews, source

def get_reviews_from_web_search(product: str, location: str = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Get reviews from general web search."""
    reviews = []
    source = None
    
    try:
        # Search for reviews
        search_query = f"{product} reviews"
        if location:
            search_query += f" {location}"
            
        search_query = search_query.replace(' ', '+')
        url = f"https://www.google.com/search?q={search_query}"
        
        html_content = make_web_request(url)
        if not html_content:
            return reviews, source
            
        soup = parse_html(html_content)
        if not soup:
            return reviews, source
        
        source = "Web search"
        
        # Try to find review snippets
        for div in soup.find_all('div', class_='BNeawe'):
            text = div.text
            if len(text) > 50 and any(word in text.lower() for word in ['review', 'rating', 'recommend', 'stars', 'loved', 'hated']):
                # Try to extract a rating if present
                rating = None
                rating_match = re.search(r'(\d\.?\d?)/5', text)
                if rating_match:
                    try:
                        rating = float(rating_match.group(1))
                    except:
                        pass
                
                reviews.append({
                    'text': text,
                    'rating': rating,
                    'source': 'Web search'
                })
    except Exception as e:
        logger.warning(f"Error searching for reviews: {e}")
    
    return reviews, source

def analyze_sentiment(reviews: List[Dict[str, Any]]) -> Tuple[float, float, float, List[str], List[str]]:
    """Analyze sentiment of reviews and extract key phrases."""
    positive_count = 0
    neutral_count = 0
    negative_count = 0
    
    positive_phrases = []
    negative_phrases = []
    
    for review in reviews:
        # Skip empty reviews
        if not review.get('text') or len(review['text']) < 10:
            continue
            
        # Use NLTK sentiment analyzer
        sentiment = sia.polarity_scores(review['text'])
        compound_score = sentiment['compound']
        
        # Also consider star rating if available
        if review.get('rating') is not None:
            if review['rating'] >= 4:
                positive_count += 1
            elif review['rating'] <= 2:
                negative_count += 1
            else:
                neutral_count += 1
        else:
            # Use sentiment score only
            if compound_score >= 0.05:
                positive_count += 1
            elif compound_score <= -0.05:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Extract key phrases
        sentences = review['text'].split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10 or len(sentence) > 150:
                continue
            
            sentence_sentiment = sia.polarity_scores(sentence)
            if sentence_sentiment['compound'] >= 0.4:  # Higher threshold for positive phrases
                positive_phrases.append(sentence)
            elif sentence_sentiment['compound'] <= -0.3:  # Lower threshold for negative phrases
                negative_phrases.append(sentence)
    
    total = positive_count + neutral_count + negative_count
    if total == 0:
        return 0, 0, 0, [], []
        
    positive_percent = (positive_count / total) * 100
    neutral_percent = (neutral_count / total) * 100
    negative_percent = (negative_count / total) * 100
    
    return positive_percent, neutral_percent, negative_percent, positive_phrases, negative_phrases

def extract_sentiment_themes(product: str, positive_phrases: List[str], negative_phrases: List[str]) -> Tuple[str, str]:
    """Extract key themes from positive and negative phrases."""
    positive_prompt = f"Extract 3-5 key themes from these positive comments about {product}:\n\n" + '\n'.join(positive_phrases[:15])
    positive_themes = gpt4o.invoke(positive_prompt).content
    
    negative_prompt = f"Extract 3-5 key themes from these negative comments about {product}:\n\n" + '\n'.join(negative_phrases[:15])
    negative_themes = gpt4o.invoke(negative_prompt).content
    
    return positive_themes, negative_themes

def generate_sentiment_estimation(product: str, location: str = None) -> str:
    """Generate an estimated sentiment analysis when no reviews are found."""
    prompt = f"""Create a consumer sentiment analysis for {product}{" in " + location if location else ""}.
    Include:
    1. An estimated percentage breakdown of positive, neutral, and negative sentiment
    2. Key themes that likely appear in positive reviews
    3. Key themes that likely appear in negative reviews
    4. Sample quotes that might represent actual consumer opinions
    
    Format your response with markdown headings and bullet points for readability.
    Note in your response that this is an AI-generated estimation due to limited review data.
    """
    
    response = gpt4o.invoke(prompt)
    return response.content

@tool
def consumer_sentiment_analysis(product: str, location: str = None) -> str:
    """
    Analyze consumer sentiment about a product or service using real data, including local reviews.
    
    Args:
        product: The product or service name to analyze
        location: Optional location for local products/services
    
    Returns:
        Summary of consumer sentiment
    """
    logger.info(f"Analyzing consumer sentiment for: {product} in {location if location else 'any location'}")
    
    # Initialize variables
    reviews = []
    sources = []
    
    # Try to get reviews from various sources
    google_reviews, google_source = get_reviews_from_google_maps(product, location)
    if google_reviews:
        reviews.extend(google_reviews)
        if google_source:
            sources.append(google_source)
    
    # If we don't have enough reviews from Google Maps, try Amazon
    if len(reviews) < 10:
        amazon_reviews, amazon_source = get_reviews_from_amazon(product)
        if amazon_reviews:
            reviews.extend(amazon_reviews)
            if amazon_source:
                sources.append(amazon_source)
    
    # If we still don't have enough reviews, try looking for reviews on Google
    if len(reviews) < 5:
        web_reviews, web_source = get_reviews_from_web_search(product, location)
        if web_reviews:
            reviews.extend(web_reviews)
            if web_source:
                sources.append(web_source)
    
    # If we have reviews, analyze them
    if reviews:
        # Analyze sentiment
        positive_percent, neutral_percent, negative_percent, positive_phrases, negative_phrases = analyze_sentiment(reviews)
        
        # Extract themes
        if positive_phrases and negative_phrases:
            positive_themes, negative_themes = extract_sentiment_themes(product, positive_phrases, negative_phrases)
        else:
            positive_themes = "No clear positive themes identified."
            negative_themes = "No clear negative themes identified."
        
        # Format the source text
        source_text = " and ".join(sources) if sources else "various sources"
        
        # Prepare the result
        result = f"# Consumer Sentiment Analysis: {product}\n\n"
        result += f"Based on analysis of {len(reviews)} reviews from {source_text}:\n\n"
        
        result += f"## Sentiment Distribution\n\n"
        result += f"- **Positive**: {positive_percent:.1f}%\n"
        result += f"- **Neutral**: {neutral_percent:.1f}%\n"
        result += f"- **Negative**: {negative_percent:.1f}%\n\n"
        
        result += f"## Key Positive Themes\n\n{positive_themes}\n\n"
        result += f"## Key Negative Themes\n\n{negative_themes}\n\n"
        
        # Add sample quotes
        if positive_phrases:
            result += "## Sample Positive Comments\n\n"
            for phrase in positive_phrases[:3]:  # Limit to 3 examples
                result += f"- \"{phrase}\"\n"
            result += "\n"
        
        if negative_phrases:
            result += "## Sample Negative Comments\n\n"
            for phrase in negative_phrases[:3]:  # Limit to 3 examples
                result += f"- \"{phrase}\"\n"
            result += "\n"
        
        return result
    
    # If we couldn't find reviews, use GPT-4o to generate a reasonable response
    return generate_sentiment_estimation(product, location)

# Market size estimation functions
def extract_market_data_from_text(snippets: List[str]) -> Tuple[List[float], List[float]]:
    """Extract market size and growth rate data from text snippets."""
    market_sizes = []
    growth_rates = []
    
    for snippet in snippets:
        # Look for market size numbers
        size_matches = re.findall(r'(\$?\s?\d+\.?\d*)\s*(billion|million|trillion)', snippet, re.IGNORECASE)
        for match in size_matches:
            value, unit = match
            try:
                value = float(value.replace('$', '').strip())
                if unit.lower() == 'million':
                    value /= 1000  # Convert to billions
                elif unit.lower() == 'trillion':
                    value *= 1000  # Convert to billions
                market_sizes.append(value)
            except ValueError:
                continue
        
        # Look for growth rates (CAGR)
        growth_matches = re.findall(r'(\d+\.?\d*)\s*%', snippet)
        for match in growth_matches:
            try:
                growth_rate = float(match)
                if growth_rate <= 30:  # Filter out implausibly high percentages
                    growth_rates.append(growth_rate)
            except ValueError:
                continue
    
    return market_sizes, growth_rates

def get_market_data_from_serp_api(industry: str, region: str) -> Tuple[List[str], List[float], List[float]]:
    """Get market size data from SERP API."""
    snippets = []
    market_sizes = []
    growth_rates = []
    
    if not SERP_API_KEY or not SERPAPI_AVAILABLE:
        return snippets, market_sizes, growth_rates
    
    try:
        search_query = f"{industry} market size {region} billion dollars report"
        
        search_params = {
            "engine": "google",
            "q": search_query,
            "api_key": SERP_API_KEY
        }
        
        search_results = GoogleSearch(search_params).get_dict()
        
        # Extract snippets from organic results
        if 'organic_results' in search_results:
            for result in search_results['organic_results']:
                if 'snippet' in result:
                    snippets.append(result['snippet'])
        
        # Extract market size and growth rate data
        if snippets:
            extracted_sizes, extracted_rates = extract_market_data_from_text(snippets)
            market_sizes.extend(extracted_sizes)
            growth_rates.extend(extracted_rates)
    except Exception as e:
        logger.warning(f"Error using SERP API for market size: {e}")
    
    return snippets, market_sizes, growth_rates

def get_market_data_from_web_scraping(industry: str, region: str) -> Tuple[List[str], List[float], List[float]]:
    """Get market size data through web scraping."""
    snippets = []
    market_sizes = []
    growth_rates = []
    
    try:
        # Try to find market reports via web search
        search_query = f"{industry} market size {region} billion dollars"
        search_query = search_query.replace(' ', '+')
        
        url = f"https://www.google.com/search?q={search_query}"
        
        html_content = make_web_request(url)
        if not html_content:
            return snippets, market_sizes, growth_rates
            
        soup = parse_html(html_content)
        if not soup:
            return snippets, market_sizes, growth_rates
        
        # Extract snippets that might contain market size information
        for div in soup.select('div.VwiC3b'):
            text = div.text
            if any(keyword in text.lower() for keyword in ['market size', 'billion', 'million', 'cagr', 'growth']):
                snippets.append(text)
        
        # Extract market size and growth rate data
        if snippets:
            extracted_sizes, extracted_rates = extract_market_data_from_text(snippets)
            market_sizes.extend(extracted_sizes)
            growth_rates.extend(extracted_rates)
    except Exception as e:
        logger.warning(f"Error scraping market size data: {e}")
    
    return snippets, market_sizes, growth_rates

def get_regional_market_data(industry: str, regions: List[str]) -> Dict[str, float]:
    """Get market share percentages for different regions."""
    local_market_data = {}
    
    for local_region in regions:
        try:
            local_search_query = f"{industry} market size {local_region} percentage share"
            local_search_query = local_search_query.replace(' ', '+')
            
            url = f"https://www.google.com/search?q={local_search_query}"
            
            html_content = make_web_request(url)
            if not html_content:
                continue
                
            soup = parse_html(html_content)
            if not soup:
                continue
            
            local_snippets = []
            for div in soup.select('div.VwiC3b'):
                text = div.text
                if any(keyword in text.lower() for keyword in ['market share', 'percent', 'regional breakdown']):
                    local_snippets.append(text)
            
            # Look for percentage patterns for this region
            percentage_matches = []
            for snippet in local_snippets:
                matches = re.findall(rf'({local_region}|{local_region.split()[0]})[^\d]*(\d+\.?\d*)%', snippet, re.IGNORECASE)
                percentage_matches.extend([float(m[1]) for m in matches if float(m[1]) <= 100])
            
            if percentage_matches:
                # Use median value
                local_market_data[local_region] = sum(percentage_matches) / len(percentage_matches)
        except Exception as e:
            logger.warning(f"Error getting data for {local_region}: {e}")
    
    return local_market_data

def filter_outliers(values: List[float]) -> List[float]:
    """Filter outliers from a list of values using IQR method."""
    if len(values) < 3:
        return values
        
    values.sort()
    q1_idx = len(values) // 4
    q3_idx = (3 * len(values)) // 4
    q1 = values[q1_idx]
    q3 = values[q3_idx]
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    filtered_values = [val for val in values if lower_bound <= val <= upper_bound]
    return filtered_values if filtered_values else values

def extract_additional_market_info(industry: str, region: str, snippets: List[str]) -> str:
    """Extract additional market information using GPT-4o."""
    prompt = f"""Based on these market research snippets about the {industry} market in {region}, provide:
    
    1. Key market drivers (3-5 points)
    2. Major market challenges (2-3 points)
    3. Leading companies in this market (3-5 companies)
    4. Future trends (2-3 points)
    
    Snippets:
    {snippets[:5]}
    
    Format your response with markdown headings and bullet points.
    """
    
    additional_info = gpt4o.invoke(prompt).content
    return additional_info

@tool
def market_size_estimation(industry: str, region: str = "global", include_local: bool = False) -> str:
    """
    Estimate the market size for a specific industry and region using real data, including local market data when requested.
    
    Args:
        industry: The industry to analyze
        region: The geographic region (default: global)
        include_local: Whether to include local market breakdowns
    
    Returns:
        Market size estimation and growth projections
    """
    logger.info(f"Estimating market size for {industry} in {region} (include_local={include_local})")
    
    # Initialize variables
    all_snippets = []
    market_sizes = []
    growth_rates = []
    
    # Try to get market data from various sources
    serp_snippets, serp_sizes, serp_rates = get_market_data_from_serp_api(industry, region)
    all_snippets.extend(serp_snippets)
    market_sizes.extend(serp_sizes)
    growth_rates.extend(serp_rates)
    
    # If SERP API didn't yield enough results, try web scraping
    if len(market_sizes) < 2 or len(growth_rates) < 2:
        web_snippets, web_sizes, web_rates = get_market_data_from_web_scraping(industry, region)
        all_snippets.extend(web_snippets)
        market_sizes.extend(web_sizes)
        growth_rates.extend(growth_rates)
    
    # For local market breakdown, search for region-specific data if requested
    local_market_data = {}
    if include_local and region.lower() == "global":
        regions_to_search = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
        local_market_data = get_regional_market_data(industry, regions_to_search)
    
    # Generate the market size report
    result = f"# Market Size Analysis: {industry} in {region}\n\n"
    
    # Process market size data
    if market_sizes:
        # Filter out outliers
        filtered_sizes = filter_outliers(market_sizes)
        
        # Use median value for market size
        median_size = sorted(filtered_sizes)[len(filtered_sizes)//2]
        result += f"## Current Market Size\n\n"
        result += f"**${median_size:.1f} billion**\n\n"
        
        # Process growth rate data
        if growth_rates:
            # Filter out outliers
            filtered_rates = filter_outliers(growth_rates)
            
            # Use median value for growth rate
            median_growth = sorted(filtered_rates)[len(filtered_rates)//2]
            result += f"## Annual Growth Rate (CAGR)\n\n"
            result += f"**{median_growth:.1f}%**\n\n"
            
            # Calculate projected market size
            projected_size = median_size * (1 + (median_growth/100)) ** 5
            result += f"## Projected Market Size (5 Years)\n\n"
            result += f"**${projected_size:.1f} billion**\n\n"
        
        # Add regional breakdown if available
        if local_market_data:
            result += "## Regional Market Share\n\n"
            
            for region_name, percentage in local_market_data.items():
                result += f"- **{region_name}**: {percentage:.1f}%\n"
            
            result += "\n"
            
            # Add a regional market size breakdown
            result += "## Regional Market Size\n\n"
            
            for region_name, percentage in local_market_data.items():
                regional_size = median_size * (percentage / 100)
                result += f"- **{region_name}**: ${regional_size:.1f} billion\n"
            
            result += "\n"
    
    # Extract additional information from snippets
    if all_snippets:
        additional_info = extract_additional_market_info(industry, region, all_snippets)
        result += additional_info
    
    # If we couldn't find real data, mention this
    if not market_sizes and not growth_rates:
        result += "\n\n*Note: This analysis is based on AI estimation as specific market research data was limited.*"
    
    return result

# SWOT analysis functions
def find_competitors_for_company(company_name: str, location: str = None, industry: str = None) -> List[str]:
    """Find competitors for a company based on industry or location."""
    competitors = []
    
    # For public companies with industry information
    if industry:
        try:
            # Use SERP API if available
            if SERP_API_KEY and SERPAPI_AVAILABLE:
                search_params = {
                    "engine": "google",
                    "q": f"top companies in {industry} industry competitors of {company_name}",
                    "api_key": SERP_API_KEY
                }
                
                search_results = GoogleSearch(search_params).get_dict()
                
                if 'organic_results' in search_results:
                    # Extract potential competitor names from snippets
                    for result in search_results['organic_results'][:3]:
                        if 'snippet' in result:
                            snippet = result['snippet']
                            
                            # Look for company name patterns
                            potential_names = re.findall(r'([A-Z][a-zA-Z\'\-&]+(?: [A-Z][a-zA-Z\'\-&]+)*)', snippet)
                            
                            for name in potential_names:
                                # Filter out common words that might be capitalized
                                if (name not in ['The', 'This', 'These', 'They', 'Their', 'Inc', 'Ltd'] and 
                                    len(name) > 3 and name != company_name and name not in competitors):
                                    competitors.append(name)
                                    
                                    # Limit to 5 competitors
                                    if len(competitors) >= 5:
                                        break
            
            # Fallback to web scraping if no results from SERP API
            if not competitors:
                search_query = f"top companies in {industry} industry competitors"
                search_query = search_query.replace(' ', '+')
                
                url = f"https://www.google.com/search?q={search_query}"
                
                html_content = make_web_request(url)
                if html_content:
                    soup = parse_html(html_content)
                    if soup:
                        for div in soup.select('div.VwiC3b'):
                            text = div.text
                            
                            # Look for company name patterns
                            potential_names = re.findall(r'([A-Z][a-zA-Z\'\-&]+(?: [A-Z][a-zA-Z\'\-&]+)*)', text)
                            
                            for name in potential_names:
                                # Filter out common words that might be capitalized
                                if (name not in ['The', 'This', 'These', 'They', 'Their', 'Inc', 'Ltd'] and 
                                    len(name) > 3 and name != company_name and name not in competitors):
                                    competitors.append(name)
                                    
                                    # Limit to 5 competitors
                                    if len(competitors) >= 5:
                                        break
        except Exception as e:
            logger.warning(f"Error finding competitors by industry: {e}")
    
    # For local businesses, try to get competitors from Google Maps
    elif location and gmaps:
        try:
            # Search for the company using Google Places API
            places_result = gmaps.places(f"{company_name} {location}")
            
            if places_result['status'] == 'OK' and places_result['results']:
                place = places_result['results'][0]
                
                # Get the business type
                business_type = None
                if 'types' in place:
                    for t in place['types']:
                        if t not in ['point_of_interest', 'establishment']:
                            business_type = t
                            break
                
                if business_type:
                    # Search for similar businesses
                    nearby_search = gmaps.places_nearby(
                        location=place['geometry']['location'],
                        radius=5000,  # 5km radius
                        type=business_type
                    )
                    
                    if nearby_search['status'] == 'OK':
                        for nearby_place in nearby_search['results']:
                            name = nearby_place.get('name', '')
                            
                            if name and name != company_name and name not in competitors:
                                competitors.append(name)
                                
                                # Limit to 5 competitors
                                if len(competitors) >= 5:
                                    break
        except Exception as e:
            logger.warning(f"Error finding local competitors: {e}")
    
    return competitors

def generate_swot_from_data(company_name: str, company_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate a SWOT analysis using GPT-4o based on company data."""
    swot_prompt = f"""Perform a detailed SWOT analysis for {company_name} based on the following information:

    Company Information:
    {json.dumps(company_data, indent=2)}
    
    Provide 4-6 specific points for each category (Strengths, Weaknesses, Opportunities, Threats).
    Each point should be detailed and specific to this company, not generic.
    
    Format the response as a JSON with keys: "Strengths", "Weaknesses", "Opportunities", "Threats".
    Each key should have an array of strings as its value.
    
    Use this exact format (ensure it's valid JSON):
    {{
      "Strengths": ["Strength 1", "Strength 2"],
      "Weaknesses": ["Weakness 1", "Weakness 2"],
      "Opportunities": ["Opportunity 1", "Opportunity 2"],
      "Threats": ["Threat 1", "Threat 2"]
    }}
    """
    
    swot_response = gpt4o.invoke(swot_prompt)
    
    try:
        # Parse the JSON response
        content = swot_response.content.strip()
        
        # If response starts with ```json and ends with ```, strip those out
        if content.startswith('```json') and content.endswith('```'):
            content = content[7:-3].strip()
        elif content.startswith('```') and content.endswith('```'):
            content = content[3:-3].strip()
            
        # Parse the JSON response
        swot = json.loads(content)
        
        # Ensure all required keys are present
        required_keys = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
        for key in required_keys:
            if key not in swot:
                swot[key] = []
        
        return swot
    except Exception as e:
        logger.error(f"Error parsing SWOT analysis: {e}")
        return create_fallback_swot(company_name, company_data)

def create_fallback_swot(company_name: str, company_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Create a fallback SWOT analysis when GPT-4o parsing fails."""
    # Determine company type
    if company_data.get('is_public', False):
        return {
            "Strengths": [
                f"Established market position in the {company_data.get('industry', 'industry')}",
                "Brand recognition",
                "Financial resources for growth and investment",
                "Access to capital markets"
            ],
            "Weaknesses": [
                "Potential bureaucracy due to organizational size",
                "Higher operational costs",
                "Slower decision-making processes",
                "Regulatory scrutiny"
            ],
            "Opportunities": [
                "Expansion into new markets",
                "Strategic acquisitions",
                "Product diversification",
                "Digital transformation initiatives"
            ],
            "Threats": [
                "Intense industry competition",
                "Disruptive innovations from startups",
                "Changing regulations",
                "Economic uncertainty"
            ]
        }
    elif 'location' in company_data:  # Local business
        location = company_data.get('location', 'the area')
        return {
            "Strengths": [
                f"Local presence in {location}",
                "Community relationships",
                "Personalized customer service",
                "Adaptability to local market needs"
            ],
            "Weaknesses": [
                "Limited resources compared to larger competitors",
                "Restricted geographic reach",
                "Potentially limited marketing capabilities",
                "Lower economies of scale"
            ],
            "Opportunities": [
                "Growing local customer base",
                "Expanding to nearby regions",
                "Digital marketing to reach new customers",
                "Strategic partnerships with other local businesses"
            ],
            "Threats": [
                "Entry of national/global chains into the local market",
                "Economic downturns affecting the local economy",
                "Changing consumer preferences",
                "Rising operational costs"
            ]
        }
    else:  # Generic
        return {
            "Strengths": [
                "Brand recognition",
                "Product/service quality",
                "Customer loyalty",
                "Experienced management team"
            ],
            "Weaknesses": [
                "Limited resources",
                "Geographic limitations",
                "Gaps in product/service offerings",
                "Operational inefficiencies"
            ],
            "Opportunities": [
                "Market expansion",
                "New product/service development",
                "Strategic partnerships",
                "Digital transformation"
            ],
            "Threats": [
                "Competitive pressure",
                "Changing market trends",
                "Economic uncertainty",
                "Regulatory challenges"
            ]
        }

@tool
def swot_analysis(company_name: str, location: str = None) -> Dict[str, List[str]]:
    """
    Perform a SWOT analysis for a company using real data, including local business data if applicable.
    
    Args:
        company_name: The name of the company to analyze
        location: Optional location to help identify local businesses
    
    Returns:
        Dictionary with Strengths, Weaknesses, Opportunities, and Threats
    """
    logger.info(f"Performing SWOT analysis for: {company_name} in {location if location else 'unknown location'}")
    
    # Get company data
    company_data = get_company_data(company_name, location)
    
    # Get recent news
    try:
        news = search_news(f"{company_name} company news")
        company_data['news'] = news
    except Exception as e:
        logger.error(f"Error getting news: {e}")
        company_data['news'] = f"Unable to retrieve recent news for {company_name}."
    
    # Add location to company data
    if location:
        company_data['location'] = location
    
    # Get competitors if possible
    competitors = find_competitors_for_company(
        company_name, 
        location, 
        company_data.get('industry')
    )
    
    # Add competitors to company data
    if competitors:
        company_data['competitors'] = competitors
    
    # Generate SWOT analysis
    return generate_swot_from_data(company_name, company_data)