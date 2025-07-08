import streamlit as st
import os
from dotenv import load_dotenv
from supervisor import Supervisor
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Global theme adjustments */
    :root {
        --primary-color: #2E86C1;
        --accent-color: #F39C12;
        --background-color: #F8F9F9;
        --sidebar-color: #EBF5FB;
        --text-color: #2C3E50;
        --header-color: #1A5276;
        --card-bg-color: white;
        --success-color: #27AE60;
        --error-color: #E74C3C;
        --warning-color: #F39C12;
        --info-color: #3498DB;
    }
    
    /* Page background */
    .stApp {
        background-color: var(--background-color);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-color);
        border-right: 1px solid rgba(0,0,0,0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--header-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    
    h1 {
        font-size: 2.5rem !important;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--primary-color);
        margin-bottom: 1.5rem;
    }
    
    h2 {
        font-size: 1.8rem !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(0,0,0,0.1);
    }
    
    h3 {
        font-size: 1.4rem !important;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }
    
    /* Text area input */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 12px;
        font-size: 1.05rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 2px 8px rgba(46,134,193,0.3);
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Primary button */
    .stButton button[kind="primary"] {
        background-color: var(--primary-color);
    }
    
    /* Secondary button */
    .stButton button:not([kind="primary"]) {
        border: 1px solid var(--text-color);
        color: var(--text-color);
    }
    
    /* Cards for content */
    .card {
        background-color: var(--card-bg-color);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        border-top: 4px solid var(--primary-color);
    }
    
    /* Status indicators */
    .status-container {
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .status-processing {
        background-color: rgba(52, 152, 219, 0.1);
        border-left: 4px solid var(--info-color);
    }
    
    .status-success {
        background-color: rgba(39, 174, 96, 0.1);
        border-left: 4px solid var(--success-color);
    }
    
    .status-error {
        background-color: rgba(231, 76, 60, 0.1);
        border-left: 4px solid var(--error-color);
    }
    
    /* Result container styling */
    .result-container {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Download button */
    .download-button {
        margin-top: 1.5rem;
        text-align: center;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(0,0,0,0.1);
        color: #7F8C8D;
        font-size: 0.9rem;
    }
    
    /* Checkboxes in sidebar */
    .stCheckbox label p {
        font-weight: 500;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    /* Tags */
    .tag {
        display: inline-block;
        background-color: rgba(46,134,193,0.1);
        color: var(--primary-color);
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .tag-primary {
        background-color: rgba(46,134,193,0.1);
        color: var(--primary-color);
    }
    
    .tag-accent {
        background-color: rgba(243,156,18,0.1);
        color: var(--accent-color);
    }
    
    /* Logo */
    .logo-container {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .logo {
        width: 100px;
        height: 100px;
        margin-bottom: 0.5rem;
    }
    
    .logo-text {
        font-weight: 600;
        color: var(--primary-color);
        font-size: 1.2rem;
    }
    
    /* Improved markdown rendering */
    .research-report h1, .research-report h2, .research-report h3 {
        color: var(--header-color);
    }
    
    .research-report blockquote {
        border-left: 4px solid var(--primary-color);
        padding-left: 1rem;
        color: #555;
        font-style: italic;
    }
    
    .research-report ul, .research-report ol {
        padding-left: 1.5rem;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# Set page config
st.set_page_config(
    page_title="Market Research AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize supervisor
@st.cache_resource
def get_supervisor():
    return Supervisor()

# Helper function to create styled cards
def st_card(title, content, icon=None):
    if icon:
        header = f"<h3><span style='margin-right:10px;'>{icon}</span>{title}</h3>"
    else:
        header = f"<h3>{title}</h3>"
        
    st.markdown(f"""
    <div class="card">
        {header}
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

# Helper function to create tags
def st_tag(text, tag_type="primary"):
    return f'<span class="tag tag-{tag_type}">{text}</span>'

# Format current date
current_date = datetime.now().strftime("%B %d, %Y")

# Sidebar design with branded header
st.sidebar.markdown("""
<div class="logo-container">
    <img src="https://img.icons8.com/fluency/96/000000/economic-improvement.png" class="logo">
    <div class="logo-text">Market Research AI</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"<div style='text-align:center; margin-bottom:1.5rem;'>{current_date}</div>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin-bottom:1.5rem;'>", unsafe_allow_html=True)

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    # API key input
    st.markdown("""
    <div class="card">
        <h3>🔑 API Key Required</h3>
        <p>Please enter your OpenAI API key to continue.</p>
    </div>
    """, unsafe_allow_html=True)
    
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API key set! Please refresh the page.")
        st.experimental_rerun()
    st.stop()

# Initialize supervisor after API key check
try:
    supervisor = get_supervisor()
except Exception as e:
    st.error(f"Error initializing the market research system: {str(e)}")
    st.error("Please check your configuration and try again.")
    st.stop()

# Agent selection with improved UI
st.sidebar.markdown("""
<h3 style='color: var(--header-color); margin-bottom:1rem;'>
    <span style='margin-right:8px;'>🔍</span>Research Components
</h3>
""", unsafe_allow_html=True)

use_competitive = st.sidebar.checkbox("Competitive Analysis", value=True, 
    help="Analyzes competitors, their strengths, weaknesses, and market positioning")
use_consumer = st.sidebar.checkbox("Consumer Insights", value=True, 
    help="Analyzes consumer behavior, preferences, and sentiment")
use_market_size = st.sidebar.checkbox("Market Sizing", value=True, 
    help="Estimates market size, growth projections, and segmentation")
use_trends = st.sidebar.checkbox("Market Trends", value=True, 
    help="Identifies current and emerging market trends")

# Location setting in sidebar with icon
st.sidebar.markdown("""
<h3 style='color: var(--header-color); margin-top:1.5rem; margin-bottom:1rem;'>
    <span style='margin-right:8px;'>📍</span>Target Location
</h3>
""", unsafe_allow_html=True)

location = st.sidebar.text_input("Location:", 
                                placeholder="e.g., San Francisco, Mumbai, etc.",
                                help="Specify a location for more targeted research")

# Analysis depth with icon
st.sidebar.markdown("""
<h3 style='color: var(--header-color); margin-top:1.5rem; margin-bottom:1rem;'>
    <span style='margin-right:8px;'>⚙️</span>Research Settings
</h3>
""", unsafe_allow_html=True)

research_depth = st.sidebar.select_slider(
    "Research Depth", 
    options=["Basic", "Standard", "Comprehensive"], 
    value="Standard",
    help="Determines the depth and detail of the analysis"
)

# Industry selection with improved styling
industries = [
    "Select Industry (Optional)",
    "Technology", 
    "Healthcare", 
    "Finance", 
    "Retail", 
    "Education",
    "Food & Beverage",
    "Manufacturing",
    "Entertainment",
    "Real Estate",
    "Transportation",
    "Energy"
]
industry = st.sidebar.selectbox("Industry:", industries)

# Display active components with tags
st.sidebar.markdown("<hr style='margin-top:1.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='margin-bottom:1rem;'>Active Components</h4>", unsafe_allow_html=True)

# Create tag HTML
active_tags = ""
if use_competitive:
    active_tags += st_tag("Competitive", "primary")
if use_consumer:
    active_tags += st_tag("Consumer", "accent")
if use_market_size:
    active_tags += st_tag("Market Size", "primary")
if use_trends:
    active_tags += st_tag("Trends", "accent")

st.sidebar.markdown(active_tags, unsafe_allow_html=True)

st.sidebar.markdown("<hr style='margin-top:1.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)

# Info box with styling
st.sidebar.markdown("""
<div style="background-color:rgba(52, 152, 219, 0.1); padding:1rem; border-radius:8px; border-left:4px solid #3498DB;">
    <h4 style="color:#2980B9; margin-top:0;">About This Tool</h4>
    <p style="margin-bottom:0;">This AI-powered application analyzes market conditions, competitors, consumer behavior, and trends to provide comprehensive market research insights.</p>
</div>
""", unsafe_allow_html=True)

# Main content with modern styling
st.markdown("""
<h1 class="animate-fade-in">📊 Market Research Intelligence</h1>
""", unsafe_allow_html=True)

# Introduction card
st_card(
    "AI-Powered Market Research",
    """
    <p>Get comprehensive market research insights powered by advanced AI. Enter your research question 
    below to analyze market size, competitive landscape, consumer insights, and emerging trends.</p>
    
    <p>Our system combines data from multiple specialized research agents to create a holistic view of your market opportunity.</p>
    """,
    "🤖"
)

# Example queries in an expander with better styling
with st.expander("💡 See example research questions"):
    examples = [
        "What is the market size for electric vehicles in Europe?",
        "Who are the main competitors for Netflix and what are their strengths?",
        "What are consumer sentiments about plant-based meat alternatives?",
        "I want to start a coffee shop in San Francisco, what is the market size and how to start it?",
        "Analyze the fitness app market and provide recommendations for a new entrant",
        "What are the latest trends in sustainable fashion?"
    ]
    
    for example in examples:
        st.markdown(f"• {example}")

# User input with enhanced styling
st.markdown("""
<h3 style='margin-top:1.5rem;'>Research Question</h3>
""", unsafe_allow_html=True)

user_query = st.text_area("What would you like to research?", 
                        height=100, 
                        placeholder="E.g., I want to start an IT class in Ahmedabad, what is the market size and how should I approach it?")

# Format the query with additional context from sidebar
def format_query(base_query, location, industry, depth):
    # Start with the base query
    formatted_query = base_query
    
    # Add location if provided
    if location and location.strip():
        if "location" not in base_query.lower() and location.lower() not in base_query.lower():
            formatted_query += f" in {location}"
    
    # Add industry if selected
    if industry and industry != "Select Industry (Optional)":
        if industry.lower() not in base_query.lower():
            formatted_query += f" Focus on the {industry} industry."
    
    # Add depth requirements
    if depth == "Comprehensive":
        formatted_query += " Please provide comprehensive and detailed analysis with in-depth market data."
    elif depth == "Basic":
        formatted_query += " Please provide a concise overview with key highlights only."
    
    # Add which components to include
    components = []
    if use_competitive:
        components.append("competitive analysis")
    if use_consumer:
        components.append("consumer insights")
    if use_market_size:
        components.append("market sizing")
    if use_trends:
        components.append("market trends")
    
    if components:
        formatted_query += f" Include {', '.join(components)}."
    
    return formatted_query

# Create container for results
result_container = st.container()

# Button row with improved styling
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    research_button = st.button("🔍 Conduct Research", type="primary", disabled=not user_query, use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear Results", use_container_width=True)
    
# Display parameters when query exists
if user_query:
    formatted_query = format_query(user_query, location, industry, research_depth)
    with col3:
        with st.expander("View research parameters"):
            params_md = f"""
            **Query:** {formatted_query}\n
            **Research depth:** {research_depth}\n
            **Components:** {"Competitive" if use_competitive else "❌"} · {"Consumer" if use_consumer else "❌"} · {"Market Size" if use_market_size else "❌"} · {"Trends" if use_trends else "❌"}\n
            """
            if location:
                params_md += f"**Location:** {location}\n"
            if industry and industry != "Select Industry (Optional)":
                params_md += f"**Industry:** {industry}"
            
            st.markdown(params_md)

if clear_button:
    if 'research_results' in st.session_state:
        st.session_state.research_results = None
        st.experimental_rerun()

# Handle research button click
if research_button:
    # Format the query
    full_query = format_query(user_query, location, industry, research_depth)
    
    # Create session state to store results if they don't exist
    if 'research_results' not in st.session_state:
        st.session_state.research_results = None
    
    # Custom styled status indicators
    with result_container:
        status_container = st.empty()
        status_container.markdown("""
        <div class="status-container status-processing">
            <h4 style="margin-top:0;">🔄 Conducting Market Research</h4>
            <p id="status-message">Initializing research process...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Run the analysis
        try:
            # Update status
            status_container.markdown("""
            <div class="status-container status-processing">
                <h4 style="margin-top:0;">🔄 Conducting Market Research</h4>
                <p>Analyzing market size and competitive landscape...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Run the supervisor
            response = supervisor.run(full_query)
            
            # Update status (success)
            status_container.markdown("""
            <div class="status-container status-success">
                <h4 style="margin-top:0;">✅ Research Complete</h4>
                <p>Market research has been successfully completed!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Store in session state
            st.session_state.research_results = response
            
        except Exception as e:
            # Update status (error)
            status_container.markdown(f"""
            <div class="status-container status-error">
                <h4 style="margin-top:0;">❌ Research Error</h4>
                <p>An error occurred: {str(e)}</p>
                <p>Please try again with a different query or check your API key configuration.</p>
            </div>
            """, unsafe_allow_html=True)

# Display results if they exist
with result_container:
    if st.session_state.get('research_results') is not None:
        response = st.session_state.research_results
        
        # Display the entire response in a styled container
        st.markdown("""
        <div class="result-container animate-fade-in">
            <h2 style="margin-top:0;">📑 Market Research Report</h2>
            <div class="research-report">
        """, unsafe_allow_html=True)
        
        # Display the markdown content
        st.markdown(response)
        
        # Close the container
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a styled download button
        st.markdown("""
        <div class="download-button">
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="📥 Download Complete Report",
            data=response,
            file_name=f"market_research_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True
        )
        
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)

# Footer with modern styling
st.markdown("""
<div class="footer">
    <p>Market Research AI © 2025 | Powered by AI Agents</p>
    <p style="font-size:0.8rem; margin-top:0.5rem;">Combining specialized agents for competitive analysis, consumer insights, market sizing, and trend identification</p>
</div>
""", unsafe_allow_html=True)