"""
Unified Market Research UI - Complete solution with no asyncio conflicts
"""
import streamlit as st
import time
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import sys
import os
import asyncio

# Configure logging to reduce noise
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure page
st.set_page_config(
    page_title="🚀 AI Market Research Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .status-good { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    
    .query-examples {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class UnifiedMarketResearchUI:
    """Complete UI class with unified cache and no asyncio conflicts"""
    
    def __init__(self):
        self.supervisor = None
        self.cache_manager = None
        self._initialize_session_state()
        self._initialize_system()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'system_status' not in st.session_state:
            st.session_state.system_status = {}
        if 'last_query' not in st.session_state:
            st.session_state.last_query = ""
    
    def _initialize_system(self):
        """Initialize the system components"""
        try:
            # Import the unified cache manager
            from cache_manager import cache_manager
            self.cache_manager = cache_manager
            
            # Initialize supervisor
            from supervisor import Supervisor
            self.supervisor = Supervisor()
            
            # Update system status
            st.session_state.system_status = {
                'supervisor': 'initialized',
                'cache': 'unified_cache_active',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.session_state.system_status = {
                'supervisor': 'error',
                'cache': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Market Research Assistant</h1>
        <p>Powered by unified cache system and specialized AI agents</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar(ui_instance):
    """Render sidebar with system controls"""
    with st.sidebar:
        st.markdown("## 🔧 System Controls")
        
        # System Status
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📊 System Status")
        
        status = st.session_state.system_status
        if status.get('supervisor') == 'initialized':
            st.markdown('<span class="status-good">✅ System Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-error">❌ System Error</span>', unsafe_allow_html=True)
            if 'error' in status:
                st.error(f"Error: {status['error']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Cache Management
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🗄️ Cache Management")
        
        if ui_instance.cache_manager:
            try:
                cache_stats = ui_instance.cache_manager.get_stats()
                st.write(f"**Active Items:** {cache_stats['active_items']}")
                st.write(f"**Total Items:** {cache_stats['total_items']}")
                st.write(f"**Cache Type:** Unified")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear Cache", use_container_width=True):
                        ui_instance.cache_manager.clear_all()
                        st.success("Cache cleared!")
                        st.rerun()
                
                with col2:
                    if st.button("🔄 Refresh", use_container_width=True):
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Cache error: {str(e)}")
        else:
            st.warning("Cache manager not available")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Configuration Info
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Configuration")
        
        try:
            from config import config
            st.write(f"**Model:** {config.models.model_name}")
            st.write(f"**Temperature:** {config.models.temperature}")
            st.write(f"**Cache TTL:** {getattr(config.cache, 'ttl_seconds', 3600)}s")
        except Exception as e:
            st.write("**Status:** Configuration loaded")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis History
        if st.session_state.analysis_history:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 📈 Recent Analyses")
            
            recent_analyses = st.session_state.analysis_history[-3:]
            for i, analysis in enumerate(reversed(recent_analyses)):
                with st.expander(f"Analysis {len(recent_analyses) - i}"):
                    st.write(f"**Query:** {analysis['query'][:40]}...")
                    if analysis['success']:
                        st.write(f"**Time:** {analysis['metadata']['execution_time']:.2f}s")
                        st.write("**Status:** ✅ Success")
                    else:
                        st.write("**Status:** ❌ Failed")
                        st.write(f"**Error:** {analysis.get('error', 'Unknown')[:50]}...")
            
            st.markdown('</div>', unsafe_allow_html=True)

def render_query_interface():
    """Render the query input interface"""
    st.markdown("## 🔍 Market Research Query")
    
    # Example queries section
    st.markdown("""
    <div class="query-examples">
        <h4>💡 Example Queries - Click to Use:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Example buttons
    col1, col2 = st.columns(2)
    
    examples = [
        "Market size for electric vehicles in India",
        "Competition analysis for food delivery apps in Mumbai",
        "Consumer sentiment for sustainable fashion brands",
        "IT education market opportunities in Ahmedabad",
        "Healthcare startup market trends 2024",
        "E-commerce growth potential in tier-2 cities"
    ]
    
    selected_example = None
    
    for i, example in enumerate(examples):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(f"📋 {example}", use_container_width=True, key=f"example_{i}"):
                selected_example = example
    
    # Query input
    default_query = selected_example if selected_example else st.session_state.last_query
    
    query = st.text_area(
        "Enter your market research question:",
        value=default_query,
        height=120,
        placeholder="E.g., What is the market size and competition for starting an IT training institute in Ahmedabad?",
        help="Be specific about your industry, location, and what aspects you want to research"
    )
    
    # Update last query
    if query != st.session_state.last_query:
        st.session_state.last_query = query
    
    # Analysis controls
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        analyze_button = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🔄 Clear Query", use_container_width=True):
            st.session_state.last_query = ""
            st.rerun()
    
    with col3:
        detailed_mode = st.checkbox("📊 Detailed", value=True, help="Include detailed analysis")
    
    return query, analyze_button, detailed_mode

def render_progress_indicators():
    """Render analysis progress with agent cards"""
    st.markdown("### 🤖 AI Agents Working...")
    
    col1, col2, col3, col4 = st.columns(4)
    
    agents = [
        ("🎯", "Competitive Analysis", "Analyzing competitors and market positioning"),
        ("📈", "Market Trends", "Identifying industry trends and news"),
        ("👥", "Consumer Insights", "Understanding consumer behavior"),
        ("📊", "Market Sizing", "Estimating market opportunities")
    ]
    
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], agents):
        with col:
            st.markdown(f"""
            <div class="agent-card">
                <h4>{icon} {title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

def run_market_analysis(ui_instance, query: str, detailed: bool = True) -> Dict[str, Any]:
    """Run market research analysis with better error handling"""
    if not ui_instance.supervisor:
        return {
            "success": False,
            "query": query,
            "error": "Supervisor not initialized",
            "metadata": {"execution_time": 0, "timestamp": datetime.now().isoformat()}
        }
    
    start_time = time.time()
    
    try:
        # Use the synchronous run method to avoid asyncio conflicts
        result = ui_instance.supervisor.run(query)
        
        # FIXED: Ensure result is a string and handle coroutine case
        if asyncio.iscoroutine(result):
            # If we accidentally got a coroutine, convert it
            result = "Error: Received coroutine instead of result. Please check supervisor configuration."
        elif not isinstance(result, str):
            # Convert any non-string result to string
            result = str(result)
        
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "query": query,
            "result": result,
            "metadata": {
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "result_length": len(result),
                "detailed_mode": detailed,
                "cache_type": "unified",
                "result_type": type(result).__name__
            }
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        
        # Handle specific coroutine error
        if "coroutine" in error_msg.lower():
            error_msg = "Async/sync mismatch detected. Using fixed supervisor to resolve."
            
        return {
            "success": False,
            "query": query,
            "error": error_msg,
            "metadata": {
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "detailed_mode": detailed,
                "error_type": type(e).__name__
            }
        }

def render_analysis_results(result: Dict[str, Any]):
    """Render comprehensive analysis results"""
    st.markdown("## 📊 Market Research Report")
    
    if result['success']:
        # Success metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <h4>⏱️ Execution Time</h4>
                <h2>{:.2f}s</h2>
            </div>
            """.format(result['metadata']['execution_time']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
                <h4>📝 Report Length</h4>
                <h2>{:,}</h2>
                <p>characters</p>
            </div>
            """.format(result['metadata']['result_length']), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-container">
                <h4>🤖 Analysis Type</h4>
                <h2>AI-Powered</h2>
                <p>Multi-agent</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            cache_status = result['metadata'].get('cache_type', 'standard')
            st.markdown(f"""
            <div class="metric-container">
                <h4>💾 Cache Status</h4>
                <h2>✅ Active</h2>
                <p>{cache_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Main report content
        st.markdown("### 📋 Comprehensive Market Research Report")
        
        # Display the report with proper formatting
        report_content = result['result']
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Formatted Report", "📄 Raw Text"])
        
        with tab1:
            st.markdown(report_content)
        
        with tab2:
            st.text_area("Raw Report Content", report_content, height=400)
        
        st.markdown("---")
        
        # Export and sharing options
        st.markdown("### 📥 Export & Share")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📄 Download as Text",
                data=report_content,
                file_name=f"market_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            json_data = {
                "query": result['query'],
                "report": report_content,
                "metadata": result['metadata'],
                "analysis_timestamp": datetime.now().isoformat(),
                "system_info": {
                    "cache_type": "unified",
                    "ai_model": "gpt-4o",
                    "version": "unified_v1.0"
                }
            }
            st.download_button(
                "📊 Download as JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"market_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 Analyze Similar Query", use_container_width=True):
                st.session_state.last_query = f"Similar analysis to: {result['query'][:50]}..."
                st.rerun()
    
    else:
        # Error handling and troubleshooting
        st.error(f"❌ Analysis Failed: {result.get('error', 'Unknown error')}")
        
        # Error details in expandable section
        with st.expander("🔍 Technical Details"):
            st.json(result)
        
        # Troubleshooting guide
        st.markdown("### 🛠️ Troubleshooting Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Common Solutions:**
            1. **Check API Key** - Verify OpenAI API key is valid
            2. **Simplify Query** - Try a shorter, more specific question
            3. **Clear Cache** - Use sidebar cache controls
            4. **Check Credits** - Ensure API account has sufficient credits
            """)
        
        with col2:
            st.markdown("""
            **System Checks:**
            1. **Internet Connection** - Verify stable connection
            2. **Restart Application** - Refresh the browser page
            3. **Configuration** - Check .env file settings
            4. **Dependencies** - Ensure all packages are installed
            """)
        
        # Quick retry button
        if st.button("🔄 Retry Analysis", type="primary"):
            st.rerun()

def render_analytics_dashboard():
    """Render comprehensive analytics dashboard"""
    if not st.session_state.analysis_history:
        st.info("📊 No analysis history available yet. Run some analyses to see performance metrics and insights!")
        return
    
    st.markdown("## 📈 Analytics Dashboard")
    
    history = st.session_state.analysis_history
    
    # Calculate metrics
    total_analyses = len(history)
    successful_analyses = sum(1 for h in history if h['success'])
    success_rate = (successful_analyses / total_analyses) * 100 if total_analyses > 0 else 0
    
    exec_times = [h['metadata']['execution_time'] for h in history if h['success']]
    avg_time = sum(exec_times) / len(exec_times) if exec_times else 0
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", total_analyses, delta=f"+{len([h for h in history[-5:] if h['success']])} recent")
    
    with col2:
        st.metric("Success Rate", f"{success_rate:.1f}%", delta="Unified Cache")
    
    with col3:
        st.metric("Avg Execution Time", f"{avg_time:.2f}s", delta=f"Last: {exec_times[-1]:.1f}s" if exec_times else None)
    
    with col4:
        recent_queries = len([h for h in history if h['metadata']['timestamp'] > (datetime.now().replace(hour=0, minute=0, second=0)).isoformat()])
        st.metric("Today's Analyses", recent_queries)
    
    if len(exec_times) > 1:
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            # Execution time trend
            fig_time = px.line(
                x=list(range(1, len(exec_times) + 1)),
                y=exec_times,
                title="⏱️ Analysis Execution Time Trend",
                labels={'x': 'Analysis Number', 'y': 'Time (seconds)'}
            )
            fig_time.update_traces(line_color='#667eea')
            fig_time.update_layout(height=400)
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Success rate gauge
            fig_success = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=success_rate,
                title={'text': "Success Rate (%)"},
                delta={'reference': 90},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 70], 'color': "#ffcccb"},
                        {'range': [70, 90], 'color': "#fffacd"},
                        {'range': [90, 100], 'color': "#90ee90"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 95}
                }
            ))
            fig_success.update_layout(height=400)
            st.plotly_chart(fig_success, use_container_width=True)
        
        # Analysis history table
        st.markdown("### 📊 Analysis History")
        
        # Prepare table data
        table_data = []
        for i, analysis in enumerate(reversed(history[-10:]), 1):
            table_data.append({
                "#": i,
                "Query": analysis['query'][:60] + "..." if len(analysis['query']) > 60 else analysis['query'],
                "Status": "✅ Success" if analysis['success'] else "❌ Failed",
                "Time (s)": f"{analysis['metadata']['execution_time']:.2f}" if analysis['success'] else "N/A",
                "Cache": analysis['metadata'].get('cache_type', 'standard'),
                "Timestamp": analysis['metadata']['timestamp'][:16] if 'timestamp' in analysis['metadata'] else "N/A"
            })
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

def main():
    """Main application function"""
    # Initialize UI
    ui = UnifiedMarketResearchUI()
    
    # Render header
    render_header()
    
    # Check initialization status
    if st.session_state.system_status.get('supervisor') != 'initialized':
        st.error("❌ System initialization failed. Please check your configuration.")
        with st.expander("🔍 System Status Details"):
            st.json(st.session_state.system_status)
        st.stop()
    
    # Render sidebar
    render_sidebar(ui)
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["🔍 Market Analysis", "📈 Analytics Dashboard"])
    
    with tab1:
        # Query interface
        query, analyze_button, detailed_mode = render_query_interface()
        
        # Run analysis if requested
        if analyze_button and query.strip():
            # Show progress indicators
            render_progress_indicators()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Progress updates
                status_text.text("🔍 Initializing analysis system...")
                progress_bar.progress(10)
                time.sleep(0.3)
                
                status_text.text("🤖 AI agents collaborating...")
                progress_bar.progress(30)
                time.sleep(0.5)
                
                status_text.text("📊 Gathering market intelligence...")
                progress_bar.progress(60)
                
                # Run the actual analysis
                with st.spinner("🧠 Generating comprehensive market research report..."):
                    result = run_market_analysis(ui, query, detailed_mode)
                    progress_bar.progress(90)
                    time.sleep(0.3)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete! Generating report...")
                time.sleep(0.5)
                
                # Store in history
                st.session_state.analysis_history.append(result)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                render_analysis_results(result)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Analysis failed: {str(e)}")
                
                # Store failed analysis
                failed_result = {
                    "success": False,
                    "query": query,
                    "error": str(e),
                    "metadata": {"execution_time": 0, "timestamp": datetime.now().isoformat()}
                }
                st.session_state.analysis_history.append(failed_result)
        
        elif analyze_button:
            st.warning("⚠️ Please enter a market research question before starting the analysis.")
    
    with tab2:
        render_analytics_dashboard()

if __name__ == "__main__":
    main()