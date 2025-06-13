import streamlit as st

# Configure page - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="🔬 Sentiment Analysis Dashboard | Tech Titanians",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/mooncakeSG/Sentiment-Analysis-',
        'Report a bug': 'https://github.com/mooncakeSG/Sentiment-Analysis-/issues/1',
        'About': "# Sentiment Analysis Dashboard\nBuilt by the Tech Titanians\n\nAnalyze sentiment in text data using state-of-the-art AI models."
    }
)

# Removed duplicate CSS block - using enhanced CSS later in the file

from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.let_it_rain import rain
from streamlit_extras.grid import grid
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Comprehensive "How to Use" will be in the main sidebar below

# Initialize loading state
if 'app_loaded' not in st.session_state:
    st.session_state.app_loaded = False

# Show loading animations only during initial load
if not st.session_state.app_loaded:
    # Create loading container
    loading_container = st.container()
    
    with loading_container:
        # Load and display animation
        lottie_sentiment = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_qp1q7mct.json")
        
        # Display loading animation
        st_lottie(lottie_sentiment, height=160, key="loading_anim")
        
        # Loading text with progress
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h3 style="color: #4F46E5; margin-bottom: 1rem;">🔬 Loading Sentiment Analysis Dashboard</h3>
            <p style="color: #6B7280;">Initializing AI models and components...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate loading progress
        import time
        loading_steps = [
            "Loading sentiment analysis models...",
            "Initializing keyword extraction...",
            "Setting up visualizations...",
            "Preparing user interface...",
            "Dashboard ready!"
        ]
        
        for i, step in enumerate(loading_steps):
            status_text.text(step)
            progress_bar.progress((i + 1) / len(loading_steps))
            time.sleep(0.5)  # Adjust timing as needed
        
        # Rain animation during loading
        rain(
            emoji="💡",
            font_size=28,
            falling_speed=5,
            animation_length="2s",  # Limited duration for loading
        )
        
        # Mark app as loaded
        st.session_state.app_loaded = True
        
        # Clear loading components and rerun
        loading_container.empty()
        st.rerun()


import streamlit as st
import pandas as pd
import json
import tempfile
from utils import (
    analyze_sentiment,
    extract_keywords,
    export_to_pdf,
    get_download_link,
    explain_sentiment,
    handle_followup_question,
    validate_text_input,
    validate_file_content,
    safe_sentiment_analysis,
    safe_keyword_extraction,
    display_error_with_help,
    display_success_with_details,
    display_warning_with_action,
    validate_and_process_file,
    MODEL_LIMITATIONS
)
from optimization import (
    ModelManager,
    BatchProcessor,
    VisualizationOptimizer,
    optimize_memory_usage,
    handle_errors,
    timed_cache
)
from visualizations import (
    create_sentiment_distribution, 
    create_confidence_chart, 
    create_keyword_importance,
    generate_wordcloud
)
from datetime import datetime
import plotly.express as px

# Initialize models with caching
@st.cache_resource
def initialize_models():
    """Initialize and cache models for better performance."""
    return {
        'sentiment': ModelManager.get_model("sentiment"),
        'keyword': ModelManager.get_model("keyword")
    }

# Cache expensive computations
@st.cache_data(ttl=3600)
def cached_sentiment_analysis(text: str):
    """Cache sentiment analysis results."""
    return analyze_sentiment(text)

@st.cache_data(ttl=3600)
def cached_keyword_extraction(text: str):
    """Cache keyword extraction results."""
    return extract_keywords(text)

@st.cache_data(ttl=3600)
def cached_visualization(data: pd.DataFrame, viz_type: str, **kwargs):
    """Cache visualization results."""
    return VisualizationOptimizer.create_visualization(data, viz_type, **kwargs)

# Page configuration already set at the top of the file

# Adaptive CSS for Light & Dark Modes
st.markdown("""
<style>
    /* Font & Reset */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    /* CSS Variables for Theme Colors */
    :root {
        --bg1: #F8FAFC;
        --bg2: #E0E7FF;
        --bg3: #FFFFFF;
        --text-primary: #111827;
        --text-secondary: #6B7280;
        --border-color: #E5E7EB;
        --card-bg: #FFFFFF;
        --input-bg: #FFFFFF;
        --shadow-light: rgba(0, 0, 0, 0.1);
        --shadow-medium: rgba(0, 0, 0, 0.08);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
    }

    /* Adapt background/text based on system theme */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg1: #1f2937;
            --bg2: #374151;
            --bg3: #1f2937;
            --text-primary: #f8fafc;
            --text-secondary: #d1d5db;
            --border-color: #374151;
            --card-bg: #1f2937;
            --input-bg: #1f2937;
            --shadow-light: rgba(0, 0, 0, 0.3);
            --shadow-medium: rgba(0, 0, 0, 0.2);
        }
        
        html, body {
            background-color: #0f172a !important;
            color: #f8fafc !important;
        }
        
        .main > div {
            background-color: #1e293b !important;
            color: #f8fafc !important;
        }
        
        .stTextArea textarea,
        .stTextInput input,
        .stSelectbox div,
        .stDataFrame,
        .stFileUploader {
            background-color: var(--input-bg) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-color) !important;
        }
        
        .stMetricLabel, .stMetricValue {
            color: #f8fafc !important;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background: var(--card-bg) !important;
            border-color: var(--border-color) !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary) !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: var(--bg2) !important;
            color: var(--text-primary) !important;
        }
    }

    /* Light mode defaults */
    @media (prefers-color-scheme: light) {
        html, body {
            background-color: #F9FAFB;
            color: #111827;
        }
        
        .main > div {
            background-color: #F9FAFB;
            color: #111827;
        }
    }

    /* Global styling */
    .main > div {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Enhanced text area styling */
    .stTextArea > div > div > textarea {
        background-color: var(--input-bg);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        font-size: 1rem;
        line-height: 1.6;
        box-shadow: 0 1px 3px var(--shadow-light);
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #4F46E5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        outline: none;
    }

    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%);
    }

    /* Download button improvements */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(5, 150, 105, 0.2);
    }

    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3);
        background: linear-gradient(135deg, #047857 0%, #065F46 100%);
    }

    /* Metric container styling */
    .metric-container {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--bg1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px var(--shadow-medium);
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }

    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px var(--shadow-light);
    }

    /* Enhanced hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        box-shadow: 0 10px 40px rgba(79, 70, 229, 0.2);
        border-radius: 20px;
        padding: 3rem 2rem 2rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><polygon fill="rgba(255,255,255,0.05)" points="0,0 1000,300 1000,1000 0,700"/></svg>');
        pointer-events: none;
    }

    .hero-banner h1 {
        color: #fff;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.25rem;
        font-weight: 400;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }

    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--card-bg);
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 2px 8px var(--shadow-medium);
        border: 1px solid var(--border-color);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        background-color: transparent;
        border: none;
        color: var(--text-secondary);
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--bg1);
        color: var(--text-primary);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #FFFFFF;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);
    }

    /* Card-style containers */
    .analysis-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--bg1) 100%);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 16px var(--shadow-medium);
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        transition: all 0.2s ease;
    }

    .analysis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px var(--shadow-light);
    }

    /* Enhanced file uploader */
    .stFileUploader {
        border: 2px dashed #4F46E5;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: var(--bg1);
        transition: all 0.2s ease;
    }

    .stFileUploader:hover {
        border-color: #7C3AED;
        background: var(--bg2);
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        border-radius: 10px;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 3rem 0 2rem 0;
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 500;
        background: linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%);
        border-radius: 20px 20px 0 0;
        margin-top: 3rem;
        border-top: 1px solid var(--border-color);
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .hero-banner h1 {
            font-size: 2rem;
        }
        .hero-subtitle {
            font-size: 1rem;
        }
        .main > div {
            padding: 1rem;
        }
        .analysis-card {
            padding: 1rem;
        }
    }

    /* Loading animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* Hide default Streamlit elements but keep menu */
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Enhanced Header with better layout
st.markdown("""
<div class="hero-banner">
    <h1>🔬 Sentiment Analysis Dashboard</h1>
    <p class="hero-subtitle">Analyze sentiment in text data using state-of-the-art natural language processing</p>
    <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem; font-weight: 500;">
            AI-Powered Analysis
        </span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem; font-weight: 500;">
            Interactive Visualizations
        </span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem; font-weight: 500;">
            Batch Processing
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-container" style="text-align: center;">
        <h3 style="color: #4F46E5; margin: 0; font-size: 1.2rem; font-weight: 600;">Fast Analysis</h3>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Results in <2 seconds</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container" style="text-align: center;">
        <h3 style="color: #059669; margin: 0; font-size: 1.2rem; font-weight: 600;">High Accuracy</h3>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.9rem;">92%+ benchmark</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-container" style="text-align: center;">
        <h3 style="color: #D97706; margin: 0; font-size: 1.2rem; font-weight: 600;">Deep Insights</h3>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.9rem;">5-class sentiment</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-container" style="text-align: center;">
        <h3 style="color: #7C3AED; margin: 0; font-size: 1.2rem; font-weight: 600;">Export Ready</h3>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.9rem;">PDF, CSV, JSON</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Title and description
st.markdown("""
<div style='text-align: center; color: #6c757d; margin-bottom: 2rem;'>
    Analyze sentiment in text data using state-of-the-art natural language processing.
    Get insights from customer reviews, social media posts, or any text content.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Dashboard Info header at the top with darker font
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.2); margin: 0 0 1rem 0;">
        <h2 style="color: #2D3748; margin: 0; font-size: 1.5rem; font-weight: 700;">🔬 Dashboard Info</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced "How to Use This App" section
    with st.expander("📘 How to Use This App", expanded=True):
        st.markdown("""
        ### Quick Start Guide
        
        **Single Text Analysis:**
        1. Go to **"Single Text Analysis"** tab
        2. Type or paste your text in the text area
        3. Click **"Analyze Sentiment"** button
        4. View results and ask follow-up questions
        
        **Batch Analysis:**
        1. Go to **"Batch Analysis"** tab  
        2. Upload a **CSV** or **TXT** file
        3. Review the analysis results table
        4. Download reports in multiple formats
        
        **Comparative Analysis:**
        1. Go to **"Comparative Analysis"** tab
        2. Enter 2-5 texts you want to compare
        3. Run analysis to see side-by-side comparison
        4. Get insights and recommendations
        
        ### Best Practices
        • Use **natural language** for best results
        • Texts between **10-500 words** work optimally  
        • Try the **sample packs** for quick testing
        • Check **confidence scores** for reliability
        """)
    
    # Technologies used section
    st.markdown("### 🛠️ Technologies")
    st.info("""
    🤖 **Hugging Face Transformers** - Advanced sentiment analysis using BERT models  
    🔑 **KeyBERT Algorithm** - Intelligent keyword and phrase extraction  
    🎨 **Streamlit Framework** - Interactive web application interface  
    📊 **Plotly Charts** - Dynamic visualizations and charts
    """)
    
    # Quick stats and features
    st.markdown("### ✨ Key Features")
    st.success("""
    📝 **5-Class Sentiment Scale**: Very Negative → Very Positive  
    🎯 **High Accuracy**: 92%+ accuracy on benchmark datasets  
    ⚡ **Lightning Fast**: Analysis completed in <2 seconds  
    🔄 **Batch Processing**: Handle up to 10,000 texts at once  
    🌍 **Multi-language**: Supports English and other major languages  
    📊 **Rich Visualizations**: Interactive charts and word clouds  
    💾 **Export Options**: PDF, CSV, and JSON download formats
    """)
    
    # Model Limitations
    with st.expander("⚠️ Model Limitations", expanded=False):
        st.markdown("### 📋 Specifications")
        st.write(f"• **Max Length**: {MODEL_LIMITATIONS['max_text_length']} tokens")
        st.write(f"• **Confidence**: >{MODEL_LIMITATIONS['confidence_threshold']}")
        st.write(f"• **Languages**: {', '.join(MODEL_LIMITATIONS['supported_languages'])}")
        
        st.markdown("### 🚨 Known Issues")
        for limitation in MODEL_LIMITATIONS['known_limitations']:
            st.write(f"• {limitation}")
    
    # Enhanced Support section
    st.markdown("---")
    st.markdown("### Need Help?")
    
    help_col1, help_col2 = st.columns(2)
    with help_col1:
        if st.button("Quick Help", use_container_width=True):
            st.success("Check the **'How to Use This App'** section at the top!")
    
    with help_col2:
        if st.button("Report Issue", use_container_width=True):
            st.info("[Report bugs on GitHub](https://github.com/mooncakeSG/Sentiment-Analysis-/issues/1)")
    
    # Additional help options
    help_col3, help_col4 = st.columns(2) 
    with help_col3:
        if st.button("Best Practices", use_container_width=True):
            st.info("""
            **Guidelines:**
            • Keep texts natural and conversational
            • Check confidence scores (>70% is reliable)
            • Use sample packs to explore features
            • Export results for further analysis
            """)
    
    with help_col4:
        if st.button("Examples", use_container_width=True):
            st.success("""
            **Sample Texts:**
            • "I love this product! It's amazing."
            • "The service was terrible and slow."
            • "It's okay, nothing special though."
            • "Best purchase I've ever made!"
            """)
    
    # Footer with credits
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.7); font-size: 0.8rem;">
        Made ❤️ by<br><strong>Tech Titanians</strong>
    </div>
    """, unsafe_allow_html=True)

# Main content
tab1, tab2, tab3 = st.tabs(["📝 Single Text Analysis", "📚 Batch Analysis", "🔄 Comparative Analysis"])

with tab1:
    # Enhanced single text input section
    st.markdown("""
    <div class="analysis-card">
        <h3 style="color: #4F46E5; margin-top: 0;">Single Text Analysis</h3>
        <p style="color: #6B7280; margin-bottom: 1.5rem;">Enter any text to analyze its sentiment using our AI-powered model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced text input with better styling
    text_input = st.text_area(
        "Enter text to analyze:", 
        height=150,
        placeholder="Example: 'I love this new product! It works perfectly and exceeded my expectations.'",
        help="**Tips for better analysis:**\n\n• Use natural language text\n• Minimum 3 characters, maximum 5000\n• Avoid excessive special characters\n• Multiple sentences provide better context",
        label_visibility="visible"
    )
    
    # Add character counter and analyze button
    analyze_button_disabled = True
    if text_input:
        char_count = len(text_input)
        if char_count < 3:
            st.error(f"Too short: {char_count}/3 minimum characters")
        elif char_count > 5000:
            st.error(f"Too long: {char_count}/5000 maximum characters")
        else:
            st.success(f"Ready for analysis: {char_count} characters")
            analyze_button_disabled = False
    
    # Analyze button with improved styling
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button(
            "Analyze Sentiment", 
            use_container_width=True, 
            disabled=analyze_button_disabled,
            type="primary",
            help="Click to analyze the sentiment of your text" if not analyze_button_disabled else "Enter valid text to enable analysis"
        )
    
    # Check if we have stored results in session state
    if 'analysis_results' in st.session_state and 'current_text' in st.session_state:
        if st.session_state.current_text == text_input:
            # Use stored results
            result = st.session_state.analysis_results['result']
            keywords = st.session_state.analysis_results['keywords']
            explanation = st.session_state.analysis_results['explanation']
            
            # Set flags to show results
            show_results = True
        else:
            # Text changed, clear stored results
            if 'analysis_results' in st.session_state:
                del st.session_state.analysis_results
            if 'current_text' in st.session_state:
                del st.session_state.current_text
            show_results = False
    else:
        show_results = False

    if text_input and analyze_clicked:
        # Validate input first
        validation_error = validate_text_input(text_input)
        if validation_error:
            display_error_with_help(validation_error, "validation")
        else:
            try:
                with st.spinner("🔍 Analyzing text..."):
                    # Use safe analysis functions
                    result = safe_sentiment_analysis(text_input)
                    
                    # Check for analysis errors
                    if 'error' in result:
                        display_error_with_help(result['error'], "processing")
                    else:
                        # Show warning if present
                        if 'warning' in result:
                            st.warning(result['warning'])
                        
                        # Extract keywords safely
                        keywords = safe_keyword_extraction(text_input)
                        if not keywords:
                            st.info("ℹ️ No significant keywords were extracted from this text.")
                        
                        # Generate explanation safely
                        try:
                            explanation = explain_sentiment(text_input, result)
                        except Exception as e:
                            st.warning(f"⚠️ Could not generate detailed explanation: {str(e)}")
                            explanation = {'reliability': 'Unknown', 'limitations': []}
                        
                        # Store results in session state
                        st.session_state.analysis_results = {
                            'result': result,
                            'keywords': keywords,
                            'explanation': explanation
                        }
                        st.session_state.current_text = text_input
                        show_results = True
                        
                        # Clear any previous follow-up answers when new analysis is performed
                        st.session_state.custom_answer = None
                        st.session_state.custom_question_asked = None
                        st.session_state.predefined_answer = None
                        st.session_state.predefined_question_type = None
            
            except Exception as e:
                display_error_with_help(f"Unexpected error occurred: {str(e)}", "general")
                st.stop()
    
    # Display results if we have them (either from fresh analysis or stored in session state)
    if show_results and 'result' in locals():
        # Enhanced results display with mobile-friendly layout
        st.markdown("---")
        st.markdown("### Analysis Results")
        
        # Create responsive columns
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            # Sentiment result card
            sentiment_color_map = {
                "Very Positive": ("#059669", "🎉"),
                "Positive": ("#10B981", "😊"), 
                "Neutral": ("#6B7280", "😐"),
                "Negative": ("#EF4444", "😔"),
                "Very Negative": ("#DC2626", "😢")
            }
            
            color, emoji = sentiment_color_map.get(result['sentiment'], ("#6B7280", "🤖"))
            
            st.markdown(f"""
            <div class="analysis-card" style="text-align: center; border: 2px solid {color};">
                <h2 style="color: {color}; margin: 0; font-size: 3rem;">{emoji}</h2>
                <h3 style="color: {color}; margin: 0.5rem 0;">{result['sentiment']}</h3>
                <p style="color: var(--text-secondary); margin: 0; font-size: 1.1rem;">
                    Confidence: <strong>{result['confidence']:.1%}</strong>
                </p>
                <div style="background: {color}; height: 4px; border-radius: 2px; 
                           width: {result['confidence']*100}%; margin: 1rem auto 0 auto;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Analysis details
            st.markdown("#### Analysis Details")
            st.info(f"""
            **Use Case**: {result.get('use_case', 'General Analysis')}  
            **Model**: BERT Multilingual  
            **Processing Time**: <2 seconds  
            **Reliability**: {explanation.get('reliability', 'High')}
            """)
        
        with col2:
            # Keywords section
            st.markdown("#### Key Insights")
            
            if keywords:
                # Display keywords as styled tags with improved approach
                st.markdown("**Key Words & Phrases:**")
                
                # Create keyword display with better HTML structure
                keywords_to_show = keywords[:6]  # Limit to 6 for better display
                
                # Use columns for better layout
                if len(keywords_to_show) > 0:
                    cols = st.columns(min(len(keywords_to_show), 3))
                    for i, keyword in enumerate(keywords_to_show):
                        with cols[i % 3]:
                            # Use a simpler, more reliable approach
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); 
                                color: white; 
                                padding: 0.5rem 1rem; 
                                border-radius: 20px; 
                                margin: 0.25rem 0; 
                                text-align: center; 
                                font-size: 0.85rem;
                                font-weight: 500;
                                box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);
                            ">
                                {keyword}
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #FEF3C7; border-radius: 12px; padding: 1.5rem; border: 1px solid #FCD34D;">
                    <p style="margin: 0; color: #92400E;">No significant keywords detected in this text.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Limitations if any
            if explanation.get('limitations'):
                st.markdown("#### Analysis Limitations")
                limitations_html = ""
                for limitation in explanation['limitations']:
                    limitations_html += f"• {limitation}<br>"
                
                st.markdown(f"""
                <div style="background: #FEF3C7; border-radius: 12px; padding: 1rem; border: 1px solid #FCD34D;">
                    <p style="margin: 0; color: #92400E; font-size: 0.9rem;">{limitations_html}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Follow-up Questions Section - RESTORED FUNCTIONALITY
        st.markdown("---")
        st.markdown("### Ask Follow-up Questions")
        st.markdown("Get deeper insights about your analysis results with AI-powered explanations.")
        
        # Predefined question buttons
        follow_col1, follow_col2, follow_col3 = st.columns(3)
        
        # Initialize session state for predefined answers
        if 'predefined_answer' not in st.session_state:
            st.session_state.predefined_answer = None
        if 'predefined_question_type' not in st.session_state:
            st.session_state.predefined_question_type = None
        
        with follow_col1:
            if st.button("Why this sentiment?", use_container_width=True, key="why_sentiment"):
                with st.spinner("Generating explanation..."):
                    explanation_result = handle_followup_question("Why was this labeled this way?", text_input, result, keywords)
                    st.session_state.predefined_answer = explanation_result
                    st.session_state.predefined_question_type = "Why this sentiment?"
        
        with follow_col2:
            if st.button("What keywords influenced this?", use_container_width=True, key="keywords_influence"):
                with st.spinner("Analyzing keywords..."):
                    explanation_result = handle_followup_question("What keywords caused this result?", text_input, result, keywords)
                    st.session_state.predefined_answer = explanation_result
                    st.session_state.predefined_question_type = "What keywords influenced this?"
        
        with follow_col3:
            if st.button("How confident is this?", use_container_width=True, key="confidence_analysis"):
                with st.spinner("Analyzing confidence..."):
                    explanation_result = handle_followup_question("How confident is this result?", text_input, result, keywords)
                    st.session_state.predefined_answer = explanation_result
                    st.session_state.predefined_question_type = "How confident is this?"
        
        # Display predefined answer if available (but not if custom answer is shown)
        if st.session_state.predefined_answer and st.session_state.predefined_question_type and not (st.session_state.custom_answer and st.session_state.custom_question_asked):
            st.markdown(f"#### {st.session_state.predefined_question_type}")
            st.markdown(st.session_state.predefined_answer)
            
            # Add button to clear the predefined answer
            if st.button("Ask Another Question", key="clear_predefined_answer"):
                st.session_state.predefined_answer = None
                st.session_state.predefined_question_type = None
                st.rerun()

        # Custom question input
        st.markdown("**Or ask your own question:**")
        custom_question = st.text_input(
            "Ask anything about this analysis:",
            placeholder="e.g., Why is this negative? What made you choose this classification?",
            help="Ask questions about the sentiment, keywords, confidence, or anything else about the analysis",
            key="custom_followup_question"
        )
        
        # Initialize session state for custom answers if not exists
        if 'custom_answer' not in st.session_state:
            st.session_state.custom_answer = None
        if 'custom_question_asked' not in st.session_state:
            st.session_state.custom_question_asked = None
        
        if custom_question:
            if st.button("Get Answer", type="primary", key="get_custom_answer"):
                with st.spinner("Generating answer..."):
                    explanation_result = handle_followup_question(custom_question, text_input, result, keywords)
                    # Store the answer in session state
                    st.session_state.custom_answer = explanation_result
                    st.session_state.custom_question_asked = custom_question
        
        # Display stored custom answer if available
        if st.session_state.custom_answer and st.session_state.custom_question_asked:
            st.markdown("#### Answer to Your Question")
            st.markdown(f"**Q: {st.session_state.custom_question_asked}**")
            st.markdown(st.session_state.custom_answer)
            
            # Add button to clear the answer
            if st.button("Ask Another Question", key="clear_custom_answer"):
                st.session_state.custom_answer = None
                st.session_state.custom_question_asked = None
                st.rerun()

        # Quick help section
        with st.expander("Example Questions You Can Ask"):
            st.markdown("""
            **About Sentiment Classification:**
            - "Why is this labeled as negative?"
            - "What made you classify this as positive?"
            - "Why neutral instead of positive?"
            
            **About Keywords:**
            - "What keywords influenced this result?"
            - "Which words caused this classification?"
            - "How do keywords affect sentiment?"
            
            **About Confidence:**
            - "How confident are you in this result?"
            - "Why is the confidence score low?"
            - "Is this result reliable?"
            
            **About Improvement:**
            - "How can I improve the accuracy?"
            - "What would make this analysis better?"
            - "Tips for better sentiment analysis?"
            
            **About Definitions:**
            - "What does 'Very Positive' mean?"
            - "Explain the sentiment scale"
            - "What is sentiment analysis?"
            """)

with tab2:
    st.markdown("### Batch Analysis")
    st.markdown("Upload a CSV or TXT file containing multiple texts for batch sentiment analysis.")
    
    # File upload with enhanced error handling
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["csv", "txt"],
        help="Upload a CSV file with text in the first column, or a TXT file with one text per line"
    )
    
    if uploaded_file:
        # Use the comprehensive file validation function
        df, file_type = validate_and_process_file(uploaded_file)
        
        if df is not None:
            # Get the text column
            first_col = df.iloc[:, 0] if file_type == "csv" else df["text"]
            
            # Enhanced batch processing with better error handling
            try:
                # Validate texts before processing
                valid_texts = []
                invalid_count = 0
                
                with st.spinner("🔍 Validating texts..."):
                    for text in first_col.tolist():
                        if validate_text_input(str(text)) is None:
                            valid_texts.append(text)
                        else:
                            invalid_count += 1
                
                # Show validation results
                if invalid_count > 0:
                    display_warning_with_action(
                        f"Found {invalid_count} invalid entries that will be skipped.",
                        [
                            "Review your data for empty or invalid text entries",
                            "Ensure all text meets the minimum requirements (3-5000 characters)",
                            "Check for non-text data in your file"
                        ]
                    )
                
                if not valid_texts:
                    display_error_with_help("No valid texts found for analysis.", "validation")
                else:
                    # Process the valid texts with deployment-aware processing
                    with st.spinner(f"🚀 Analyzing {len(valid_texts)} texts..."):
                        try:
                            # Check if we're in a deployment environment
                            import os
                            import socket
                            
                            # Enhanced deployment detection
                            is_deployed = (
                                os.getenv('STREAMLIT_SHARING_MODE') == '1' or
                                'streamlit' in os.getenv('HOME', '').lower() or
                                os.getenv('DYNO') is not None or
                                os.getenv('RAILWAY_ENVIRONMENT') is not None or
                                'streamlit.app' in socket.getfqdn() or
                                'render.com' in socket.getfqdn() or
                                os.getenv('RENDER') is not None or
                                os.getenv('STREAMLIT_CLOUD') is not None or
                                'streamlit-cloud' in socket.getfqdn() or
                                # Force deployment mode for large batches to be safe
                                len(valid_texts) > 30
                            )
                            
                            if is_deployed:
                                st.info("🚀 Deployment environment detected - using optimized processing")
                                from deployment_fix import process_batch_deployment_safe
                                results_df = process_batch_deployment_safe(valid_texts)
                            else:
                                st.info("💻 Local environment detected - using full processing")
                                results_df = BatchProcessor.process_batch(valid_texts)
                            
                        except Exception as batch_error:
                            display_error_with_help(
                                f"Batch processing failed: {str(batch_error)}", 
                                "processing",
                                [
                                    "Try processing with a smaller dataset",
                                    "Check your internet connection", 
                                    "Refresh the page and try again",
                                    "For deployment issues, the app will use simplified processing"
                                ]
                            )
                            
                                                         # Fallback to deployment-safe processing
                             try:
                                 st.info("🔄 Trying deployment-safe processing as fallback...")
                                 from deployment_fix import process_batch_deployment_safe
                                 results_df = process_batch_deployment_safe(valid_texts)
                             except Exception as fallback_error:
                                 # Emergency fallback
                                 try:
                                     st.warning("🚨 Using emergency processing mode...")
                                     from emergency_deployment_fix import emergency_batch_processor
                                     results_df = emergency_batch_processor(valid_texts)
                                 except Exception as emergency_error:
                                     st.error(f"❌ All processing methods failed: {str(emergency_error)}")
                                     st.stop()
                    
                    # Optimize memory usage
                    try:
                        results_df = optimize_memory_usage(results_df)
                    except Exception as memory_error:
                        st.warning(f"⚠️ Memory optimization failed: {str(memory_error)}. Continuing with original data.")
                    
                    # Validate results
                    if results_df is None or results_df.empty:
                        display_error_with_help("No results were generated from the analysis.", "processing")
                        st.stop()
                    
                    # Check for analysis failures in results
                    if 'sentiment' in results_df.columns:
                        failed_analyses = results_df[results_df['sentiment'].isin(['Analysis Failed', 'Error', None])].shape[0]
                        successful_analyses = len(results_df) - failed_analyses
                        
                        if failed_analyses > 0:
                            display_warning_with_action(
                                f"{failed_analyses} analyses failed and may show as 'Error' in results.",
                                [
                                    "Review the failed entries for data quality issues",
                                    "Consider cleaning or reformatting problematic texts",
                                    "These entries will be marked as 'Analysis Failed' in results"
                                ]
                            )
                        
                        if successful_analyses == 0:
                            display_error_with_help("All analyses failed. Please check your data quality and try again.", "validation")
                            st.stop()
                        
                        # Success message with details
                        success_details = {
                            "Total Processed": len(results_df),
                            "Successful Analyses": successful_analyses,
                            "Failed Analyses": failed_analyses,
                            "Success Rate": f"{successful_analyses/len(results_df)*100:.1f}%",
                            "File Type": file_type.upper()
                        }
                        display_success_with_details("Batch analysis completed!", success_details)
                    else:
                        display_success_with_details(f"Processing completed for {len(results_df)} texts!")
                    
                    # Display results with tabs for different views
                    st.markdown("---")
                    results_tab1, results_tab2, results_tab3 = st.tabs(["📊 Results Table", "📈 Visualizations", "📤 Export Options"])
                    
                    with results_tab1:
                        st.subheader("Analysis Results")
                        # Add original text column for display
                        display_df = results_df.copy()
                        display_df['original_text'] = first_col.values[:len(results_df)]
                        
                        # Reorder columns for better display
                        if 'original_text' in display_df.columns:
                            cols = ['original_text', 'sentiment', 'confidence', 'use_case']
                            display_cols = [col for col in cols if col in display_df.columns]
                            display_df = display_df[display_cols]
                            
                            # Truncate long text for display
                            display_df['original_text'] = display_df['original_text'].astype(str).str[:100] + "..."
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        sentiment_counts = results_df['sentiment'].value_counts()
                        total_texts = len(results_df)
                        
                        with col1:
                            st.metric("Total Texts", total_texts)
                        with col2:
                            most_common = sentiment_counts.index[0] if len(sentiment_counts) > 0 else "N/A"
                            st.metric("Most Common", most_common)
                        with col3:
                            avg_confidence = results_df['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                        with col4:
                            positive_count = sentiment_counts.get('Positive', 0) + sentiment_counts.get('Very Positive', 0)
                            positive_pct = positive_count / total_texts * 100 if total_texts > 0 else 0
                            st.metric("Positive %", f"{positive_pct:.1f}%")
                    
                    with results_tab2:
                        st.subheader("Sentiment Distribution")
                        chart_type = st.selectbox(
                            "Select chart type",
                            ["bar", "pie", "donut", "line"],
                            key="batch_chart_type"
                        )
                        
                        # Create and display the selected chart
                        try:
                            fig = cached_visualization(results_df, "sentiment_distribution", plot_type=chart_type)
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Could not generate visualization. Please try a different chart type.")
                        except Exception as e:
                            st.error(f"Error creating visualization: {str(e)}")
                        
                        # Word Cloud
                        st.subheader("Word Cloud")
                        try:
                            # Generate word cloud from all texts
                            wordcloud_buf = cached_visualization(results_df, "wordcloud")
                            if wordcloud_buf is not None:
                                st.image(wordcloud_buf)
                            else:
                                st.warning("Could not generate word cloud. Please try again.")
                        except Exception as e:
                            st.error(f"Error generating word cloud: {str(e)}")
                    
                    with results_tab3:
                        # Enhanced export options with professional styling
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%); 
                                    padding: 2rem; border-radius: 20px; margin: 2rem 0; text-align: center;
                                    box-shadow: 0 10px 40px rgba(79, 70, 229, 0.3);">
                            <h2 style="color: white; margin: 0; font-size: 2rem; font-weight: 700;">📤 Export Analysis Results</h2>
                            <p style="color: rgba(255,255,255,0.9); margin: 1rem 0 0 0; font-size: 1.1rem;">
                                Download your comprehensive sentiment analysis in your preferred format
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Export options with enhanced styling
                        export_col1, export_col2, export_col3 = st.columns(3, gap="large")

                        with export_col1:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%); 
                                        border-radius: 16px; padding: 1.5rem; text-align: center; 
                                        border: 1px solid #E5E7EB; box-shadow: 0 4px 16px rgba(0,0,0,0.08);
                                        margin-bottom: 1rem;">
                                <h3 style="color: #059669; margin: 0; font-size: 2rem;">📊</h3>
                                <h4 style="color: #059669; margin: 0.5rem 0;">CSV Format</h4>
                                <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">
                                    Spreadsheet-ready data for Excel, Google Sheets, and analysis tools
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.download_button(
                                label="Download CSV",
                                data=results_df.to_csv(index=False),
                                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                help="Download results as CSV file for spreadsheet applications"
                            )

                        with export_col2:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%); 
                                        border-radius: 16px; padding: 1.5rem; text-align: center; 
                                        border: 1px solid #E5E7EB; box-shadow: 0 4px 16px rgba(0,0,0,0.08);
                                        margin-bottom: 1rem;">
                                <h3 style="color: #4F46E5; margin: 0; font-size: 2rem;">🔗</h3>
                                <h4 style="color: #4F46E5; margin: 0.5rem 0;">JSON Format</h4>
                                <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">
                                    API-ready structured data for applications and integrations
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.download_button(
                                label="Download JSON",
                                data=results_df.to_json(orient="records", indent=2),
                                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True,
                                help="Download results as JSON file for API integrations"
                            )

                        with export_col3:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%); 
                                        border-radius: 16px; padding: 1.5rem; text-align: center; 
                                        border: 1px solid #E5E7EB; box-shadow: 0 4px 16px rgba(0,0,0,0.08);
                                        margin-bottom: 1rem;">
                                <h3 style="color: #DC2626; margin: 0; font-size: 2rem;">📋</h3>
                                <h4 style="color: #DC2626; margin: 0.5rem 0;">PDF Report</h4>
                                <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">
                                    Professional report with charts, statistics, and insights
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Create visualizations dict for PDF
                            try:
                                # Get the current selected chart type from the UI
                                selected_chart_type = st.session_state.get("batch_chart_type", "bar")
                                
                                # Get the current visualization with user's selected chart type
                                fig = cached_visualization(results_df, "sentiment_distribution", plot_type=selected_chart_type)
                                wordcloud_buf = cached_visualization(results_df, "wordcloud")
                                
                                visualizations = {
                                    f"Sentiment Distribution ({selected_chart_type.title()} Chart)": fig,
                                    "Word Cloud": wordcloud_buf
                                }
                                
                                # Generate PDF
                                pdf_buffer = export_to_pdf(results_df, visualizations)
                                
                                # Enhanced download button with chart type info
                                chart_type_label = selected_chart_type.title()
                                st.download_button(
                                    label=f"Download PDF ({chart_type_label} Chart)",
                                    data=pdf_buffer.getvalue(),
                                    file_name=f"sentiment_report_{selected_chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                    help=f"Download comprehensive PDF report with {chart_type_label.lower()} chart visualization"
                                )
                            except Exception as e:
                                st.error(f"PDF generation temporarily unavailable: {str(e)}")
                                st.info("Try downloading CSV or JSON format instead")

                        # Additional export info
                        st.markdown("""
                        <div style="background: #F3F4F6; border-radius: 12px; padding: 1.5rem; margin-top: 2rem; text-align: center;">
                            <h4 style="color: #374151; margin: 0 0 1rem 0;">📈 Export Features</h4>
                            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
                                <div style="flex: 1; min-width: 200px;">
                                    <span style="color: #059669; font-weight: 600;">✅ Complete Data</span><br>
                                    <span style="color: #6B7280; font-size: 0.9rem;">All analysis results included</span>
                                </div>
                                <div style="flex: 1; min-width: 200px;">
                                    <span style="color: #4F46E5; font-weight: 600;">🎨 Formatted</span><br>
                                    <span style="color: #6B7280; font-size: 0.9rem;">Ready for presentation</span>
                                </div>
                                <div style="flex: 1; min-width: 200px;">
                                    <span style="color: #DC2626; font-weight: 600;">📊 With Charts</span><br>
                                    <span style="color: #6B7280; font-size: 0.9rem;">Visual insights included</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                display_error_with_help(f"Unexpected error during processing: {str(e)}", "general")
                st.stop()

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%); 
                padding: 2rem; border-radius: 20px; margin: 2rem 0; text-align: center;
                box-shadow: 0 10px 40px rgba(79, 70, 229, 0.3);">
        <h2 style="color: white; margin: 0; font-size: 2rem; font-weight: 700;">🔄 Advanced Comparative Analysis</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 1rem 0 0 0; font-size: 1.1rem;">
            Compare multiple texts with comprehensive insights, statistical analysis, and actionable recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced input section with sample data
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### 📝 Text Input")
    with col2:
        num_texts = st.number_input("Number of texts", min_value=2, max_value=5, value=2, help="Compare 2-5 texts simultaneously")
    with col3:
        st.markdown("### 🎯 Quick Start")
        
        # Sample data categories
        sample_categories = {
            "🛍️ Product Reviews": [
                {
                    "label": "5-Star Review",
                    "text": "This product exceeded all my expectations! The quality is outstanding, delivery was lightning fast, and customer service went above and beyond. I've recommended it to all my friends and family. Absolutely worth every penny and more!"
                },
                {
                    "label": "1-Star Review", 
                    "text": "Completely disappointed with this purchase. The product arrived damaged, took forever to ship, and customer service was unhelpful and rude. Would never buy from this company again. Save your money!"
                },
                {
                    "label": "3-Star Review",
                    "text": "The product is okay, nothing special. It does what it's supposed to do but the quality could be better for the price. Shipping was reasonable but packaging was minimal."
                }
            ],
            "📱 Social Media Posts": [
                {
                    "label": "Excited Post",
                    "text": "Just got the promotion I've been working towards for months! 🎉 So grateful for my amazing team and all the support. Time to celebrate! #blessed #career #success"
                },
                {
                    "label": "Complaint Post",
                    "text": "Stuck in traffic for 2 hours because of this construction project. Why can't they work on roads during off-peak hours? This is ridiculous and affecting everyone's commute."
                },
                {
                    "label": "Neutral Update",
                    "text": "Attended the quarterly meeting today. Discussed budget allocations and project timelines for Q4. Meeting notes will be shared by end of week."
                }
            ],
            "📞 Customer Feedback": [
                {
                    "label": "Positive Feedback",
                    "text": "I wanted to reach out and express my sincere appreciation for the exceptional service I received from your support team. They resolved my issue quickly and professionally. This level of care keeps me as a loyal customer."
                },
                {
                    "label": "Critical Feedback",
                    "text": "I'm extremely frustrated with the lack of response to my previous inquiries. It's been over a week and I still haven't received any updates on my order status. This poor communication is unacceptable."
                },
                {
                    "label": "Mixed Feedback",
                    "text": "The product itself is great and works as advertised, but the ordering process was confusing and the website was difficult to navigate. Good product, poor user experience."
                }
            ],
            "📰 News & Analysis": [
                {
                    "label": "Positive Economic News",
                    "text": "The latest economic indicators show strong growth in the technology sector, with unemployment rates reaching historic lows. Consumer confidence is up and businesses are reporting increased investment in innovation."
                },
                {
                    "label": "Concerning Report",
                    "text": "Environmental scientists warn that current pollution levels in major cities have reached critical thresholds. Immediate action is required to prevent long-term health consequences for millions of residents."
                },
                {
                    "label": "Factual Report",
                    "text": "The federal budget committee announced the allocation of $2.5 billion for infrastructure improvements across 15 states. The projects are scheduled to begin in Q1 of next year."
                }
            ],
            "🎬 Movie Reviews": [
                {
                    "label": "Glowing Review",
                    "text": "A masterpiece of cinematography! The director's vision comes alive with stunning visuals, compelling performances, and a soundtrack that perfectly complements every scene. This film will be remembered for decades."
                },
                {
                    "label": "Harsh Critique",
                    "text": "Two hours of my life I'll never get back. Poor plot development, wooden acting, and special effects that looked like they were made in someone's basement. How did this get greenlit?"
                },
                {
                    "label": "Balanced Review",
                    "text": "The film has some strong moments, particularly in the second act, but suffers from pacing issues and underdeveloped characters. Worth watching for the cinematography alone."
                }
            ],
            "💼 Business Communications": [
                {
                    "label": "Enthusiastic Email",
                    "text": "I'm thrilled to announce that our Q3 results have exceeded all projections! Thanks to everyone's hard work and dedication, we've achieved a 15% increase in revenue. Looking forward to building on this momentum."
                },
                {
                    "label": "Concern Memo",
                    "text": "Recent market analysis indicates potential challenges ahead. We need to reassess our current strategy and implement cost-saving measures to maintain competitiveness in this uncertain environment."
                },
                {
                    "label": "Status Update",
                    "text": "Project Alpha is currently 75% complete and on schedule for the December deadline. All team leads have confirmed resource availability for the final phase implementation."
                }
            ]
        }
        
        selected_category = st.selectbox(
            "Load Sample Pack",
            list(sample_categories.keys()),
            help="Choose a pre-built sample pack to test comparative analysis"
        )
        
        if st.button("Load Sample Pack", use_container_width=True):
            st.session_state['load_samples'] = True
            st.session_state['sample_data'] = sample_categories[selected_category]
            st.session_state['sample_num_texts'] = len(sample_categories[selected_category])
            st.success(f"Loaded {len(sample_categories[selected_category])} sample texts!")
            st.rerun()
    
    # Check if samples should be loaded
    if st.session_state.get('load_samples', False):
        num_texts = st.session_state.get('sample_num_texts', num_texts)
        st.session_state['load_samples'] = False  # Reset flag
    
    # Dynamic text input with better UX
    texts = []
    text_labels = []
    
    # Get sample data if loaded (for initial population)
    sample_data = st.session_state.get('sample_data', [])
    
    for i in range(num_texts):
        with st.container():
            st.markdown(f"**Text {i+1}:**")
            col_text, col_label = st.columns([4, 1])
            
            with col_text:
                # Use sample data if available for initial value only
                default_text = sample_data[i]['text'] if i < len(sample_data) else ""
                text = st.text_area(
                    f"Content for Text {i+1}", 
                    value=default_text,
                    height=120, 
                    key=f"comp_text_{i}",  # Unique key for comparative analysis
                    placeholder=f"Enter text {i+1} for comparison analysis..."
                )
            
            with col_label:
                # Use sample label if available for initial value only
                default_label = sample_data[i]['label'] if i < len(sample_data) else f"Text {i+1}"
                label = st.text_input(
                    f"Label {i+1}", 
                    value=default_label,
                    key=f"comp_label_{i}",  # Unique key for comparative analysis
                    help="Custom label for this text"
                )
            
            # Always read current text area value (this is the key fix)
            current_text = text if text else ""
            current_label = label if label else f"Text {i+1}"
            
            # Validate text input
            if current_text and current_text.strip():
                validation_error = validate_text_input(current_text)
                if validation_error:
                    st.error(f"{validation_error}")
                else:
                    texts.append(current_text)
                    text_labels.append(current_label)
                    st.success(f"Text {i+1} ready for analysis")
    
    # Clear sample data after first use to prevent re-population
    if st.session_state.get('sample_data') and not st.session_state.get('samples_cleared', False):
        st.session_state['samples_cleared'] = True
    
    # Analysis button with enhanced styling
    if len(texts) >= 2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run Comparative Analysis", type="primary", use_container_width=True, key="run_comparative_analysis"):
            try:
                with st.spinner("Performing comprehensive comparative analysis..."):
                    # Process all texts for comparison
                    comparison_results = []
                    detailed_results = []
                    
                    for i, text in enumerate(texts):
                        result = safe_sentiment_analysis(text)
                        if 'error' not in result:
                            keywords = safe_keyword_extraction(text)
                            explanation = explain_sentiment(text, result)
                            
                            comparison_results.append({
                                'Label': text_labels[i] if i < len(text_labels) else f"Text {i+1}",
                                'Content_Preview': text[:100] + "..." if len(text) > 100 else text,
                                'Sentiment': result['sentiment'],
                                'Confidence': result['confidence'],
                                'Use_Case': result.get('use_case', 'General'),
                                'Word_Count': len(text.split()),
                                'Character_Count': len(text),
                                'Key_Phrases': ", ".join(keywords) if keywords else "No keywords"
                            })
                            
                            detailed_results.append({
                                'index': i,
                                'label': text_labels[i] if i < len(text_labels) else f"Text {i+1}",
                                'text': text,
                                'result': result,
                                'keywords': keywords,
                                'explanation': explanation
                            })
                    
                    if comparison_results:
                        comparison_df = pd.DataFrame(comparison_results)
                        
                        # Create comprehensive results display
                        st.markdown("---")
                        st.markdown("## Comparative Analysis Results")
                        
                        # Summary metrics with enhanced styling
                        st.markdown("### Quick Insights")
                        
                        # Calculate key statistics
                        avg_confidence = comparison_df['Confidence'].mean()
                        sentiment_counts = comparison_df['Sentiment'].value_counts()
                        most_common_sentiment = sentiment_counts.index[0] if not sentiment_counts.empty else "Unknown"
                        confidence_range = comparison_df['Confidence'].max() - comparison_df['Confidence'].min()
                        
                        # Enhanced metrics display
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric(
                                "📈 Average Confidence", 
                                f"{avg_confidence:.1%}",
                                help="Average confidence across all analyzed texts"
                            )
                        
                        with metric_col2:
                            st.metric(
                                "🏆 Dominant Sentiment", 
                                most_common_sentiment,
                                help="Most frequently occurring sentiment"
                            )
                        
                        with metric_col3:
                            st.metric(
                                "📏 Confidence Range", 
                                f"{confidence_range:.1%}",
                                help="Difference between highest and lowest confidence scores"
                            )
                        
                        with metric_col4:
                            total_words = comparison_df['Word_Count'].sum()
                            st.metric(
                                "📝 Total Words", 
                                f"{total_words:,}",
                                help="Combined word count of all texts"
                            )
                        
                        # Enhanced tabbed results view
                        result_tab1, result_tab2, result_tab3, result_tab4, result_tab5 = st.tabs([
                            "📋 Summary Table", 
                            "📊 Visualizations", 
                            "🔍 Detailed Analysis", 
                            "📈 Statistical Insights",
                            "💼 Recommendations"
                        ])
                        
                        with result_tab1:
                            st.subheader("📋 Comparison Summary")
                            st.dataframe(
                                comparison_df.drop('Content_Preview', axis=1), 
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Content preview expandable section
                            st.subheader("📖 Content Preview")
                            for i, row in comparison_df.iterrows():
                                with st.expander(f"📄 {row['Label']} - {row['Sentiment']} ({row['Confidence']:.1%} confidence)"):
                                    st.write(row['Content_Preview'])
                        
                        with result_tab2:
                            st.subheader("📊 Comparative Visualizations")
                            
                            # Side-by-side confidence comparison
                            viz_col1, viz_col2 = st.columns(2)
                            
                            with viz_col1:
                                st.markdown("**Confidence Comparison**")
                                conf_fig = px.bar(
                                    comparison_df, 
                                    x='Label', 
                                    y='Confidence', 
                                    color='Sentiment',
                                    title="Confidence Scores by Text",
                                    color_discrete_map={
                                        'Very Positive': '#059669',
                                        'Positive': '#10B981',
                                        'Neutral': '#6B7280',
                                        'Negative': '#EF4444',
                                        'Very Negative': '#DC2626'
                                    }
                                )
                                conf_fig.update_layout(height=400, showlegend=True)
                                st.plotly_chart(conf_fig, use_container_width=True)
                            
                            with viz_col2:
                                st.markdown("**Sentiment Distribution**")
                                sent_fig = px.pie(
                                    values=sentiment_counts.values,
                                    names=sentiment_counts.index,
                                    title="Overall Sentiment Distribution",
                                    color_discrete_map={
                                        'Very Positive': '#059669',
                                        'Positive': '#10B981',
                                        'Neutral': '#6B7280',
                                        'Negative': '#EF4444',
                                        'Very Negative': '#DC2626'
                                    }
                                )
                                sent_fig.update_layout(height=400)
                                st.plotly_chart(sent_fig, use_container_width=True)
                            
                            # Word count vs confidence scatter plot
                            st.markdown("**Text Length vs Confidence Analysis**")
                            scatter_fig = px.scatter(
                                comparison_df,
                                x='Word_Count',
                                y='Confidence',
                                color='Sentiment',
                                size='Character_Count',
                                hover_data=['Label'],
                                title="Text Length vs Confidence Correlation",
                                color_discrete_map={
                                    'Very Positive': '#059669',
                                    'Positive': '#10B981',
                                    'Neutral': '#6B7280',
                                    'Negative': '#EF4444',
                                    'Very Negative': '#DC2626'
                                }
                            )
                            scatter_fig.update_layout(height=500)
                            st.plotly_chart(scatter_fig, use_container_width=True)
                        
                        with result_tab3:
                            st.subheader("🔍 Detailed Individual Analysis")
                            
                            for item in detailed_results:
                                with st.expander(f"📄 {item['label']} - Detailed Analysis", expanded=False):
                                    detail_col1, detail_col2 = st.columns([2, 1])
                                    
                                    with detail_col1:
                                        st.markdown("**Text Content:**")
                                        st.write(item['text'])
                                        
                                        st.markdown("**Key Phrases:**")
                                        if item['keywords']:
                                            st.write(", ".join(item['keywords']))
                                        else:
                                            st.write("No key phrases extracted")
                                    
                                    with detail_col2:
                                        st.markdown("**Analysis Results:**")
                                        st.metric("Sentiment", item['result']['sentiment'])
                                        st.metric("Confidence", f"{item['result']['confidence']:.1%}")
                                        st.metric("Use Case", item['result'].get('use_case', 'General'))
                                        
                                        # Reliability indicator
                                        reliability = item['explanation']['reliability']
                                        reliability_color = {
                                            'Very High': '🟢',
                                            'High': '🟢', 
                                            'Good': '🟡',
                                            'Moderate': '🟠',
                                            'Low': '🔴'
                                        }.get(reliability, '⚪')
                                        st.markdown(f"**Reliability:** {reliability_color} {reliability}")
                                        
                                        # Limitations
                                        if item['explanation']['limitations']:
                                            st.markdown("**Limitations:**")
                                            for limitation in item['explanation']['limitations']:
                                                st.warning(f"⚠️ {limitation}")
                        
                        with result_tab4:
                            st.subheader("📈 Statistical Insights")
                            
                            # Correlation analysis
                            stat_col1, stat_col2 = st.columns(2)
                            
                            with stat_col1:
                                st.markdown("**📊 Descriptive Statistics**")
                                stats_df = comparison_df[['Confidence', 'Word_Count', 'Character_Count']].describe()
                                st.dataframe(stats_df, use_container_width=True)
                                
                                # Confidence distribution
                                st.markdown("**📈 Confidence Distribution**")
                                conf_hist = px.histogram(
                                    comparison_df, 
                                    x='Confidence', 
                                    title="Confidence Score Distribution",
                                    nbins=min(10, len(comparison_df))
                                )
                                st.plotly_chart(conf_hist, use_container_width=True)
                            
                            with stat_col2:
                                st.markdown("**🔗 Correlation Analysis**")
                                
                                # Calculate correlations
                                numeric_df = comparison_df[['Confidence', 'Word_Count', 'Character_Count']]
                                correlation_matrix = numeric_df.corr()
                                
                                # Display correlation heatmap
                                corr_fig = px.imshow(
                                    correlation_matrix,
                                    title="Feature Correlation Matrix",
                                    color_continuous_scale="RdBu",
                                    aspect="auto"
                                )
                                st.plotly_chart(corr_fig, use_container_width=True)
                                
                                # Insights from correlations
                                st.markdown("**📝 Key Insights:**")
                                conf_word_corr = correlation_matrix.loc['Confidence', 'Word_Count']
                                
                                if abs(conf_word_corr) > 0.5:
                                    direction = "positive" if conf_word_corr > 0 else "negative"
                                    st.info(f"📍 Strong {direction} correlation between text length and confidence ({conf_word_corr:.2f})")
                                else:
                                    st.info(f"📍 Weak correlation between text length and confidence ({conf_word_corr:.2f})")
                        
                        with result_tab5:
                            st.subheader("💼 Actionable Recommendations")
                            
                            # Generate recommendations based on analysis
                            rec_col1, rec_col2 = st.columns(2)
                            
                            with rec_col1:
                                st.markdown("**🎯 Improvement Opportunities**")
                                
                                # Analyze lowest confidence texts
                                lowest_conf_idx = comparison_df['Confidence'].idxmin()
                                lowest_conf_text = comparison_df.iloc[lowest_conf_idx]
                                
                                st.warning(f"⚠️ **Lowest Confidence Text:** {lowest_conf_text['Label']} ({lowest_conf_text['Confidence']:.1%})")
                                st.markdown("**Suggested Actions:**")
                                st.markdown("• Add more context or detail")
                                st.markdown("• Review for mixed sentiment indicators")
                                st.markdown("• Consider manual review for accuracy")
                                
                                # Analyze text length recommendations
                                short_texts = comparison_df[comparison_df['Word_Count'] < 10]
                                if not short_texts.empty:
                                    st.info("📝 **Short Texts Detected**")
                                    st.markdown("Consider expanding these texts for better analysis accuracy:")
                                    for _, text in short_texts.iterrows():
                                        st.markdown(f"• {text['Label']} ({text['Word_Count']} words)")
                            
                            with rec_col2:
                                st.markdown("**✅ Best Practices Identified**")
                                
                                # Identify highest confidence text
                                highest_conf_idx = comparison_df['Confidence'].idxmax()
                                highest_conf_text = comparison_df.iloc[highest_conf_idx]
                                
                                st.success(f"🏆 **Highest Confidence Text:** {highest_conf_text['Label']} ({highest_conf_text['Confidence']:.1%})")
                                st.markdown("**Success Factors:**")
                                st.markdown(f"• Word count: {highest_conf_text['Word_Count']} words")
                                st.markdown(f"• Clear sentiment: {highest_conf_text['Sentiment']}")
                                st.markdown("• Well-structured content")
                                
                                # Overall recommendations
                                st.markdown("**📊 Overall Strategy:**")
                                if avg_confidence > 0.7:
                                    st.success("🎉 High overall confidence - analysis is reliable")
                                elif avg_confidence > 0.5:
                                    st.warning("⚡ Moderate confidence - consider additional context")
                                else:
                                    st.error("🔄 Low confidence - recommend manual review")
                        
                        # Enhanced export section
                        st.markdown("---")
                        st.markdown("### 📤 Export Comparative Analysis")
                        
                        export_col1, export_col2, export_col3 = st.columns(3)
                        
                        with export_col1:
                            st.download_button(
                                label="📊 Download Comparison CSV",
                                data=comparison_df.to_csv(index=False),
                                file_name=f"comparative_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                help="Download detailed comparison results"
                            )
                        
                        with export_col2:
                            comparison_json = {
                                'analysis_timestamp': datetime.now().isoformat(),
                                'summary_metrics': {
                                    'average_confidence': float(avg_confidence),
                                    'dominant_sentiment': most_common_sentiment,
                                    'confidence_range': float(confidence_range),
                                    'total_texts': len(comparison_results)
                                },
                                'detailed_results': comparison_results
                            }
                            
                            st.download_button(
                                label="🔗 Download JSON Report",
                                data=json.dumps(comparison_json, indent=2),
                                file_name=f"comparative_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True,
                                help="Download structured analysis report"
                            )
                        
                        with export_col3:
                            try:
                                # Create visualizations for PDF
                                pdf_visualizations = {
                                    "Confidence Comparison": conf_fig,
                                    "Sentiment Distribution": sent_fig,
                                    "Length vs Confidence": scatter_fig
                                }
                                
                                pdf_buffer = export_to_pdf(comparison_df, pdf_visualizations)
                                st.download_button(
                                    label="📋 Download PDF Report",
                                    data=pdf_buffer.getvalue(),
                                    file_name=f"comparative_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                    help="Download comprehensive PDF report with visualizations"
                                )
                            except Exception as pdf_error:
                                st.error(f"⚠️ PDF generation temporarily unavailable: {str(pdf_error)}")
                                st.info("💡 Try downloading CSV or JSON format instead")
                    
                    else:
                        display_error_with_help("No texts could be analyzed for comparison.", "processing")
            
            except Exception as e:
                display_error_with_help(f"Comparison analysis failed: {str(e)}", "general")
    
    else:
        # Help section when not enough texts
        st.markdown("### 💡 Getting Started")
        st.info("📝 Enter at least 2 texts above to begin comparative analysis")
        
        # Feature preview
        st.markdown("### 🚀 What You'll Get:")
        preview_col1, preview_col2, preview_col3 = st.columns(3)
        
        with preview_col1:
            st.markdown("""
            **📊 Visual Comparisons**
            - Side-by-side sentiment analysis
            - Confidence score comparisons  
            - Interactive charts and graphs
            """)
        
        with preview_col2:
            st.markdown("""
            **📈 Statistical Insights**
            - Correlation analysis
            - Distribution patterns
            - Performance metrics
            """)
        
        with preview_col3:
            st.markdown("""
            **💼 Actionable Recommendations**
            - Improvement suggestions
            - Best practice identification
            - Quality optimization tips
            """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Tech Titanians</p>
</div>
""", unsafe_allow_html=True)
