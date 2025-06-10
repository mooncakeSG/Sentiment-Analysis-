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
    plot_sentiment_distribution,
    generate_wordcloud
)

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

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    h1 {
        color: #1f77b4;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        text-align: center;
    }
    
    /* Subtitle styling */
    h2 {
        color: #2c3e50;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Make dataframes use full width */
    .st-emotion-cache-1v0mbdj {
        width: 100%;
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Enhance text areas */
    .stTextArea textarea {
        font-size: 16px !important;
        line-height: 1.5;
        padding: 12px;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #1f77b4;
        box-shadow: 0 0 0 1px #1f77b4;
    }
    
    /* Style file uploader */
    .stUploadedFile {
        border-radius: 8px;
        padding: 16px;
        background-color: white;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Enhance metrics */
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
    }
    
    /* Style download buttons */
    .stDownloadButton {
        width: 100%;
        border-radius: 8px;
        margin: 4px 0;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    
    .stDownloadButton:hover {
        background-color: #155a8f;
    }
    
    /* Enhance tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 8px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 8px;
        gap: 8px;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
        border-bottom: none;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2c3e50;
        color: white;
    }
    
    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Error message styling */
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Warning message styling */
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6c757d;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    .export-section {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem 1rem 1.5rem 1rem;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .export-btn {
        width: 100%;
        height: 48px;
        font-size: 1.1rem;
        font-weight: 500;
        border-radius: 8px;
        border: none;
        background: #1f77b4;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        transition: background 0.2s;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .export-btn:hover {
        background-color: #155a8f;
    }
    .export-label {
        font-size: 1.05rem;
        color: #333;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# Add professional styling and layout improvements
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f9fafb;
            color: #333;
        }
        /* Card style for DataFrame and export container */
        .dataframe-container, .export-card {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }
        /* Table header */
        thead tr th {
            background-color: #f0f0f0 !important;
            color: #222 !important;
            font-weight: bold !important;
        }
        /* Export buttons */
        button[data-testid="baseButton-secondary"] {
            border-radius: 8px !important;
            background-color: #6366f1 !important;
            color: white !important;
            border: none !important;
            font-weight: 500;
            font-size: 1.05rem;
        }
        button[data-testid="baseButton-secondary"]:hover {
            background-color: #4338ca !important;
            color: #f3f4f6 !important;
        }
        /* Footer style */
        footer { visibility: hidden; }
        .css-18ni7ap { padding-top: 0rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- Inject improved CSS for all UI areas ---
st.markdown("""
<style>
/* Hero Banner */
.hero-banner {
    background: linear-gradient(to right, #6a11cb, #2575fc);
    box-shadow: 0 4px 24px rgba(106,17,203,0.08);
    border-radius: 18px;
    padding: 1.5rem 2rem 1.2rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero-banner h1 {
    color: #fff;
    font-size: 2.5em;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.5px;
}
@media (max-width: 600px) {
    .hero-banner h1 { font-size: 1.5em; }
}

/* Tabbed Nav Buttons */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: #f3f4f6;
    border-radius: 12px;
    padding: 6px 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    background-color: #f0f4f8;
    border: 1px solid #ccc;
    margin-right: 0px;
    transition: 0.3s;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #e0e7ff;
    border-color: #6366f1;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(to right, #6366f1, #818cf8);
    color: #fff;
    border: 1.5px solid #6366f1;
    box-shadow: 0 2px 8px rgba(99,102,241,0.08);
}

/* Upload CSV Box & File Info */
.stUploadedFile {
    background-color: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    padding: 12px 18px !important;
    border-radius: 10px !important;
    margin-top: 8px !important;
    font-size: 0.98em !important;
    color: #374151 !important;
}
.file-info-card {
    background: #f3f4f6;
    border-radius: 10px;
    padding: 10px 18px;
    margin: 10px 0 18px 0;
    font-size: 0.97em;
    color: #374151;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* Export Section */
.export-container {
    background-color: #f3f4f6;
    padding: 16px 10px 10px 10px;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-top: 24px;
    text-align: center;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}
.export-container button {
    margin: 0 10px;
    background-color: #6366f1 !important;
    color: white !important;
    border-radius: 8px !important;
    font-size: 1.05rem;
    font-weight: 500;
    min-width: 140px;
    padding: 0.5rem 0.2rem;
}
.export-container button:hover {
    background-color: #4338ca !important;
    color: #f3f4f6 !important;
}
.export-caption {
    text-align: center;
    color: #888;
    font-size: 0.97rem;
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# Global error handler wrapper - wrap entire main application
try:
    # --- Hero Banner ---
    st.markdown("""
    <div class="hero-banner">
        <h1>Sentiment Analysis Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("""
<div style='text-align: center; color: #6c757d; margin-bottom: 2rem;'>
    Analyze sentiment in text data using state-of-the-art natural language processing.
    Get insights from customer reviews, social media posts, or any text content.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This dashboard uses:
    - 🤖 Hugging Face Transformers for sentiment analysis
    - 🔑 KeyBERT for keyword extraction
    - 🎨 Streamlit for the user interface
    """)
    
    # Model Limitations
    with st.expander("Model Limitations", expanded=False):
        st.markdown("### Model Specifications")
        st.write(f"- Maximum text length: {MODEL_LIMITATIONS['max_text_length']} tokens")
        st.write(f"- Confidence threshold: {MODEL_LIMITATIONS['confidence_threshold']}")
        st.write(f"- Supported languages: {', '.join(MODEL_LIMITATIONS['supported_languages'])}")
        
        st.markdown("### Known Limitations")
        for limitation in MODEL_LIMITATIONS['known_limitations']:
            st.write(f"- {limitation}")

# Main content
tab1, tab2, tab3 = st.tabs(["📝 Single Text Analysis", "📚 Batch Analysis", "🔄 Comparative Analysis"])

with tab1:
    # Single text input
    text_input = st.text_area("Enter text to analyze:", height=150, 
                             help="Enter any text for sentiment analysis. Minimum 3 characters, maximum 5000 characters.")
    
    if text_input:
        # Validate input first
        validation_error = validate_text_input(text_input)
        if validation_error:
            st.error(validation_error)
            st.info("💡 **Tips for better analysis:**\n- Use natural language text\n- Ensure text is between 3-5000 characters\n- Avoid excessive special characters or formatting")
        else:
            try:
                with st.spinner("🔍 Analyzing text..."):
                    # Use safe analysis functions
                    result = safe_sentiment_analysis(text_input)
                    
                    # Check for analysis errors
                    if 'error' in result:
                        st.error(result['error'])
                        st.info("💡 **Troubleshooting:**\n- Try rephrasing your text\n- Check for unusual characters\n- Ensure text is in a supported language")
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
            except Exception as e:
                st.error(f"❌ Unexpected error occurred: {str(e)}")
                st.info("💡 Please refresh the page and try again. If the problem persists, contact support.")
                st.stop()
            
            # Create columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Analysis")
                sentiment_color = {
                    "Very Positive": "green",
                    "Positive": "green", 
                    "Neutral": "gray",
                    "Negative": "red",
                    "Very Negative": "red"
                }[result['sentiment']]
                st.markdown(f"**Sentiment:** :{sentiment_color}[{result['sentiment']}]")
                st.markdown(f"**Confidence:** {result['confidence']:.2%}")
                
                # Display explanation
                st.subheader("Analysis Explanation")
                st.write(f"**Reliability:** {explanation['reliability']}")
                if explanation['limitations']:
                    st.warning("**Limitations:**")
                    for limitation in explanation['limitations']:
                        st.write(f"- {limitation}")
            
            with col2:
                st.subheader("Keywords")
                st.write(", ".join(keywords))
                
                # Display use case information
                st.subheader("Suggested Use Case")
                st.write(f"📊 {result.get('use_case', 'General Analysis')}")
            
            # Interactive Q&A Section
            st.markdown("---")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 15px; border-radius: 12px; margin: 15px 0; text-align: center;">
                <h4 style="color: white; margin: 0; font-size: 1.2rem;">🤔 Ask Questions About This Analysis</h4>
                <p style="color: #f0f0f0; margin: 5px 0 0 0; font-size: 0.85rem;">Get detailed explanations about the sentiment classification</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Predefined question buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Why this sentiment?", use_container_width=True):
                    explanation = handle_followup_question("Why was this labeled this way?", text_input, result, keywords)
                    st.markdown("### 💡 Explanation")
                    st.markdown(explanation)
            
            with col2:
                if st.button("What keywords influenced this?", use_container_width=True):
                    explanation = handle_followup_question("What keywords caused this result?", text_input, result, keywords)
                    st.markdown("### 🔍 Keyword Analysis")
                    st.markdown(explanation)
            
            with col3:
                if st.button("How confident is this?", use_container_width=True):
                    explanation = handle_followup_question("How confident is this result?", text_input, result, keywords)
                    st.markdown("### 📊 Confidence Analysis")
                    st.markdown(explanation)
            
            # Custom question input
            st.markdown("**Or ask your own question:**")
            custom_question = st.text_input(
                "Ask anything about this analysis:",
                placeholder="e.g., Why is this negative? What made you choose this classification?",
                help="Ask questions about the sentiment, keywords, confidence, or anything else about the analysis"
            )
            
            if custom_question:
                if st.button("Get Answer", type="primary"):
                    explanation = handle_followup_question(custom_question, text_input, result, keywords)
                    st.markdown("### 🎯 Answer to Your Question")
                    st.markdown(f"**Q: {custom_question}**")
                    st.markdown(explanation)
            
            # Quick help section
            with st.expander("💡 Example Questions You Can Ask"):
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
            
            # Create DataFrame for visualizations
            df = pd.DataFrame([{
                'text': text_input,
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'keywords': ', '.join(keywords),
                'use_case': result.get('use_case', 'General Analysis')
            }])
            
            # Optimize memory usage
            df = optimize_memory_usage(df)
            
            # Visualization tabs
            viz_tab1, viz_tab2 = st.tabs(["Charts", "Word Cloud"])
            
            with viz_tab1:
                # Display sentiment distribution visualization
                st.subheader("Sentiment Distribution")
                chart_type = st.selectbox(
                    "Select chart type",
                    ["bar", "pie", "donut", "line"],
                    key="chart_type"
                )
                
                # Create and display the selected chart
                fig = plot_sentiment_distribution(df, plot_type=chart_type)
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab2:
                st.subheader("Word Cloud")
                wordcloud_buf = generate_wordcloud(text_input, [result['sentiment']])
                st.image(wordcloud_buf)
            
            # Export options with improved styling
            st.markdown("---")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
                <h3 style="color: white; margin: 0; font-size: 1.5rem;">📤 Export Results</h3>
                <p style="color: #f0f0f0; margin: 5px 0 0 0; font-size: 0.9rem;">Download your analysis in multiple formats</p>
            </div>
            """, unsafe_allow_html=True)
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="CSV Format",
                    data=csv,
                    file_name="sentiment_analysis.csv",
                    mime="text/csv",
                    help="Download results as CSV file",
                    use_container_width=True
                )
                st.markdown("<p style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 5px;'>📊 Spreadsheet Data</p>", unsafe_allow_html=True)
            
            with export_col2:
                json_str = df.to_json(orient="records")
                st.download_button(
                    label="JSON Format",
                    data=json_str,
                    file_name="sentiment_analysis.json",
                    mime="application/json",
                    help="Download results as JSON file",
                    use_container_width=True
                )
                st.markdown("<p style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 5px;'>🔗 API Ready Data</p>", unsafe_allow_html=True)
            
            with export_col3:
                # Create visualizations dict for PDF
                visualizations = {
                    "Sentiment Distribution": fig,
                    "Word Cloud": wordcloud_buf
                }
                
                # Generate PDF
                try:
                    pdf_buffer = export_to_pdf(df, visualizations)
                    st.download_button(
                        label="PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name="sentiment_analysis_report.pdf",
                        mime="application/pdf",
                        help="Download comprehensive PDF report",
                        use_container_width=True
                    )
                    st.markdown("<p style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 5px;'>📋 Professional Report</p>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")

with tab2:
    st.markdown("""
    ### Batch Analysis
    Upload a file containing texts to analyze. Supported formats:
    - **CSV files**: First column should contain text data
    - **TXT files**: One sentence/text per line
    """)
    
    uploaded_file = st.file_uploader(
        "Upload CSV or TXT file", 
        type=['csv', 'txt'],
        help="CSV: first column is text. TXT: one sentence per line."
    )
    
    if uploaded_file:
        try:
            # Show file details
            file_details = {
                "📁 Filename": uploaded_file.name,
                "📏 File size": f"{uploaded_file.size / 1024:.2f} KB",
                "📄 File type": uploaded_file.type
            }
            st.markdown("**📊 File Details:**")
            for k, v in file_details.items():
                st.write(f"- {k}: {v}")
            
            # File size validation
            max_size_mb = 10
            if uploaded_file.size > max_size_mb * 1024 * 1024:
                st.error(f"❌ File is too large ({uploaded_file.size / 1024 / 1024:.1f} MB). Maximum allowed size is {max_size_mb} MB.")
                st.stop()
            
            # Detect file type and process accordingly
            filename = uploaded_file.name.lower()
            df = None
            
            if filename.endswith(".csv"):
                try:
                    # Read and process CSV with enhanced error handling
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                    file_type = "csv"
                except UnicodeDecodeError:
                    try:
                        # Try different encoding
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                        file_type = "csv"
                        st.info("ℹ️ File encoding detected as Latin-1")
                    except Exception as e:
                        st.error(f"❌ Could not read CSV file. Error: {str(e)}")
                        st.info("💡 **Tips:** Ensure your CSV file is properly formatted and saved with UTF-8 encoding.")
                        st.stop()
                except pd.errors.EmptyDataError:
                    st.error("❌ CSV file appears to be empty or corrupted.")
                    st.stop()
                except pd.errors.ParserError as e:
                    st.error(f"❌ CSV parsing error: {str(e)}")
                    st.info("💡 **Tips:** Check that your CSV file has proper formatting (commas, quotes, etc.)")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Unexpected error reading CSV: {str(e)}")
                    st.stop()
                    
            elif filename.endswith(".txt"):
                try:
                    # Read and process TXT file with enhanced error handling
                    uploaded_file.seek(0)
                    raw_text = uploaded_file.read().decode("utf-8")
                    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                    
                    if not lines:
                        st.error("❌ TXT file is empty or contains no valid text lines.")
                        st.info("💡 **Tips:** Ensure each line contains text and the file is not empty.")
                        st.stop()
                    
                    # Create DataFrame from text lines
                    df = pd.DataFrame(lines, columns=["text"])
                    file_type = "txt"
                    
                except UnicodeDecodeError:
                    try:
                        # Try different encoding
                        uploaded_file.seek(0)
                        raw_text = uploaded_file.read().decode("latin-1")
                        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                        df = pd.DataFrame(lines, columns=["text"])
                        file_type = "txt"
                        st.info("ℹ️ File encoding detected as Latin-1")
                    except Exception as e:
                        st.error(f"❌ Could not read TXT file. Error: {str(e)}")
                        st.info("💡 **Tips:** Ensure your TXT file is saved with UTF-8 encoding.")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Error processing TXT file: {str(e)}")
                    st.stop()
                
            else:
                st.error("❌ Unsupported file type! Please upload a CSV or TXT file.")
                st.info("💡 **Supported formats:** .csv, .txt")
                st.stop()
            
            # Validate file content
            is_valid, error_msg, warning_msg = validate_file_content(df, file_type)
            
            if not is_valid:
                st.error(error_msg)
                st.info("💡 **Next steps:** Please fix the issues in your file and try uploading again.")
                st.stop()
            
            if warning_msg:
                st.warning(warning_msg)
            
            # Success message with details
            st.success(f"✅ Successfully loaded {file_type.upper()} file with {len(df)} valid entries!")
            
            # Show preview of the data
            st.markdown("**📋 Preview of uploaded data:**")
            try:
                preview_df = df.head(10).copy()
                # Truncate long text for preview
                if file_type == "csv":
                    preview_df.iloc[:, 0] = preview_df.iloc[:, 0].astype(str).str[:100] + "..."
                else:
                    preview_df["text"] = preview_df["text"].str[:100] + "..."
                st.dataframe(preview_df, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not display preview: {str(e)}")
            
            # Get the text column
            first_col = df.iloc[:, 0] if file_type == "csv" else df["text"]
            
            # Enhanced batch processing with error handling
            try:
                # Validate each text before processing
                valid_texts = []
                invalid_indices = []
                
                with st.spinner("🔍 Validating texts..."):
                    for idx, text in enumerate(first_col.tolist()):
                        validation_error = validate_text_input(str(text))
                        if validation_error:
                            invalid_indices.append(idx + 1)  # 1-based for user display
                        else:
                            valid_texts.append(text)
                
                # Show validation results
                if invalid_indices:
                    if len(invalid_indices) <= 10:
                        st.warning(f"⚠️ Skipping {len(invalid_indices)} invalid entries at rows: {', '.join(map(str, invalid_indices))}")
                    else:
                        st.warning(f"⚠️ Skipping {len(invalid_indices)} invalid entries (first 10 rows: {', '.join(map(str, invalid_indices[:10]))}, ...)")
                
                if not valid_texts:
                    st.error("❌ No valid texts found for analysis. Please check your data and try again.")
                    st.stop()
                
                st.info(f"ℹ️ Processing {len(valid_texts)} valid texts out of {len(first_col)} total entries.")
                
                # Process the valid texts with progress tracking
                with st.spinner(f"🚀 Analyzing {len(valid_texts)} texts..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        results_df = BatchProcessor.process_batch(valid_texts)
                        progress_bar.progress(1.0)
                        status_text.text("Analysis complete!")
                        
                        # Clean up progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                    except Exception as batch_error:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Batch processing failed: {str(batch_error)}")
                        st.info("💡 Try processing with a smaller file or contact support.")
                        st.stop()
                
                # Optimize memory usage
                try:
                    results_df = optimize_memory_usage(results_df)
                except Exception as memory_error:
                    st.warning(f"⚠️ Memory optimization failed: {str(memory_error)}. Continuing with original data.")
                
                # Validate results
                if results_df is None or results_df.empty:
                    st.error("❌ No results were generated from the analysis.")
                    st.stop()
                
                # Check for analysis failures in results
                if 'sentiment' in results_df.columns:
                    failed_analyses = results_df[results_df['sentiment'].isin(['Analysis Failed', 'Error', None])].shape[0]
                    successful_analyses = len(results_df) - failed_analyses
                    
                    if failed_analyses > 0:
                        st.warning(f"⚠️ {failed_analyses} analyses failed and may show as 'Error' in results.")
                    
                    if successful_analyses == 0:
                        st.error("❌ All analyses failed. Please check your data quality and try again.")
                        st.stop()
                    
                    # Success message with details
                    st.success(f"✅ Successfully analyzed {successful_analyses} out of {len(results_df)} texts!")
                else:
                    st.success(f"✅ Processing completed for {len(results_df)} texts!")
                
            except Exception as e:
                st.error(f"❌ Unexpected error during processing: {str(e)}")
                st.info("💡 Please try with a smaller file or contact support if the issue persists.")
                st.stop()
            
            # Create tabs for different views
            results_tab1, results_tab2, results_tab3 = st.tabs(["Results Table", "Visualizations", "Word Cloud"])
            
            with results_tab1:
                st.subheader("Analysis Results")
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            with results_tab2:
                # Display sentiment distribution visualization
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
            
            with results_tab3:
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
            
            # Export options with improved styling
            st.markdown("---")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
                <h3 style="color: white; margin: 0; font-size: 1.5rem;">📤 Export Analysis Results</h3>
                <p style="color: #f0f0f0; margin: 5px 0 0 0; font-size: 0.9rem;">Download your batch analysis in multiple formats</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    label="CSV Format",
                    data=results_df.to_csv(index=False),
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download batch results as CSV file"
                )
                st.markdown("<p style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 5px;'>📊 Spreadsheet Data</p>", unsafe_allow_html=True)

            with col2:
                st.download_button(
                    label="JSON Format",
                    data=results_df.to_json(orient="records"),
                    file_name="sentiment_results.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Download batch results as JSON file"
                )
                st.markdown("<p style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 5px;'>🔗 API Ready Data</p>", unsafe_allow_html=True)

            with col3:
                # Create visualizations dict for PDF
                visualizations = {
                    "Sentiment Distribution": fig,
                    "Word Cloud": wordcloud_buf
                }
                
                # Generate PDF
                try:
                    pdf_buffer = export_to_pdf(results_df, visualizations)
                    st.download_button(
                        label="PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name="sentiment_results.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        help="Download comprehensive PDF report"
                    )
                    st.markdown("<p style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 5px;'>📋 Professional Report</p>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")

            st.markdown("<p style='text-align: center; color: #888; font-size: 0.85rem; margin-top: 15px;'>Choose your preferred format to download the analysis results</p>", unsafe_allow_html=True)
        
        except ValueError as ve:
            st.error(f"❌ Validation Error: {str(ve)}")
            st.info("Please check your CSV file format and try again.")
        
        except Exception as e:
            st.error(f"❌ Processing Error: {str(e)}")
            st.info("If this error persists, please contact support.")

with tab3:
    st.markdown("""
    ### Comparative Analysis
    Enter multiple texts to compare their sentiment analysis results.
    """)
    
    num_texts = st.number_input("Number of texts to compare", min_value=2, max_value=5, value=2)
    
    texts = []
    for i in range(num_texts):
        text = st.text_area(f"Text {i+1}", height=100, key=f"text_{i}")
        if text:
            texts.append(text)
    
    if len(texts) >= 2:
        if st.button("Compare Texts"):
            with st.spinner("Analyzing texts..."):
                # Use optimized batch processor for comparison
                comparison_df = BatchProcessor.process_batch(texts)
                
                # Optimize memory usage
                comparison_df = optimize_memory_usage(comparison_df)
                
                # Display results
                st.subheader("Comparison Results")
                st.dataframe(comparison_df, use_container_width=True)
                
                # Display explanations
                st.subheader("Detailed Explanations")
                for idx, row in comparison_df.iterrows():
                    with st.expander(f"Text {idx+1} Analysis"):
                        st.write(f"**Sentiment:** {row['sentiment']}")
                        st.write(f"**Confidence:** {row['confidence']:.2%}")
                        st.write("**Key Phrases:**")
                        st.write(row['keywords'])
                        
                        # Add visualization for each text
                        text_df = pd.DataFrame([{
                            'text': row['text'],
                            'sentiment': row['sentiment'],
                            'confidence': row['confidence']
                        }])
                        
                        # Create sentiment distribution for single text
                        try:
                            fig = cached_visualization(text_df, "sentiment_distribution", plot_type="bar")
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Could not generate visualization for this text.")
                        except Exception as e:
                            st.error(f"Error creating visualization: {str(e)}")
                        
                        # Create word cloud for single text
                        try:
                            wordcloud_buf = cached_visualization(text_df, "wordcloud")
                            if wordcloud_buf is not None:
                                st.image(wordcloud_buf)
                            else:
                                st.warning("Could not generate word cloud for this text.")
                        except Exception as e:
                            st.error(f"Error generating word cloud: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Tech Titanians</p>
</div>
""", unsafe_allow_html=True)

except Exception as global_error:
    st.error("❌ An unexpected error occurred in the application")
    st.markdown("""
    ### 🔧 Troubleshooting Steps:
    1. **Refresh the page** and try again
    2. **Clear browser cache** if issues persist
    3. **Check your input data** for unusual characters or formatting
    4. **Try with smaller datasets** if processing large files
    5. **Contact support** if the problem continues
    
    ### 📧 Error Details (for support):
    """)
    with st.expander("Technical Error Information"):
        st.code(f"Error Type: {type(global_error).__name__}")
        st.code(f"Error Message: {str(global_error)}")
    
    st.info("💡 **Quick Fix:** Most issues are resolved by refreshing the page (Ctrl+F5 or Cmd+R)")
    
    # Option to restart
    if st.button("🔄 Restart Application", type="primary"):
        st.rerun()

