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
    text_input = st.text_area("Enter text to analyze:", height=150)
    
    if text_input:
        with st.spinner("Analyzing text..."):
            # Use cached functions
            result = cached_sentiment_analysis(text_input)
            explanation = explain_sentiment(text_input, result)
            keywords = cached_keyword_extraction(text_input)
            
            # Create columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Analysis")
                sentiment_color = {
                    "Positive": "green",
                    "Neutral": "gray",
                    "Negative": "red"
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
            
            # Create DataFrame for visualizations
            df = pd.DataFrame([{
                'text': text_input,
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'keywords': ', '.join(keywords)
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
            
            # Export options
            st.subheader("Export Results")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name="sentiment_analysis.csv",
                    mime="text/csv",
                    help="Download results as CSV file"
                )
            
            with export_col2:
                json_str = df.to_json(orient="records")
                st.download_button(
                    label="📋 Download JSON",
                    data=json_str,
                    file_name="sentiment_analysis.json",
                    mime="application/json",
                    help="Download results as JSON file"
                )
            
            with export_col3:
                # Export to PDF button
                if st.button("Export to PDF"):
                    try:
                        # Create visualizations dict for PDF
                        visualizations = {
                            "Sentiment Distribution": fig,
                            "Word Cloud": wordcloud_buf
                        }
                        
                        # Generate PDF
                        pdf_buffer = export_to_pdf(df, visualizations)
                        
                        # Create download button
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_buffer.getvalue(),
                            file_name="sentiment_analysis_report.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")

with tab2:
    st.markdown("""
    ### Batch Analysis
    Upload a CSV file containing texts to analyze. The file should have:
    - At least one column
    - Text data in the first column
    - No empty cells
    """)
    
    uploaded_file = st.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        help="Your CSV file should have a column of texts to analyze. The first column will be used."
    )
    
    if uploaded_file:
        try:
            # Show file details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            st.write("**File Details:**")
            for k, v in file_details.items():
                st.write(f"- {k}: {v}")
            
            # Read and process CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate input
            if df.empty:
                raise ValueError("The uploaded CSV file is empty.")
            
            if len(df.columns) < 1:
                raise ValueError("The CSV file must contain at least one column.")
            
            # Check if first column contains text data
            first_col = df.iloc[:, 0]
            if not first_col.dtype == object:
                raise ValueError("The first column must contain text data.")
            
            if first_col.isnull().any():
                raise ValueError("The text column contains empty cells. Please remove or fill them.")
            
            # Use optimized batch processor
            with st.spinner("Processing texts..."):
                results_df = BatchProcessor.process_batch(first_col.tolist())
            
            # Optimize memory usage
            results_df = optimize_memory_usage(results_df)
            
            # Success message
            st.success(f"Successfully analyzed {len(results_df)} texts!")
            
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
            
            # Export options in an expander
            csv = results_df.to_csv(index=False)
            json_str = results_df.to_json(orient="records")

            st.markdown("""
            <div style="background: #f3f4f6; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); padding: 1.5rem 1rem; max-width: 350px; margin: 0 auto 2rem auto;">
                <h4 style="text-align:center; margin-bottom: 1.2rem;">Export Analysis Results</h4>
            </div>
            """, unsafe_allow_html=True)

            st.download_button(
                label="Export as CSV 📄",
                data=csv,
                file_name="batch_sentiment_analysis.csv",
                mime="text/csv",
                key="export_csv",
                use_container_width=True
            )
            st.download_button(
                label="Export as JSON 📋",
                data=json_str,
                file_name="batch_sentiment_analysis.json",
                mime="application/json",
                key="export_json",
                use_container_width=True
            )

            # PDF: generate and download in one click, no session state
            if st.button("Generate PDF 🧾", key="export_pdf_btn", use_container_width=True):
                try:
                    visualizations = {
                        "Sentiment Distribution": fig,
                        "Word Cloud": wordcloud_buf
                    }
                    pdf_buffer = export_to_pdf(results_df, visualizations)
                    st.download_button(
                        label="Download PDF Report 🧾",
                        data=pdf_buffer.getvalue(),
                        file_name="batch_sentiment_analysis_report.pdf",
                        mime="application/pdf",
                        key="download_pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")

            st.caption("Choose your preferred format to download the analysis results.")
        
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

