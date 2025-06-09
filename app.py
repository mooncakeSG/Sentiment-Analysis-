import streamlit as st
import pandas as pd
import json
import tempfile
from utils import (
    analyze_sentiment,
    extract_keywords,
    process_batch,
    plot_sentiment_distribution,
    export_to_pdf,
    get_download_link
)

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Make dataframes use full width */
    .st-emotion-cache-1v0mbdj {
        width: 100%;
    }
    
    /* Enhance text areas */
    .stTextArea textarea {
        font-size: 16px !important;
        line-height: 1.5;
        padding: 12px;
        border-radius: 8px;
    }
    
    /* Style file uploader */
    .stUploadedFile {
        border-radius: 8px;
        padding: 16px;
        background-color: #f8f9fa;
        margin-bottom: 16px;
    }
    
    /* Enhance metrics */
    .stMetric {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Style download buttons */
    .stDownloadButton {
        width: 100%;
        border-radius: 8px;
        margin: 4px 0;
    }
    
    /* Enhance tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        gap: 8px;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 Sentiment Analysis Dashboard")
st.markdown("""
This dashboard analyzes the sentiment of text using state-of-the-art natural language processing.
You can either enter a single text or upload a CSV file for batch processing.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This dashboard uses:
- Hugging Face Transformers for sentiment analysis
- KeyBERT for keyword extraction
- Streamlit for the user interface
""")

# Main content
tab1, tab2 = st.tabs(["Single Text Analysis", "Batch Analysis"])

with tab1:
    # Single text input
    text_input = st.text_area("Enter text to analyze:", height=150)
    
    if text_input:
        with st.spinner("Analyzing text..."):
            # Get sentiment and keywords
            result = analyze_sentiment(text_input)
            keywords = extract_keywords(text_input)
            
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
            
            with col2:
                st.subheader("Keywords")
                st.write(", ".join(keywords))
            
            # Create DataFrame for download
            df = pd.DataFrame([{
                'text': text_input,
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'keywords': ', '.join(keywords)
            }])
            
            # Export options
            st.subheader("Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="sentiment_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_str = df.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="sentiment_analysis.json",
                    mime="application/json"
                )
            
            with col3:
                # Create PDF
                plot_buf = plot_sentiment_distribution(df)
                pdf_path = export_to_pdf(df, plot_buf)
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_file,
                        file_name="sentiment_analysis.pdf",
                        mime="application/pdf"
                    )

with tab2:
    st.markdown("""
    ### Batch Analysis
    Upload a CSV file containing texts to analyze. The file should have:
    - At least one column
    - Text data in the first column
    - No empty cells
    """)
    
    # File upload with additional instructions
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
            
            # Read CSV with progress indication
            with st.spinner("Reading CSV file..."):
                df = pd.read_csv(uploaded_file)
            
            # Process batch with progress bar
            with st.spinner("Processing texts..."):
                try:
                    results_df = process_batch(df)
                    
                    # Success message
                    st.success(f"Successfully analyzed {len(results_df)} texts!")
                    
                    # Display results in tabs
                    results_tab1, results_tab2 = st.tabs(["Results Table", "Visualization"])
                    
                    with results_tab1:
                        st.subheader("Analysis Results")
                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with results_tab2:
                        st.subheader("Sentiment Distribution")
                        plot_buf = plot_sentiment_distribution(results_df)
                        st.image(plot_buf)
                        
                        # Display summary statistics
                        st.markdown("### Summary Statistics")
                        sentiment_counts = results_df['sentiment'].value_counts()
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Positive", sentiment_counts.get('Positive', 0))
                        with col2:
                            st.metric("Neutral", sentiment_counts.get('Neutral', 0))
                        with col3:
                            st.metric("Negative", sentiment_counts.get('Negative', 0))
                    
                    # Export options in an expander
                    with st.expander("Export Options", expanded=True):
                        st.markdown("Download the analysis results in your preferred format:")
                        export_col1, export_col2, export_col3 = st.columns(3)
                        
                        with export_col1:
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📄 Download CSV",
                                data=csv,
                                file_name="batch_sentiment_analysis.csv",
                                mime="text/csv",
                                help="Download results as CSV file"
                            )
                        
                        with export_col2:
                            json_str = results_df.to_json(orient="records")
                            st.download_button(
                                label="📋 Download JSON",
                                data=json_str,
                                file_name="batch_sentiment_analysis.json",
                                mime="application/json",
                                help="Download results as JSON file"
                            )
                        
                        with export_col3:
                            # Create PDF
                            pdf_path = export_to_pdf(results_df, plot_buf)
                            with open(pdf_path, "rb") as pdf_file:
                                st.download_button(
                                    label="📑 Download PDF Report",
                                    data=pdf_file,
                                    file_name="batch_sentiment_analysis.pdf",
                                    mime="application/pdf",
                                    help="Download complete report as PDF"
                                )
                
                except ValueError as ve:
                    st.error(f"❌ Validation Error: {str(ve)}")
                    st.info("Please check your CSV file format and try again.")
                    
                except Exception as e:
                    st.error(f"❌ Processing Error: {str(e)}")
                    st.info("If this error persists, please contact support.")
            
        except Exception as e:
            st.error(f"❌ File Error: Could not read the CSV file.")
            st.info("Make sure your file is a valid CSV file with proper formatting.")
            st.code(str(e), language="python")

# Footer
st.markdown("---")
