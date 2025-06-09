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
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .st-emotion-cache-1v0mbdj {
        width: 100%;
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
- 🤗 Hugging Face Transformers for sentiment analysis
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
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            if len(df.columns) < 1:
                st.error("The CSV file must have at least one column containing text.")
            else:
                with st.spinner("Processing batch analysis..."):
                    # Process the batch
                    results_df = process_batch(df)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    st.dataframe(results_df)
                    
                    # Plot sentiment distribution
                    st.subheader("Sentiment Distribution")
                    plot_buf = plot_sentiment_distribution(results_df)
                    st.image(plot_buf)
                    
                    # Export options
                    st.subheader("Export Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="batch_sentiment_analysis.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        json_str = results_df.to_json(orient="records")
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name="batch_sentiment_analysis.json",
                            mime="application/json"
                        )
                    
                    with col3:
                        # Create PDF
                        pdf_path = export_to_pdf(results_df, plot_buf)
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="Download PDF",
                                data=pdf_file,
                                file_name="batch_sentiment_analysis.pdf",
                                mime="application/pdf"
                            )
                            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Hugging Face") 