import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import functions
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
from visualizations import create_sentiment_distribution, create_confidence_chart, create_keyword_importance
from optimization import (
    optimize_memory_usage,
    BatchProcessor,
    compute_metrics,
    filter_high_confidence
)

# Cache initialization
@st.cache_resource
def initialize_models():
    """Initialize and cache ML models"""
    try:
        # Models are initialized in utils.py
        return True
    except Exception as e:
        st.error(f"Failed to initialize models: {str(e)}")
        return False

@st.cache_data(ttl=3600)
def cached_sentiment_analysis(text: str):
    return analyze_sentiment(text)

@st.cache_data(ttl=3600)
def cached_keyword_extraction(text: str):
    return extract_keywords(text)

@st.cache_data(ttl=3600)
def cached_visualization(data: pd.DataFrame, viz_type: str, **kwargs):
    if viz_type == "sentiment_distribution":
        return create_sentiment_distribution(data, **kwargs)
    elif viz_type == "confidence_chart":
        return create_confidence_chart(data, **kwargs)
    elif viz_type == "keyword_importance":
        return create_keyword_importance(data, **kwargs)
    elif viz_type == "wordcloud":
        from visualizations import create_wordcloud
        return create_wordcloud(data, **kwargs)
    return None

# Initialize models
if not initialize_models():
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main > div {
        background-color: #f9fafb;
        padding: 2rem 1rem;
    }
    
    .stTextArea > div > div > textarea {
        background-color: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        font-size: 1rem;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: border-color 0.2s;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    .metric-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .css-1d391kg {
        background-color: #2c3e50;
        color: white;
    }
    
    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }
    
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
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6c757d;
        font-size: 1.2rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Header
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
    # Single text input with enhanced error handling
    text_input = st.text_area("Enter text to analyze:", height=150, 
                             help="Enter any text for sentiment analysis. Minimum 3 characters, maximum 5000 characters.")
    
    if text_input:
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
                        
                        # Display results
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
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Predefined question buttons
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("Why this sentiment?", use_container_width=True):
                                explanation_text = handle_followup_question("Why was this labeled this way?", text_input, result, keywords)
                                st.markdown("### 💡 Explanation")
                                st.markdown(explanation_text)
                        
                        with col2:
                            if st.button("What keywords influenced this?", use_container_width=True):
                                explanation_text = handle_followup_question("What keywords caused this result?", text_input, result, keywords)
                                st.markdown("### 🔍 Keyword Analysis")
                                st.markdown(explanation_text)
                        
                        with col3:
                            if st.button("How confident is this?", use_container_width=True):
                                explanation_text = handle_followup_question("How confident is this result?", text_input, result, keywords)
                                st.markdown("### 📊 Confidence Analysis")
                                st.markdown(explanation_text)
                        
                        # Custom question input
                        custom_question = st.text_input(
                            "Ask your own question:",
                            placeholder="e.g., Why is this negative? What made you choose this classification?"
                        )
                        
                        if custom_question:
                            if st.button("Get Answer", type="primary"):
                                explanation_text = handle_followup_question(custom_question, text_input, result, keywords)
                                st.markdown("### 🎯 Answer")
                                st.markdown(f"**Q: {custom_question}**")
                                st.markdown(explanation_text)
                
            except Exception as e:
                display_error_with_help(f"Unexpected error occurred: {str(e)}", "general")

with tab2:
    st.markdown("### 📚 Batch Analysis")
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
            # Enhanced batch processing with better error handling
            try:
                # Process texts with validation
                first_col = df.iloc[:, 0] if file_type == "csv" else df["text"]
                valid_texts = []
                invalid_count = 0
                
                for text in first_col.tolist():
                    if validate_text_input(str(text)) is None:
                        valid_texts.append(text)
                    else:
                        invalid_count += 1
                
                if invalid_count > 0:
                    display_warning_with_action(
                        f"Found {invalid_count} invalid entries that will be skipped.",
                        ["Review your data for empty or invalid text entries", "Ensure all text meets the minimum requirements"]
                    )
                
                if not valid_texts:
                    display_error_with_help("No valid texts found for analysis.", "validation")
                else:
                    # Process with progress tracking
                    with st.spinner(f"🚀 Analyzing {len(valid_texts)} texts..."):
                        try:
                            results_df = BatchProcessor.process_batch(valid_texts)
                            
                            # Check for processing errors
                            if results_df is None or results_df.empty:
                                display_error_with_help("Batch processing failed to produce results.", "processing")
                            else:
                                # Success with details
                                success_details = {
                                    "Total Processed": len(results_df),
                                    "Valid Texts": len(valid_texts),
                                    "Processing Success Rate": f"{len(results_df)/len(valid_texts)*100:.1f}%"
                                }
                                display_success_with_details("Batch analysis completed successfully!", success_details)
                                
                                # Display results with enhanced error handling
                                try:
                                    # Create tabs for different views
                                    results_tab1, results_tab2, results_tab3 = st.tabs(["Results Table", "Visualizations", "Export"])
                                    
                                    with results_tab1:
                                        st.subheader("Analysis Results")
                                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                                    
                                    with results_tab2:
                                        st.subheader("Sentiment Distribution")
                                        chart_type = st.selectbox("Select chart type", ["bar", "pie", "donut"])
                                        
                                        try:
                                            fig = cached_visualization(results_df, "sentiment_distribution", plot_type=chart_type)
                                            if fig is not None:
                                                st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                st.warning("Could not generate visualization.")
                                        except Exception as viz_error:
                                            st.error(f"Visualization error: {str(viz_error)}")
                                    
                                    with results_tab3:
                                        st.subheader("Export Options")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.download_button(
                                                label="📊 Download CSV",
                                                data=results_df.to_csv(index=False),
                                                file_name="sentiment_results.csv",
                                                mime="text/csv",
                                                use_container_width=True
                                            )
                                        
                                        with col2:
                                            st.download_button(
                                                label="🔗 Download JSON",
                                                data=results_df.to_json(orient="records"),
                                                file_name="sentiment_results.json",
                                                mime="application/json",
                                                use_container_width=True
                                            )
                                        
                                        with col3:
                                            try:
                                                pdf_buffer = export_to_pdf(results_df, {})
                                                st.download_button(
                                                    label="📋 Download PDF",
                                                    data=pdf_buffer.getvalue(),
                                                    file_name="sentiment_results.pdf",
                                                    mime="application/pdf",
                                                    use_container_width=True
                                                )
                                            except Exception as pdf_error:
                                                st.error(f"PDF generation failed: {str(pdf_error)}")
                                
                                except Exception as display_error:
                                    display_error_with_help(f"Error displaying results: {str(display_error)}", "general")
                        
                        except Exception as batch_error:
                            display_error_with_help(f"Batch processing failed: {str(batch_error)}", "processing")
            
            except Exception as e:
                display_error_with_help(f"Unexpected error during processing: {str(e)}", "general")

with tab3:
    st.markdown("### 🔄 Comparative Analysis")
    st.markdown("Compare sentiment analysis results for multiple texts side by side.")
    
    num_texts = st.number_input("Number of texts to compare", min_value=2, max_value=5, value=2)
    
    texts = []
    for i in range(num_texts):
        text = st.text_area(f"Text {i+1}", height=100, key=f"text_{i}")
        if text:
            # Validate each text
            validation_error = validate_text_input(text)
            if validation_error:
                st.error(f"Text {i+1}: {validation_error}")
            else:
                texts.append(text)
    
    if len(texts) >= 2:
        if st.button("Compare Texts", type="primary"):
            try:
                with st.spinner("Analyzing texts for comparison..."):
                    comparison_results = []
                    for i, text in enumerate(texts):
                        result = safe_sentiment_analysis(text)
                        if 'error' not in result:
                            comparison_results.append({
                                'Text': f"Text {i+1}",
                                'Content': text[:100] + "..." if len(text) > 100 else text,
                                'Sentiment': result['sentiment'],
                                'Confidence': result['confidence'],
                                'Use Case': result.get('use_case', 'General')
                            })
                    
                    if comparison_results:
                        comparison_df = pd.DataFrame(comparison_results)
                        st.subheader("Comparison Results")
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Create comparison visualization
                        try:
                            fig = px.bar(comparison_df, x='Text', y='Confidence', 
                                       color='Sentiment', title="Sentiment Comparison")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as viz_error:
                            st.warning(f"Could not create comparison chart: {str(viz_error)}")
                    else:
                        display_error_with_help("No texts could be analyzed for comparison.", "processing")
            
            except Exception as e:
                display_error_with_help(f"Comparison analysis failed: {str(e)}", "general")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Tech Titanians - Sentiment Analysis Dashboard</p>
</div>
""", unsafe_allow_html=True) 