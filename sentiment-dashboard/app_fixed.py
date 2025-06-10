import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import io
from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules (assuming they work correctly)
try:
    from utils import (
        safe_sentiment_analysis, 
        safe_keyword_extraction, 
        validate_text_input,
        explain_sentiment
    )
    from visualizations import create_sentiment_chart, create_confidence_chart
    from optimization import optimize_analysis_performance, export_to_pdf
except ImportError as e:
    st.error(f"⚠️ Import Error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Enhanced header styling */
    .gradient-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    /* Enhanced metrics */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Button enhancements */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9ff 0%, #e6e9ff 100%);
    }
    
    /* Error and success message styling */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
    }
</style>
""", unsafe_allow_html=True)

# Load sample data
@st.cache_data
def load_sample_data():
    """Load sample data for comparative analysis"""
    try:
        with open('comparative_samples.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def display_error_with_help(error_message: str, error_type: str = "general"):
    """Display error with helpful suggestions"""
    st.error(f"❌ {error_message}")
    
    if error_type == "input":
        st.info("💡 **Suggestions:**\n- Check for special characters\n- Ensure text is not empty\n- Try shorter text")
    elif error_type == "processing":
        st.info("💡 **Suggestions:**\n- Try again in a moment\n- Check your internet connection\n- Simplify your text")
    else:
        st.info("💡 If this problem persists, please refresh the page and try again.")

def main():
    # Header with enhanced styling
    st.markdown("""
    <div class="gradient-header">
        <h1>🎯 Advanced Sentiment Analysis Dashboard</h1>
        <p>Professional sentiment analysis with comparative insights and actionable recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["🔍 Single Text Analysis", "📊 Comparative Analysis", "📈 Batch Processing"]
    )

    if page == "📊 Comparative Analysis":
        st.markdown("## 🔄 Comparative Sentiment Analysis")
        st.markdown("Compare multiple texts to understand sentiment patterns, confidence levels, and get actionable insights.")
        
        # Quick Start section with sample packs
        sample_data = load_sample_data()
        
        if sample_data:
            st.markdown("### 🚀 Quick Start")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_category = st.selectbox(
                    "Choose a Sample Pack:",
                    ["Select a category..."] + list(sample_data.keys()),
                    help="Pre-loaded sample texts for different analysis scenarios"
                )
            
            with col2:
                if st.button("📦 Load Sample Pack", use_container_width=True):
                    if selected_category != "Select a category...":
                        st.session_state['selected_samples'] = sample_data[selected_category]
                        st.session_state['sample_loaded'] = True
                        st.success(f"✅ Loaded {selected_category} samples!")
                        st.rerun()
        
        # Number of texts selector
        st.markdown("### ⚙️ Configuration")
        num_texts = st.slider("Number of texts to compare:", 2, 5, 3, help="Select how many texts you want to analyze")
        
        # Initialize session state for texts if not exists
        if 'texts_content' not in st.session_state:
            st.session_state['texts_content'] = [""] * 5
        if 'texts_labels' not in st.session_state:
            st.session_state['texts_labels'] = [f"Text {i+1}" for i in range(5)]
        
        # Text input section
        st.markdown("### 📝 Text Input")
        
        # Load sample data if requested
        if st.session_state.get('sample_loaded', False):
            samples = st.session_state.get('selected_samples', [])
            for i, sample in enumerate(samples[:num_texts]):
                st.session_state['texts_content'][i] = sample['text']
                st.session_state['texts_labels'][i] = sample['label']
            st.session_state['sample_loaded'] = False  # Reset flag
        
        # Create text inputs
        texts = []
        labels = []
        
        for i in range(num_texts):
            with st.container():
                st.markdown(f"**Text {i+1}:**")
                col_text, col_label = st.columns([4, 1])
                
                with col_text:
                    text = st.text_area(
                        f"Content for Text {i+1}",
                        value=st.session_state['texts_content'][i],
                        height=120,
                        key=f"comp_text_{i}",
                        placeholder=f"Enter text {i+1} for comparison analysis...",
                        on_change=lambda idx=i: setattr(st.session_state, f'texts_content', 
                                                      st.session_state['texts_content'][:idx] + 
                                                      [st.session_state[f'comp_text_{idx}']] +
                                                      st.session_state['texts_content'][idx+1:])
                    )
                
                with col_label:
                    label = st.text_input(
                        f"Label {i+1}",
                        value=st.session_state['texts_labels'][i],
                        key=f"comp_label_{i}",
                        help="Custom label for this text",
                        on_change=lambda idx=i: setattr(st.session_state, f'texts_labels',
                                                      st.session_state['texts_labels'][:idx] + 
                                                      [st.session_state[f'comp_label_{idx}']] +
                                                      st.session_state['texts_labels'][idx+1:])
                    )
                
                # Update session state with current values
                st.session_state['texts_content'][i] = text
                st.session_state['texts_labels'][i] = label
                
                # Validate and collect valid texts
                if text and text.strip():
                    validation_error = validate_text_input(text)
                    if validation_error:
                        st.error(f"❌ {validation_error}")
                    else:
                        texts.append(text)
                        labels.append(label)
                        st.success(f"✅ Text {i+1} ready for analysis")
        
        # Analysis button and results
        if len(texts) >= 2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Run Comparative Analysis", type="primary", use_container_width=True):
                try:
                    with st.spinner("🔍 Performing comprehensive comparative analysis..."):
                        # Process all texts for comparison
                        comparison_results = []
                        detailed_results = []
                        
                        for i, text in enumerate(texts):
                            result = safe_sentiment_analysis(text)
                            if 'error' not in result:
                                keywords = safe_keyword_extraction(text)
                                explanation = explain_sentiment(text, result)
                                
                                comparison_results.append({
                                    'Label': labels[i] if i < len(labels) else f"Text {i+1}",
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
                                    'label': labels[i] if i < len(labels) else f"Text {i+1}",
                                    'text': text,
                                    'result': result,
                                    'keywords': keywords,
                                    'explanation': explanation
                                })
                        
                        if comparison_results:
                            comparison_df = pd.DataFrame(comparison_results)
                            
                            # Display results
                            st.markdown("---")
                            st.markdown("## 📊 Comparative Analysis Results")
                            
                            # Summary metrics
                            st.markdown("### 🎯 Quick Insights")
                            
                            avg_confidence = comparison_df['Confidence'].mean()
                            sentiment_counts = comparison_df['Sentiment'].value_counts()
                            most_common_sentiment = sentiment_counts.index[0] if not sentiment_counts.empty else "Unknown"
                            confidence_range = comparison_df['Confidence'].max() - comparison_df['Confidence'].min()
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            
                            with metric_col1:
                                st.metric("📈 Average Confidence", f"{avg_confidence:.1%}")
                            with metric_col2:
                                st.metric("🏆 Dominant Sentiment", most_common_sentiment)
                            with metric_col3:
                                st.metric("📏 Confidence Range", f"{confidence_range:.1%}")
                            with metric_col4:
                                total_words = comparison_df['Word_Count'].sum()
                                st.metric("📝 Total Words", f"{total_words:,}")
                            
                            # Results table
                            st.subheader("📋 Comparison Summary")
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                            
                            # Visualizations
                            st.subheader("📊 Visualizations")
                            
                            viz_col1, viz_col2 = st.columns(2)
                            
                            with viz_col1:
                                conf_fig = px.bar(
                                    comparison_df, 
                                    x='Label', 
                                    y='Confidence', 
                                    color='Sentiment',
                                    title="Confidence Scores by Text"
                                )
                                st.plotly_chart(conf_fig, use_container_width=True)
                            
                            with viz_col2:
                                sent_fig = px.pie(
                                    values=sentiment_counts.values,
                                    names=sentiment_counts.index,
                                    title="Overall Sentiment Distribution"
                                )
                                st.plotly_chart(sent_fig, use_container_width=True)
                            
                            # Export options
                            st.subheader("📤 Export Results")
                            
                            export_col1, export_col2 = st.columns(2)
                            
                            with export_col1:
                                st.download_button(
                                    label="📊 Download CSV",
                                    data=comparison_df.to_csv(index=False),
                                    file_name=f"comparative_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
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
                                    label="🔗 Download JSON",
                                    data=json.dumps(comparison_json, indent=2),
                                    file_name=f"comparative_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                        
                        else:
                            display_error_with_help("No texts could be analyzed for comparison.", "processing")
                
                except Exception as e:
                    display_error_with_help(f"Analysis failed: {str(e)}", "general")
        
        else:
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

    elif page == "🔍 Single Text Analysis":
        st.markdown("## 🔍 Single Text Analysis")
        st.info("This feature is available in the main app. Use this fixed version specifically for comparative analysis.")
        
    elif page == "📈 Batch Processing":
        st.markdown("## 📈 Batch Processing")
        st.info("This feature is available in the main app. Use this fixed version specifically for comparative analysis.")

if __name__ == "__main__":
    main() 