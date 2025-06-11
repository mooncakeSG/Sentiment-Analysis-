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

# Import your existing modules
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
    page_title="Sentiment Analysis Dashboard - Fixed",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .gradient-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
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

def run_comparative_analysis(texts, labels):
    """Run the comparative analysis and return results"""
    comparison_results = []
    detailed_results = []
    
    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
            
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
    
    return comparison_results, detailed_results

def main():
    # Header
    st.markdown("""
    <div class="gradient-header">
        <h1>🎯 Comparative Sentiment Analysis - Working Version</h1>
        <p>Professional sentiment analysis with comparative insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Load sample data
    sample_data = load_sample_data()
    
    # Quick Start section
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
            load_samples = st.button("📦 Load Sample Pack", use_container_width=True)
    
    # Configuration
    st.markdown("### ⚙️ Configuration")
    num_texts = st.slider("Number of texts to compare:", 2, 5, 3)
    
    # Text input section
    st.markdown("### 📝 Text Input")
    
    # Initialize text containers
    texts = []
    labels = []
    
    # Check if samples should be loaded
    load_sample_texts = []
    load_sample_labels = []
    
    if sample_data and selected_category != "Select a category..." and load_samples:
        samples = sample_data[selected_category]
        for i, sample in enumerate(samples[:num_texts]):
            load_sample_texts.append(sample['text'])
            load_sample_labels.append(sample['label'])
        st.success(f"✅ Loaded {selected_category} samples!")
    
    # Create text inputs
    for i in range(num_texts):
        with st.container():
            st.markdown(f"**Text {i+1}:**")
            col_text, col_label = st.columns([4, 1])
            
            with col_text:
                # Use loaded sample if available
                default_text = load_sample_texts[i] if i < len(load_sample_texts) else ""
                text = st.text_area(
                    f"Content for Text {i+1}",
                    value=default_text,
                    height=120,
                    key=f"text_area_{i}",
                    placeholder=f"Enter text {i+1} for comparison analysis..."
                )
            
            with col_label:
                # Use loaded sample label if available
                default_label = load_sample_labels[i] if i < len(load_sample_labels) else f"Text {i+1}"
                label = st.text_input(
                    f"Label {i+1}",
                    value=default_label,
                    key=f"text_label_{i}",
                    help="Custom label for this text"
                )
            
            # Validate and collect
            if text and text.strip():
                validation_error = validate_text_input(text)
                if validation_error:
                    st.error(f"❌ {validation_error}")
                else:
                    texts.append(text)
                    labels.append(label if label else f"Text {i+1}")
                    st.success(f"✅ Text {i+1} ready for analysis")
    
    # Analysis section
    if len(texts) >= 2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Analysis button
        if st.button("🚀 Run Comparative Analysis", type="primary", use_container_width=True):
            try:
                with st.spinner("🔍 Performing comprehensive comparative analysis..."):
                    # Run analysis
                    comparison_results, detailed_results = run_comparative_analysis(texts, labels)
                    
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
                        
                        # Tabbed results
                        tab1, tab2, tab3 = st.tabs(["📋 Summary Table", "📊 Visualizations", "🔍 Detailed Analysis"])
                        
                        with tab1:
                            st.subheader("📋 Comparison Summary")
                            st.dataframe(comparison_df.drop('Content_Preview', axis=1), use_container_width=True, hide_index=True)
                            
                            st.subheader("📖 Content Preview")
                            for i, row in comparison_df.iterrows():
                                with st.expander(f"📄 {row['Label']} - {row['Sentiment']} ({row['Confidence']:.1%} confidence)"):
                                    st.write(row['Content_Preview'])
                        
                        with tab2:
                            st.subheader("📊 Comparative Visualizations")
                            
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
                            
                            # Scatter plot
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
                            scatter_fig.update_layout(height=400)
                            st.plotly_chart(scatter_fig, use_container_width=True)
                        
                        with tab3:
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
                                        
                                        reliability = item['explanation']['reliability']
                                        reliability_color = {
                                            'Very High': '🟢',
                                            'High': '🟢', 
                                            'Good': '🟡',
                                            'Moderate': '🟠',
                                            'Low': '🔴'
                                        }.get(reliability, '⚪')
                                        st.markdown(f"**Reliability:** {reliability_color} {reliability}")
                        
                        # Export section
                        st.markdown("---")
                        st.markdown("### 📤 Export Results")
                        
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
                st.error(f"Debug info: {str(e)}")
    
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

if __name__ == "__main__":
    main() 