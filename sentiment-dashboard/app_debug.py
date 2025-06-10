import streamlit as st
import pandas as pd
from utils import safe_sentiment_analysis, safe_keyword_extraction, explain_sentiment, validate_text_input, display_error_with_help
import plotly.express as px

st.title("🔄 Debug Comparative Analysis")

# Simple text inputs
num_texts = st.number_input("Number of texts", min_value=2, max_value=5, value=3)

texts = []
text_labels = []

for i in range(num_texts):
    st.markdown(f"**Text {i+1}:**")
    
    # Pre-fill with sample data for testing
    sample_texts = [
        "This product exceeded all my expectations! The quality is outstanding, delivery was lightning fast, and customer service went above and beyond.",
        "Completely disappointed with this purchase. The product arrived damaged, took forever to ship, and customer service was unhelpful and rude.",
        "The product is okay, nothing special. It does what it's supposed to do but the quality could be better for the price."
    ]
    
    default_text = sample_texts[i] if i < len(sample_texts) else ""
    
    text = st.text_area(
        f"Content for Text {i+1}", 
        value=default_text,
        height=120, 
        key=f"debug_text_{i}",
        placeholder=f"Enter text {i+1}..."
    )
    
    label = st.text_input(
        f"Label {i+1}", 
        value=f"Sample {i+1}",
        key=f"debug_label_{i}"
    )
    
    if text:
        texts.append(text)
        text_labels.append(label)
        st.success(f"✅ Text {i+1} ready ({len(text)} characters)")

# Debug information
st.write(f"**Debug Info:**")
st.write(f"- Number of texts: {len(texts)}")
st.write(f"- Text lengths: {[len(t) for t in texts]}")

# Analysis button
if len(texts) >= 2:
    if st.button("🚀 Debug Run Analysis", type="primary"):
        st.write("🔍 Button clicked! Starting analysis...")
        
        try:
            # Simple analysis without complex error handling
            results = []
            
            for i, text in enumerate(texts):
                st.write(f"Analyzing text {i+1}...")
                
                # Test sentiment analysis
                result = safe_sentiment_analysis(text)
                st.write(f"Result: {result}")
                
                if 'error' not in result:
                    results.append({
                        'Label': text_labels[i],
                        'Text': text[:100] + "..." if len(text) > 100 else text,
                        'Sentiment': result['sentiment'],
                        'Confidence': result['confidence']
                    })
                    st.success(f"✅ Text {i+1} analyzed successfully")
                else:
                    st.error(f"❌ Error in text {i+1}: {result}")
            
            # Display results
            if results:
                st.success(f"🎉 Analysis complete! {len(results)} texts processed.")
                
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Simple visualization
                fig = px.bar(df, x='Label', y='Confidence', color='Sentiment', title="Results")
                st.plotly_chart(fig)
            else:
                st.error("No results generated")
                
        except Exception as e:
            st.error(f"Exception occurred: {str(e)}")
            st.exception(e)
else:
    st.warning(f"Need at least 2 texts to run analysis. Currently have {len(texts)} texts.") 