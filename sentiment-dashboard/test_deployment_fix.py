"""
Test script for deployment fix
Run this to verify the batch processing works in deployment
"""

import streamlit as st
from deployment_fix import process_batch_deployment_safe, check_deployment_environment

def test_deployment_fix():
    st.title("🧪 Deployment Fix Test")
    
    # Test sample data
    sample_texts = [
        "This product is absolutely amazing! I love it so much.",
        "The service was terrible and disappointing.",
        "It's okay, nothing special but not bad either.",
        "Outstanding quality and fast delivery!",
        "Poor customer support, very frustrated.",
        "Great value for money, highly recommend!",
        "The food was delicious and the staff was friendly.",
        "Had to wait too long, not worth it.",
        "Perfect experience, will come back again!",
        "Could be better, but acceptable for the price."
    ]
    
    st.write("**Sample texts to process:**")
    for i, text in enumerate(sample_texts[:3], 1):
        st.write(f"{i}. {text}")
    st.write(f"... and {len(sample_texts) - 3} more texts")
    
    # Environment detection
    is_deployed = check_deployment_environment()
    st.write(f"**Environment:** {'Deployment' if is_deployed else 'Local'}")
    
    if st.button("🚀 Test Batch Processing"):
        st.write("---")
        
        try:
            # Test the deployment-safe processing
            results_df = process_batch_deployment_safe(sample_texts)
            
            if not results_df.empty:
                st.success("✅ Test completed successfully!")
                
                # Show results
                st.subheader("Results Preview")
                st.dataframe(results_df.head(), use_container_width=True)
                
                # Show statistics
                sentiment_counts = results_df['sentiment'].value_counts()
                st.subheader("Sentiment Distribution")
                st.bar_chart(sentiment_counts)
                
                # Show sample keywords
                st.subheader("Sample Keywords")
                keywords_sample = results_df['keywords'].iloc[:5].tolist()
                for i, keywords in enumerate(keywords_sample, 1):
                    st.write(f"Text {i}: {keywords}")
                    
            else:
                st.error("❌ Test failed - no results generated")
                
        except Exception as e:
            st.error(f"❌ Test failed with error: {str(e)}")
            st.write("This indicates there may still be issues with the deployment processing.")

if __name__ == "__main__":
    test_deployment_fix() 