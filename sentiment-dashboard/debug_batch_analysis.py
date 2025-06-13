import pandas as pd
import streamlit as st
import traceback
import time
from transformers import pipeline
from keybert import KeyBERT

def debug_batch_processing():
    """Debug version of batch processing to identify hanging issues"""
    
    st.title("🔍 Batch Processing Debug Tool")
    st.write("This tool helps identify where batch processing is hanging.")
    
    # Test with sample data
    sample_texts = [
        "This is great!",
        "Not bad at all",
        "Could be better",
        "Terrible experience",
        "Outstanding quality"
    ]
    
    if st.button("Test Batch Processing"):
        st.write("Starting debug process...")
        
        # Step 1: Test model loading
        st.write("📋 Step 1: Loading sentiment model...")
        try:
            start_time = time.time()
            sentiment_model = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            load_time = time.time() - start_time
            st.success(f"✅ Sentiment model loaded in {load_time:.2f} seconds")
        except Exception as e:
            st.error(f"❌ Failed to load sentiment model: {str(e)}")
            st.write(traceback.format_exc())
            return
        
        # Step 2: Test keyword model
        st.write("📋 Step 2: Loading keyword model...")
        try:
            start_time = time.time()
            keyword_model = KeyBERT()
            load_time = time.time() - start_time
            st.success(f"✅ Keyword model loaded in {load_time:.2f} seconds")
        except Exception as e:
            st.error(f"❌ Failed to load keyword model: {str(e)}")
            st.write(traceback.format_exc())
            return
        
        # Step 3: Test sentiment analysis
        st.write("📋 Step 3: Testing sentiment analysis...")
        try:
            start_time = time.time()
            sentiment_results = sentiment_model(sample_texts)
            analysis_time = time.time() - start_time
            st.success(f"✅ Sentiment analysis completed in {analysis_time:.2f} seconds")
            st.write("Sample results:", sentiment_results[:2])
        except Exception as e:
            st.error(f"❌ Sentiment analysis failed: {str(e)}")
            st.write(traceback.format_exc())
            return
        
        # Step 4: Test keyword extraction
        st.write("📋 Step 4: Testing keyword extraction...")
        try:
            start_time = time.time()
            for i, text in enumerate(sample_texts[:2]):  # Test first 2 texts
                keywords = keyword_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=5
                )
                st.write(f"Text {i+1} keywords: {keywords}")
            
            keyword_time = time.time() - start_time
            st.success(f"✅ Keyword extraction completed in {keyword_time:.2f} seconds")
        except Exception as e:
            st.error(f"❌ Keyword extraction failed: {str(e)}")
            st.write(traceback.format_exc())
            return
        
        # Step 5: Test full processing
        st.write("📋 Step 5: Testing full batch processing...")
        try:
            start_time = time.time()
            results = []
            
            # Process each text
            for i, (text, sentiment_result) in enumerate(zip(sample_texts, sentiment_results)):
                st.write(f"Processing text {i+1}/{len(sample_texts)}: {text}")
                
                # Parse sentiment result
                score = int(sentiment_result['label'].split()[0])
                
                # Map to sentiment
                if score == 1:
                    sentiment = "Very Negative"
                elif score == 2:
                    sentiment = "Negative"
                elif score == 3:
                    sentiment = "Neutral"
                elif score == 4:
                    sentiment = "Positive"
                else:
                    sentiment = "Very Positive"
                
                # Extract keywords
                keywords = keyword_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=5
                )
                
                results.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': round(sentiment_result['score'], 3),
                    'raw_score': score,
                    'keywords': ', '.join([k[0] for k in keywords])
                })
                
                st.write(f"✅ Text {i+1} processed successfully")
            
            full_time = time.time() - start_time
            st.success(f"✅ Full batch processing completed in {full_time:.2f} seconds")
            
            # Show results
            df = pd.DataFrame(results)
            st.dataframe(df)
            
        except Exception as e:
            st.error(f"❌ Full batch processing failed: {str(e)}")
            st.write(traceback.format_exc())
            return
        
        st.balloons()
        st.success("🎉 Debug test completed successfully!")

if __name__ == "__main__":
    debug_batch_processing()