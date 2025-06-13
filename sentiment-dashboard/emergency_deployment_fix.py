"""
Emergency deployment fix - forces safe processing for large batches
Use this as a guaranteed fallback when environment detection fails
"""

import streamlit as st
import pandas as pd
import time
import gc
from typing import List

def emergency_batch_processor(texts: List[str]) -> pd.DataFrame:
    """
    Emergency batch processor that always uses safe processing
    regardless of environment detection
    """
    if not texts:
        return pd.DataFrame()
    
    # Always limit for safety
    max_texts = 50
    if len(texts) > max_texts:
        st.warning(f"⚠️ EMERGENCY MODE: Limited to {max_texts} texts for stability. You have {len(texts)} texts.")
        texts = texts[:max_texts]
    
    # Simple sentiment words
    positive_words = {
        'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'perfect', 
        'outstanding', 'brilliant', 'superb', 'awesome', 'love', 'best', 'incredible',
        'marvelous', 'terrific', 'delightful', 'pleased', 'satisfied', 'happy'
    }
    
    negative_words = {
        'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'worst', 'bad',
        'disappointing', 'poor', 'useless', 'pathetic', 'dreadful', 'appalling',
        'atrocious', 'abysmal', 'deplorable', 'frustrated', 'angry', 'annoyed'
    }
    
    results = []
    total_texts = len(texts)
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    st.info("🚨 EMERGENCY PROCESSING MODE: Using ultra-safe lightweight analysis")
    
    try:
        for i, text in enumerate(texts):
            # Update progress
            progress = (i + 1) / total_texts
            progress_bar.progress(progress)
            status_text.text(f"Processing text {i + 1} of {total_texts}...")
            
            try:
                # Truncate very long texts
                if len(text) > 300:
                    text = text[:300] + "..."
                
                # Simple sentiment analysis
                text_lower = text.lower()
                text_words = set(text_lower.split())
                
                positive_matches = len(text_words.intersection(positive_words))
                negative_matches = len(text_words.intersection(negative_words))
                
                if positive_matches > negative_matches:
                    sentiment = "Positive"
                    confidence = min(0.9, 0.6 + (positive_matches - negative_matches) * 0.1)
                elif negative_matches > positive_matches:
                    sentiment = "Negative" 
                    confidence = min(0.9, 0.6 + (negative_matches - positive_matches) * 0.1)
                else:
                    sentiment = "Neutral"
                    confidence = 0.5
                
                # Simple keyword extraction
                words = text_lower.replace(',', ' ').replace('.', ' ').split()
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                keywords = [w for w in words if len(w) > 2 and w not in stop_words and w.isalpha()]
                top_keywords = list(set(keywords))[:3]
                
                # Simple use case
                if any(word in text_lower for word in ['product', 'buy', 'purchase']):
                    use_case = 'Product Review'
                elif any(word in text_lower for word in ['service', 'support', 'help']):
                    use_case = 'Customer Service'
                else:
                    use_case = 'General Analysis'
                
                results.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': round(confidence, 3),
                    'keywords': ', '.join(top_keywords) if top_keywords else 'none',
                    'use_case': use_case
                })
                
            except Exception as e:
                st.warning(f"⚠️ Failed to process text {i + 1}: {str(e)}")
                results.append({
                    'text': text,
                    'sentiment': 'Error',
                    'confidence': 0.0,
                    'keywords': 'processing failed',
                    'use_case': 'Error'
                })
            
            # Tiny delay for stability
            if i % 5 == 0:
                time.sleep(0.05)
                gc.collect()
    
    except Exception as e:
        st.error(f"❌ Emergency processing failed: {str(e)}")
        
    finally:
        progress_bar.empty()
        status_text.empty()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    successful_count = len(df[df['sentiment'] != 'Error'])
    
    if successful_count > 0:
        st.success(f"✅ Emergency processing completed! {successful_count}/{total_texts} texts processed.")
    else:
        st.error("❌ Emergency processing failed completely.")
    
    return df 