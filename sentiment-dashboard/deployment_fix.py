"""
Deployment-optimized batch processing for Streamlit Cloud
Fixes hanging issues with batch analysis in deployed environments
"""

import streamlit as st
import pandas as pd
import time
import gc
import os
from typing import List

def check_deployment_environment():
    """Detect if running in deployment environment"""
    deployment_indicators = [
        os.getenv('STREAMLIT_SHARING_MODE') == '1',
        'streamlit' in os.getenv('HOME', '').lower(),
        os.getenv('DYNO') is not None,  # Heroku
        os.getenv('RAILWAY_ENVIRONMENT') is not None,  # Railway
    ]
    return any(deployment_indicators)

class LightweightSentimentAnalyzer:
    """Lightweight sentiment analysis that works reliably in deployment"""
    
    def __init__(self):
        self.positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'love', 'perfect', 'best', 'awesome', 'outstanding', 'brilliant',
            'superb', 'magnificent', 'terrific', 'marvelous', 'exceptional'
        ]
        
        self.negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 
            'disgusting', 'disappointing', 'poor', 'useless', 'pathetic',
            'dreadful', 'appalling', 'atrocious', 'abysmal', 'deplorable'
        ]
    
    def analyze(self, text):
        """Simple but effective sentiment analysis"""
        text_lower = text.lower()
        
        positive_score = sum(2 if word in text_lower else 0 for word in self.positive_words)
        negative_score = sum(2 if word in text_lower else 0 for word in self.negative_words)
        
        # Add context-based scoring
        if '!' in text:
            if positive_score > 0:
                positive_score += 1
            elif negative_score > 0:
                negative_score += 1
        
        if positive_score > negative_score:
            confidence = min(0.9, 0.6 + (positive_score - negative_score) * 0.1)
            return "Positive", confidence
        elif negative_score > positive_score:
            confidence = min(0.9, 0.6 + (negative_score - positive_score) * 0.1)
            return "Negative", confidence
        else:
            return "Neutral", 0.5

class LightweightKeywordExtractor:
    """Simple keyword extraction without heavy dependencies"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
            'would', 'could', 'should', 'may', 'might', 'can', 'this', 
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'hers', 'our', 'their', 'am', 'get', 'got', 'very',
            'really', 'just', 'so', 'now', 'here', 'there', 'where',
            'when', 'what', 'how', 'why', 'who', 'which', 'than', 'too'
        }
    
    def extract(self, text, top_n=3):
        """Extract keywords from text"""
        # Clean and tokenize
        words = text.lower().replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ').split()
        
        # Filter words
        keywords = []
        for word in words:
            word = word.strip('.,!?";()[]{}')
            if len(word) > 2 and word not in self.stop_words and word.isalpha():
                keywords.append(word)
        
        # Count frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [word for word, freq in top_keywords if freq > 0]

def process_batch_deployment_safe(texts: List[str]) -> pd.DataFrame:
    """
    Process batch of texts safely for deployment environments
    """
    if not texts:
        return pd.DataFrame()
    
    # Limit processing for deployment stability
    max_texts = 100
    if len(texts) > max_texts:
        st.warning(f"⚠️ Processing limited to first {max_texts} texts for deployment stability. You have {len(texts)} texts.")
        texts = texts[:max_texts]
    
    # Initialize lightweight analyzers
    sentiment_analyzer = LightweightSentimentAnalyzer()
    keyword_extractor = LightweightKeywordExtractor()
    
    results = []
    total_texts = len(texts)
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, text in enumerate(texts):
            # Update progress
            progress = (i + 1) / total_texts
            progress_bar.progress(progress)
            status_text.text(f"Processing text {i + 1} of {total_texts}...")
            
            try:
                # Truncate very long texts
                if len(text) > 500:
                    text = text[:500] + "..."
                
                # Sentiment analysis
                sentiment, confidence = sentiment_analyzer.analyze(text)
                
                # Keyword extraction
                keywords = keyword_extractor.extract(text, top_n=3)
                
                # Simple use case determination
                text_lower = text.lower()
                if any(word in text_lower for word in ['product', 'quality', 'buy', 'purchase']):
                    use_case = 'Product Review'
                elif any(word in text_lower for word in ['service', 'support', 'help']):
                    use_case = 'Customer Service'
                elif any(word in text_lower for word in ['social', 'post', 'share']):
                    use_case = 'Social Media'
                else:
                    use_case = 'General Analysis'
                
                results.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'keywords': ', '.join(keywords) if keywords else 'none detected',
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
            
            # Small delay to prevent overwhelming the system
            if i % 10 == 0:
                time.sleep(0.1)
                gc.collect()  # Clean up memory periodically
    
    except Exception as e:
        st.error(f"❌ Batch processing failed: {str(e)}")
        
    finally:
        progress_bar.empty()
        status_text.empty()
    
    # Create DataFrame with memory optimization
    df = pd.DataFrame(results)
    
    # Optimize memory usage
    for col in ['sentiment', 'use_case']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    successful_count = len(df[df['sentiment'] != 'Error'])
    
    if successful_count > 0:
        st.success(f"✅ Batch processing completed! {successful_count}/{total_texts} texts processed successfully.")
        
        if successful_count < total_texts:
            error_count = total_texts - successful_count
            st.warning(f"⚠️ {error_count} texts had processing errors and are marked as 'Error'.")
    else:
        st.error("❌ All texts failed to process. Please check your data format.")
    
    return df

def determine_use_case_simple(text):
    """Simple use case determination without heavy processing"""
    text_lower = text.lower()
    
    use_case_keywords = {
        'Product Review': ['product', 'quality', 'buy', 'purchase', 'price', 'item', 'ordered'],
        'Customer Service': ['service', 'support', 'help', 'staff', 'representative', 'call', 'phone'],
        'Social Media': ['post', 'tweet', 'share', 'like', 'comment', 'follow', 'social'],
        'Restaurant/Food': ['food', 'restaurant', 'meal', 'dish', 'taste', 'flavor', 'dining'],
        'Travel/Hotel': ['hotel', 'room', 'stay', 'travel', 'vacation', 'trip', 'booking']
    }
    
    for use_case, keywords in use_case_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return use_case
    
    return 'General Analysis' 