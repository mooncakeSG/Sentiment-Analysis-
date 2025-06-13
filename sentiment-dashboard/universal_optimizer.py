"""
Universal Batch Processor Optimizer
Works for any amount of text with automatic deployment detection and optimization
"""

import streamlit as st
import pandas as pd
import time
import gc
import os
import sys
import socket
from typing import List
import requests

def detect_streamlit_cloud():
    """Enhanced detection specifically for Streamlit Cloud"""
    detection_methods = []
    
    # Method 1: Check hostname/FQDN
    try:
        hostname = socket.getfqdn().lower()
        if any(domain in hostname for domain in ['streamlit.app', 'streamlit.io', 'streamlit-cloud']):
            detection_methods.append("hostname")
    except:
        pass
    
    # Method 2: Check environment variables
    env_indicators = [
        'STREAMLIT_SHARING_MODE',
        'STREAMLIT_CLOUD', 
        'RENDER',
        'DYNO',  # Heroku
        'RAILWAY_ENVIRONMENT'
    ]
    
    for env_var in env_indicators:
        if os.getenv(env_var):
            detection_methods.append(f"env_{env_var}")
    
    # Method 3: Check if running in a container/cloud
    try:
        if os.path.exists('/.dockerenv'):
            detection_methods.append("docker")
    except:
        pass
    
    # Method 4: Check memory constraints (cloud usually < 2GB)
    try:
        import psutil
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        if total_memory < 2:
            detection_methods.append("memory_limit")
    except:
        pass
    
    # Method 5: Check for typical cloud paths
    cloud_paths = ['/app', '/mount/src', '/workspace']
    current_path = os.getcwd().lower()
    if any(path in current_path for path in cloud_paths):
        detection_methods.append("cloud_path")
    
    # Method 6: Check if we can't write to certain directories (cloud restriction)
    try:
        test_file = '/tmp/streamlit_test'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        # If we can only write to /tmp, likely in cloud
        if not os.access(os.path.expanduser('~'), os.W_OK):
            detection_methods.append("write_restrictions")
    except:
        detection_methods.append("write_restrictions")
    
    return detection_methods

def force_deployment_optimization():
    """Always use deployment optimization regardless of detection"""
    return True

class UniversalBatchProcessor:
    """Universal processor that works optimally for any amount of text"""
    
    def __init__(self):
        self.positive_keywords = {
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'perfect',
            'outstanding', 'brilliant', 'superb', 'awesome', 'love', 'best', 'incredible',
            'marvelous', 'terrific', 'delightful', 'pleased', 'satisfied', 'happy',
            'spectacular', 'phenomenal', 'exceptional', 'remarkable', 'magnificent',
            'fabulous', 'gorgeous', 'beautiful', 'stunning', 'impressive'
        }
        
        self.negative_keywords = {
            'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'worst', 'bad',
            'disappointing', 'poor', 'useless', 'pathetic', 'dreadful', 'appalling',
            'atrocious', 'abysmal', 'deplorable', 'frustrated', 'angry', 'annoyed',
            'furious', 'outraged', 'disgusted', 'revolting', 'repulsive', 'offensive',
            'unacceptable', 'intolerable', 'insufferable', 'unbearable'
        }
    
    def get_batch_size(self, total_texts):
        """Dynamic batch size based on total texts and environment"""
        detection_methods = detect_streamlit_cloud()
        
        if detection_methods or force_deployment_optimization():
            # Cloud environment - use smaller batches
            if total_texts <= 10:
                return 3
            elif total_texts <= 50:
                return 5
            elif total_texts <= 100:
                return 8
            else:
                return 10
        else:
            # Local environment - can use larger batches
            if total_texts <= 50:
                return 10
            elif total_texts <= 200:
                return 20
            else:
                return 30
    
    def analyze_sentiment_optimized(self, text):
        """Optimized sentiment analysis for any text length"""
        if not text or len(text.strip()) == 0:
            return "Neutral", 0.5
        
        # Normalize text
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        # Count sentiment words
        positive_matches = len(text_words.intersection(self.positive_keywords))
        negative_matches = len(text_words.intersection(self.negative_keywords))
        
        # Advanced scoring with context
        positive_score = positive_matches * 2
        negative_score = negative_matches * 2
        
        # Boost score for emphasis
        if '!' in text:
            if positive_matches > 0:
                positive_score += 1
            elif negative_matches > 0:
                negative_score += 1
        
        # Check for negation
        negation_words = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor'}
        has_negation = any(word in text_words for word in negation_words)
        
        if has_negation:
            # Flip scores if negation detected
            positive_score, negative_score = negative_score, positive_score
        
        # Determine sentiment
        if positive_score > negative_score:
            confidence = min(0.95, 0.6 + (positive_score - negative_score) * 0.05)
            return "Positive", confidence
        elif negative_score > positive_score:
            confidence = min(0.95, 0.6 + (negative_score - positive_score) * 0.05)
            return "Negative", confidence
        else:
            return "Neutral", 0.5
    
    def extract_keywords_fast(self, text, max_keywords=3):
        """Fast keyword extraction optimized for deployment"""
        if not text:
            return []
        
        # Enhanced stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'hers', 'our', 'their', 'get', 'got', 'very',
            'really', 'just', 'so', 'now', 'here', 'there', 'when', 'what',
            'how', 'why', 'who', 'which', 'than', 'too', 'also', 'then', 'than'
        }
        
        # Clean and tokenize
        text_clean = text.lower()
        for punct in '.,!?";()[]{}':
            text_clean = text_clean.replace(punct, ' ')
        
        words = text_clean.split()
        
        # Filter and count
        word_freq = {}
        for word in words:
            if (len(word) > 2 and 
                word not in stop_words and 
                word.isalpha() and 
                not word.isdigit()):
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in top_keywords[:max_keywords]]
    
    def determine_use_case_fast(self, text):
        """Fast use case determination"""
        text_lower = text.lower()
        
        # Quick keyword matching
        if any(word in text_lower for word in ['product', 'buy', 'purchase', 'quality', 'price']):
            return 'Product Review'
        elif any(word in text_lower for word in ['service', 'support', 'help', 'staff']):
            return 'Customer Service'
        elif any(word in text_lower for word in ['food', 'restaurant', 'meal', 'taste']):
            return 'Restaurant/Food'
        elif any(word in text_lower for word in ['hotel', 'room', 'stay', 'booking']):
            return 'Travel/Hotel'
        elif any(word in text_lower for word in ['post', 'tweet', 'share', 'social']):
            return 'Social Media'
        else:
            return 'General Analysis'
    
    def process_universal_batch(self, texts: List[str]) -> pd.DataFrame:
        """Universal batch processor optimized for any amount of text"""
        if not texts:
            return pd.DataFrame()
        
        # Detect environment
        detection_methods = detect_streamlit_cloud()
        is_cloud = bool(detection_methods) or force_deployment_optimization()
        
        # Show environment info
        if is_cloud:
            st.info(f"🚀 Cloud environment detected (methods: {', '.join(detection_methods)}) - using optimized processing")
            # Limit texts for cloud stability
            max_texts = 200
            if len(texts) > max_texts:
                st.warning(f"⚠️ Limited to {max_texts} texts for cloud stability. You have {len(texts)} texts.")
                texts = texts[:max_texts]
        else:
            st.info("💻 Local environment detected - using standard processing")
        
        # Dynamic batch processing
        batch_size = self.get_batch_size(len(texts))
        total_texts = len(texts)
        total_batches = (total_texts + batch_size - 1) // batch_size
        
        st.info(f"📊 Processing {total_texts} texts in {total_batches} batches (batch size: {batch_size})")
        
        results = []
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        try:
            for batch_idx in range(0, total_texts, batch_size):
                batch_texts = texts[batch_idx:batch_idx + batch_size]
                current_batch = batch_idx // batch_size + 1
                
                status_container.info(f"🔄 Processing batch {current_batch}/{total_batches} ({len(batch_texts)} texts)")
                
                # Process each text in the batch
                for i, text in enumerate(batch_texts):
                    overall_progress = (batch_idx + i) / total_texts
                    progress_bar.progress(overall_progress)
                    
                    try:
                        # Truncate very long texts for performance
                        if len(text) > 1000:
                            text = text[:1000] + "..."
                        
                        # Sentiment analysis
                        sentiment, confidence = self.analyze_sentiment_optimized(text)
                        
                        # Keywords
                        keywords = self.extract_keywords_fast(text, max_keywords=3)
                        
                        # Use case
                        use_case = self.determine_use_case_fast(text)
                        
                        results.append({
                            'text': text,
                            'sentiment': sentiment,
                            'confidence': round(confidence, 3),
                            'keywords': ', '.join(keywords) if keywords else 'none',
                            'use_case': use_case
                        })
                        
                    except Exception as e:
                        results.append({
                            'text': text,
                            'sentiment': 'Error',
                            'confidence': 0.0,
                            'keywords': 'processing failed',
                            'use_case': 'Error'
                        })
                        st.warning(f"⚠️ Failed to process text at position {batch_idx + i + 1}")
                
                # Memory cleanup between batches
                if current_batch % 3 == 0:  # Every 3 batches
                    gc.collect()
                    time.sleep(0.1)
                
        except Exception as e:
            st.error(f"❌ Batch processing error: {str(e)}")
        
        finally:
            progress_bar.progress(1.0)
            status_container.empty()
            progress_bar.empty()
        
        # Create optimized DataFrame
        df = pd.DataFrame(results)
        
        # Memory optimization
        for col in ['sentiment', 'use_case']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Results summary
        successful_count = len(df[df['sentiment'] != 'Error'])
        error_count = total_texts - successful_count
        
        if successful_count > 0:
            st.success(f"✅ Processing completed! {successful_count}/{total_texts} texts processed successfully.")
            if error_count > 0:
                st.warning(f"⚠️ {error_count} texts had processing errors.")
        else:
            st.error("❌ All texts failed to process.")
        
        return df

# Create global instance
universal_processor = UniversalBatchProcessor()