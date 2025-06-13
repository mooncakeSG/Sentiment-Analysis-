import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import streamlit as st
import gc
import psutil
import os

# Lightweight imports only when needed
def get_transformers():
    """Lazy import transformers to avoid memory issues"""
    try:
        from transformers import pipeline
        return pipeline
    except ImportError:
        st.error("❌ Transformers library not available")
        return None

def get_keybert():
    """Lazy import KeyBERT to avoid memory issues"""
    try:
        from keybert import KeyBERT
        return KeyBERT
    except ImportError:
        st.warning("⚠️ KeyBERT not available, keywords will be simplified")
        return None

def determine_use_case(text):
    """Lightweight use case determination"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['review', 'product', 'quality', 'bought']):
        return 'Product Review'
    elif any(word in text_lower for word in ['service', 'support', 'help', 'issue']):
        return 'Customer Service'
    elif any(word in text_lower for word in ['post', 'tweet', 'social', 'share']):
        return 'Social Media'
    else:
        return 'General Analysis'

class DeploymentOptimizedProcessor:
    """
    Optimized batch processor specifically designed for deployment constraints.
    Handles memory limitations, timeouts, and model loading issues.
    """
    
    _sentiment_model = None
    _keyword_model = None
    
    @classmethod
    def get_memory_usage(cls):
        """Get current memory usage"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0
    
    @classmethod
    def check_memory_limit(cls, limit_mb=500):
        """Check if memory usage is within limits"""
        current = cls.get_memory_usage()
        if current > limit_mb:
            st.warning(f"⚠️ High memory usage: {current:.1f}MB. Optimizing...")
            gc.collect()
            return False
        return True
    
    @classmethod
    def load_lightweight_sentiment_model(cls):
        """Load a lightweight sentiment model optimized for deployment"""
        if cls._sentiment_model is None:
            try:
                st.info("🔄 Loading lightweight sentiment model...")
                
                # Try to use a smaller, faster model for deployment
                pipeline_func = get_transformers()
                if pipeline_func:
                    cls._sentiment_model = pipeline_func(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        max_length=512,
                        truncation=True
                    )
                    st.success("✅ Lightweight sentiment model loaded!")
                else:
                    # Fallback to simple rule-based sentiment
                    cls._sentiment_model = "rule_based"
                    st.info("📋 Using rule-based sentiment analysis as fallback")
                    
            except Exception as e:
                st.warning(f"⚠️ Model loading failed, using rule-based fallback: {str(e)}")
                cls._sentiment_model = "rule_based"
                
        return cls._sentiment_model
    
    @classmethod
    def simple_sentiment_analysis(cls, text):
        """Simple rule-based sentiment analysis as fallback"""
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect', 'best', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disgusting', 'disappointing', 'poor', 'useless']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            if positive_count >= 2:
                return {'label': 'POSITIVE', 'score': 0.8}
            else:
                return {'label': 'POSITIVE', 'score': 0.6}
        elif negative_count > positive_count:
            if negative_count >= 2:
                return {'label': 'NEGATIVE', 'score': 0.8}
            else:
                return {'label': 'NEGATIVE', 'score': 0.6}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    @classmethod
    def extract_simple_keywords(cls, text, top_n=3):
        """Simple keyword extraction without complex models"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        words = text.lower().split()
        words = [word.strip('.,!?";()[]{}') for word in words if len(word) > 2]
        keywords = [word for word in words if word not in stop_words]
        
        # Count frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [word for word, freq in top_keywords]
    
    @classmethod
    def process_batch_optimized(cls, texts: List[str], max_batch_size: int = 5) -> pd.DataFrame:
        """
        Process texts in very small batches optimized for deployment constraints.
        """
        if not texts:
            return pd.DataFrame()
        
        # Limit total texts for deployment
        if len(texts) > 50:
            st.warning(f"⚠️ Processing limited to first 50 texts due to deployment constraints. You uploaded {len(texts)} texts.")
            texts = texts[:50]
        
        results = []
        sentiment_model = cls.load_lightweight_sentiment_model()
        
        # Process in very small batches
        total_texts = len(texts)
        processed = 0
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        try:
            for i in range(0, len(texts), max_batch_size):
                batch = texts[i:i + max_batch_size]
                batch_num = i // max_batch_size + 1
                total_batches = (len(texts) + max_batch_size - 1) // max_batch_size
                
                status_container.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
                
                # Check memory before processing each batch
                if not cls.check_memory_limit():
                    st.warning("⚠️ Memory limit reached, stopping processing")
                    break
                
                for j, text in enumerate(batch):
                    try:
                        # Truncate long texts
                        if len(text) > 1000:
                            text = text[:1000] + "..."
                        
                        # Sentiment analysis
                        if sentiment_model == "rule_based":
                            sentiment_result = cls.simple_sentiment_analysis(text)
                        else:
                            try:
                                sentiment_result = sentiment_model(text)[0]
                            except Exception:
                                sentiment_result = cls.simple_sentiment_analysis(text)
                        
                        # Map sentiment
                        label = sentiment_result['label'].upper()
                        if 'POSITIVE' in label or label == 'LABEL_2':
                            sentiment = "Positive"
                        elif 'NEGATIVE' in label or label == 'LABEL_0':
                            sentiment = "Negative"
                        else:
                            sentiment = "Neutral"
                        
                        # Simple keyword extraction
                        keywords = cls.extract_simple_keywords(text)
                        
                        # Use case
                        use_case = determine_use_case(text)
                        
                        results.append({
                            'text': text,
                            'sentiment': sentiment,
                            'confidence': round(sentiment_result['score'], 3),
                            'keywords': ', '.join(keywords),
                            'use_case': use_case
                        })
                        
                        processed += 1
                        progress = processed / total_texts
                        progress_bar.progress(progress)
                        
                    except Exception as text_error:
                        st.warning(f"⚠️ Failed to process text {processed + 1}: {str(text_error)}")
                        results.append({
                            'text': text,
                            'sentiment': 'Error',
                            'confidence': 0.0,
                            'keywords': 'processing failed',
                            'use_case': 'Error'
                        })
                        processed += 1
                        progress_bar.progress(processed / total_texts)
                
                # Small delay and memory cleanup between batches
                time.sleep(0.2)
                gc.collect()
                
        except Exception as e:
            st.error(f"❌ Batch processing error: {str(e)}")
        
        finally:
            progress_bar.empty()
            status_container.empty()
        
        # Create results DataFrame
        df = pd.DataFrame(results)
        
        # Memory optimization
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'text':
                df[col] = df[col].astype('category')
        
        successful_count = len(df[df['sentiment'] != 'Error'])
        st.success(f"✅ Processing completed! {successful_count}/{len(texts)} texts processed successfully.")
        
        if successful_count < len(texts):
            st.info(f"💡 {len(texts) - successful_count} texts had processing errors and are marked as 'Error'.")
        
        return df

class BatchProcessor:
    """
    Main batch processor that automatically chooses between full and optimized processing
    based on deployment environment.
    """
    
    @staticmethod
    def process_batch(texts: List[str], batch_size: int = 32) -> pd.DataFrame:
        """
        Automatically choose processing method based on environment
        """
        # Check if we're in a deployment environment
        is_deployed = (
            os.getenv('STREAMLIT_SHARING_MODE') == '1' or
            'streamlit' in os.getenv('HOME', '').lower() or
            os.getenv('DYNO') is not None or  # Heroku
            os.getenv('RAILWAY_ENVIRONMENT') is not None or  # Railway
            psutil.virtual_memory().total < 2 * 1024 * 1024 * 1024  # Less than 2GB RAM
        )
        
        if is_deployed:
            st.info("🚀 Detected deployment environment - using optimized processing")
            return DeploymentOptimizedProcessor.process_batch_optimized(texts, max_batch_size=3)
        else:
            st.info("💻 Local environment detected - using full processing")
            # Try to import the original processor
            try:
                from optimization import BatchProcessor as OriginalProcessor
                return OriginalProcessor.process_batch(texts, batch_size)
            except:
                st.warning("⚠️ Falling back to optimized processing")
                return DeploymentOptimizedProcessor.process_batch_optimized(texts, max_batch_size=5)

# Memory cleanup utility
def cleanup_memory():
    """Force memory cleanup"""
    gc.collect()
    if hasattr(gc, 'set_threshold'):
        gc.set_threshold(700, 10, 10)