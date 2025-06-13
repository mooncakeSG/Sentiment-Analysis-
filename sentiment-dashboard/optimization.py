import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from transformers import pipeline
from keybert import KeyBERT
import streamlit as st
from visualizations import (
    create_sentiment_distribution,
    generate_wordcloud
)
import threading
import signal
import os

# Import the determine_use_case function from utils
def determine_use_case(text):
    """
    Determine the most relevant use case for the analyzed text.
    """
    # Keywords associated with different use cases
    use_cases = {
        'social_media': ['post', 'tweet', 'comment', 'like', 'share', 'follow'],
        'customer_feedback': ['review', 'feedback', 'rating', 'experience', 'service'],
        'product_review': ['product', 'quality', 'price', 'feature', 'bought', 'purchased'],
        'brand_monitoring': ['brand', 'company', 'reputation', 'market', 'competitor'],
        'market_research': ['market', 'trend', 'industry', 'consumer', 'demand'],
        'customer_service': ['support', 'help', 'issue', 'problem', 'resolution'],
        'competitive_intel': ['competitor', 'versus', 'compared', 'alternative', 'better']
    }
    
    text_lower = text.lower()
    use_case_scores = {}
    
    for case, keywords in use_cases.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        use_case_scores[case] = score
    
    # Get the use case with highest keyword matches
    best_case = max(use_case_scores.items(), key=lambda x: x[1])[0]
    
    # Map to human-readable names
    use_case_names = {
        'social_media': 'Social Media Analysis',
        'customer_feedback': 'Customer Feedback Analysis',
        'product_review': 'Product Review Classification',
        'brand_monitoring': 'Brand Monitoring',
        'market_research': 'Market Research',
        'customer_service': 'Customer Service Optimization',
        'competitive_intel': 'Competitive Intelligence'
    }
    
    return use_case_names.get(best_case, 'General Analysis')

# Cache configuration
CACHE_TTL = 3600  # 1 hour cache time
MAX_CACHE_SIZE = 1000  # Maximum number of cached items

def timed_cache(ttl: int = CACHE_TTL, max_size: int = MAX_CACHE_SIZE):
    """
    A caching decorator that includes timing information and cache size management.
    
    Args:
        ttl: Time to live for cache entries in seconds
        max_size: Maximum number of items to keep in cache
    """
    def decorator(func: Callable) -> Callable:
        cache: Dict[str, Tuple[Any, float]] = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from function arguments
            key = str((args, sorted(kwargs.items())))
            
            # Check if result is in cache and not expired
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return result
            
            # Execute function and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Manage cache size
            if len(cache) >= max_size:
                # Remove oldest entry
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            # Store result with timestamp
            cache[key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator

class ModelManager:
    """
    Manages model loading and caching for better performance.
    """
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model(cls, model_name: str):
        """
        Get or load a model with caching and timeout handling.
        
        Args:
            model_name: Name of the model to load
        """
        if model_name not in cls._models:
            try:
                if model_name == "sentiment":
                    if 'streamlit' in globals():
                        st.info("🔄 Loading sentiment analysis model (first time may take 1-2 minutes)...")
                    
                    cls._models[model_name] = pipeline(
                        "sentiment-analysis",
                        model="nlptown/bert-base-multilingual-uncased-sentiment",
                        tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
                    )
                    
                    if 'streamlit' in globals():
                        st.success("✅ Sentiment model loaded successfully!")
                        
                elif model_name == "keyword":
                    if 'streamlit' in globals():
                        st.info("🔄 Loading keyword extraction model...")
                    
                    cls._models[model_name] = KeyBERT()
                    
                    if 'streamlit' in globals():
                        st.success("✅ Keyword model loaded successfully!")
                        
            except Exception as e:
                if 'streamlit' in globals():
                    st.error(f"❌ Failed to load {model_name} model: {str(e)}")
                raise e
                
        return cls._models[model_name]

class BatchProcessor:
    """
    Optimizes batch processing of text data with improved error handling and progress tracking.
    """
    
    @staticmethod
    def process_batch(texts: List[str], batch_size: int = 16) -> pd.DataFrame:
        """
        Process a batch of texts efficiently with improved error handling.
        
        Args:
            texts: List of texts to process
            batch_size: Size of processing batches (reduced for stability)
        """
        if not texts:
            return pd.DataFrame()
            
        results = []
        
        try:
            # Load models with progress indication
            sentiment_model = ModelManager.get_model("sentiment")
            keyword_model = ModelManager.get_model("keyword")
            
            # Process in smaller batches to prevent hanging
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(texts), batch_size):
                batch = texts[batch_idx:batch_idx + batch_size]
                current_batch = batch_idx // batch_size + 1
                
                if 'streamlit' in globals():
                    st.info(f"🔄 Processing batch {current_batch}/{total_batches} ({len(batch)} texts)...")
                
                try:
                    # Sentiment analysis with timeout protection
                    sentiment_results = sentiment_model(batch)
                    
                    # Process each result in the batch
                    for i, (text, sentiment_result) in enumerate(zip(batch, sentiment_results)):
                        try:
                            # Parse sentiment score
                            score = int(sentiment_result['label'].split()[0])
                            
                            # Map scores to five sentiment classes
                            if score == 1:
                                sentiment = "Very Negative"
                            elif score == 2:
                                sentiment = "Negative"
                            elif score == 3:
                                sentiment = "Neutral"
                            elif score == 4:
                                sentiment = "Positive"
                            else:  # score == 5
                                sentiment = "Very Positive"
                            
                            # Extract keywords with error handling
                            try:
                                keywords = keyword_model.extract_keywords(
                                    text,
                                    keyphrase_ngram_range=(1, 2),
                                    stop_words='english',
                                    top_n=3  # Reduced for performance
                                )
                                keyword_str = ', '.join([k[0] for k in keywords])
                            except Exception as keyword_error:
                                keyword_str = "keyword extraction failed"
                                if 'streamlit' in globals():
                                    st.warning(f"⚠️ Keyword extraction failed for text {batch_idx + i + 1}: {str(keyword_error)}")
                            
                            # Determine use case
                            try:
                                use_case = determine_use_case(text)
                            except Exception:
                                use_case = "General Analysis"
                            
                            results.append({
                                'text': text,
                                'sentiment': sentiment,
                                'confidence': round(sentiment_result['score'], 3),
                                'raw_score': score,
                                'keywords': keyword_str,
                                'use_case': use_case
                            })
                            
                        except Exception as text_error:
                            # Handle individual text processing errors
                            results.append({
                                'text': text,
                                'sentiment': 'Analysis Failed',
                                'confidence': 0.0,
                                'raw_score': 0,
                                'keywords': 'error',
                                'use_case': 'Error'
                            })
                            if 'streamlit' in globals():
                                st.warning(f"⚠️ Failed to process text {batch_idx + i + 1}: {str(text_error)}")
                                
                except Exception as batch_error:
                    # Handle batch processing errors
                    for text in batch:
                        results.append({
                            'text': text,
                            'sentiment': 'Batch Failed',
                            'confidence': 0.0,
                            'raw_score': 0,
                            'keywords': 'batch error',
                            'use_case': 'Error'
                        })
                    if 'streamlit' in globals():
                        st.error(f"❌ Batch {current_batch} failed: {str(batch_error)}")
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
                
        except Exception as model_error:
            if 'streamlit' in globals():
                st.error(f"❌ Model loading or initialization failed: {str(model_error)}")
            raise model_error
        
        df = pd.DataFrame(results)
        
        if 'streamlit' in globals():
            successful_count = len(df[~df['sentiment'].str.contains('Failed|Error', na=False)])
            st.success(f"✅ Batch processing completed! {successful_count}/{len(texts)} texts processed successfully.")
        
        return df

class VisualizationOptimizer:
    """
    Optimizes visualization generation and caching.
    """
    
    @staticmethod
    @timed_cache(ttl=300)  # 5-minute cache for visualizations
    def create_visualization(
        data: pd.DataFrame,
        viz_type: str,
        **kwargs
    ) -> Any:
        """
        Create and cache visualizations.
        
        Args:
            data: DataFrame containing visualization data
            viz_type: Type of visualization to create
            **kwargs: Additional visualization parameters
        """
        try:
            if viz_type == "sentiment_distribution":
                return create_sentiment_distribution(data, **kwargs)
            elif viz_type == "wordcloud":
                if 'text' in data.columns:
                    return generate_wordcloud(data['text'].tolist())
                else:
                    return None
            else:
                raise ValueError(f"Unknown visualization type: {viz_type}")
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            return None

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage.
    
    Args:
        df: DataFrame to optimize
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def handle_errors(func: Callable) -> Callable:
    """
    Error handling decorator for robust function execution.
    
    Args:
        func: Function to wrap with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper 