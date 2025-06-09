import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from transformers import pipeline
from keybert import KeyBERT
import streamlit as st
from visualizations import (
    plot_sentiment_distribution,
    generate_wordcloud
)

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
        Get or load a model with caching.
        
        Args:
            model_name: Name of the model to load
        """
        if model_name not in cls._models:
            if model_name == "sentiment":
                cls._models[model_name] = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment"
                )
            elif model_name == "keyword":
                cls._models[model_name] = KeyBERT()
        return cls._models[model_name]

class BatchProcessor:
    """
    Optimizes batch processing of text data.
    """
    
    @staticmethod
    @timed_cache(ttl=300)  # 5-minute cache for batch results
    def process_batch(texts: List[str], batch_size: int = 32) -> pd.DataFrame:
        """
        Process a batch of texts efficiently.
        
        Args:
            texts: List of texts to process
            batch_size: Size of processing batches
        """
        results = []
        sentiment_model = ModelManager.get_model("sentiment")
        keyword_model = ModelManager.get_model("keyword")
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Parallel sentiment analysis
            sentiment_results = sentiment_model(batch)
            
            # Process each result
            for text, result in zip(batch, sentiment_results):
                score = int(result['label'].split()[0])
                sentiment = "Negative" if score <= 2 else "Neutral" if score == 3 else "Positive"
                
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
                    'confidence': round(result['score'], 3),
                    'raw_score': score,
                    'keywords': ', '.join([k[0] for k in keywords])
                })
        
        return pd.DataFrame(results)

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
                return plot_sentiment_distribution(data, **kwargs)
            elif viz_type == "wordcloud":
                return generate_wordcloud(data['text'].tolist())
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