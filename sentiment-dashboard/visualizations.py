import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st

# Define consistent color scheme
SENTIMENT_COLORS = {
    'Positive': '#2ecc71',  # Green
    'Neutral': '#95a5a6',   # Gray
    'Negative': '#e74c3c'   # Red
}

def plot_sentiment_distribution(df, plot_type='bar'):
    """
    Create sentiment distribution plot using Plotly.
    Returns a Plotly figure object.
    """
    sentiment_counts = df['sentiment'].value_counts()
    
    # Ensure specific order
    order = ['Positive', 'Neutral', 'Negative']
    counts = [sentiment_counts.get(sent, 0) for sent in order]
    colors = [SENTIMENT_COLORS[sent] for sent in order]

    if plot_type == 'bar':
        fig = go.Figure(data=[
            go.Bar(
                x=order,
                y=counts,
                marker_color=colors,
                text=counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Sentiment Distribution',
            xaxis_title='Sentiment',
            yaxis_title='Count',
            template='plotly_white',
            showlegend=False,
            yaxis_gridcolor='rgba(0,0,0,0.1)',
            plot_bgcolor='white'
        )
        
    elif plot_type in ['pie', 'donut']:
        fig = go.Figure(data=[
            go.Pie(
                labels=order,
                values=counts,
                marker_colors=colors,
                textinfo='percent',
                hoverinfo='label+value',
                hole=0.5 if plot_type == 'donut' else 0
            )
        ])
        
        fig.update_layout(
            title='Sentiment Distribution',
            template='plotly_white',
            showlegend=True
        )
        
    elif plot_type == 'line':
        if 'timestamp' not in df.columns:
            df = df.copy()
            df['timestamp'] = pd.date_range(start='today', periods=len(df), freq='H')
        
        sentiment_pivot = pd.crosstab(df['timestamp'], df['sentiment'])
        
        fig = go.Figure()
        
        for sentiment in order:
            if sentiment in sentiment_pivot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_pivot.index,
                        y=sentiment_pivot[sentiment],
                        name=sentiment,
                        line=dict(color=SENTIMENT_COLORS[sentiment]),
                        mode='lines+markers'
                    )
                )
        
        fig.update_layout(
            title='Sentiment Trends Over Time',
            xaxis_title='Time',
            yaxis_title='Count',
            template='plotly_white',
            showlegend=True,
            yaxis_gridcolor='rgba(0,0,0,0.1)',
            plot_bgcolor='white'
        )
    
    # Update common layout properties
    fig.update_layout(
        font=dict(size=12),
        title_x=0.5,
        margin=dict(t=50, l=50, r=50, b=50)
    )
    
    return fig

def generate_wordcloud(texts, sentiments=None):
    """
    Generate a word cloud from texts.
    """
    if isinstance(texts, str):
        texts = [texts]
    
    # Combine all texts
    text = ' '.join(texts)
    
    # Create and generate a word cloud image
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        max_words=100,
        colormap='viridis',
        prefer_horizontal=0.7,
        collocations=False
    ).generate(text)
    
    # Display the word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    return buf

def convert_plotly_fig_to_bytes(fig):
    """Convert a plotly figure to bytes buffer for PDF export"""
    img_bytes = fig.to_image(format="png", engine="kaleido")
    buf = BytesIO(img_bytes)
    return buf 