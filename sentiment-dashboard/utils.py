import pandas as pd
import tempfile
import pdfkit
from transformers import pipeline
from keybert import KeyBERT
import matplotlib.pyplot as plt
import seaborn as sns
import json
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# Initialize models
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
keyword_model = KeyBERT()

# Define consistent color scheme
SENTIMENT_COLORS = {
    'Positive': '#2ecc71',  # Green
    'Neutral': '#95a5a6',   # Gray
    'Negative': '#e74c3c'   # Red
}

def analyze_sentiment(text):
    """
    Analyze sentiment of input text using transformers pipeline.
    Returns sentiment label and confidence score.
    """
    result = sentiment_analyzer(text)[0]
    # Convert 1-5 score to sentiment labels
    score = int(result['label'].split()[0])
    if score <= 2:
        sentiment = "Negative"
    elif score == 3:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': round(result['score'], 3),
        'raw_score': score
    }

def extract_keywords(text, top_n=5):
    """
    Extract top keywords from text using KeyBERT.
    """
    keywords = keyword_model.extract_keywords(text, 
                                           keyphrase_ngram_range=(1, 2),
                                           stop_words='english',
                                           top_n=top_n)
    return [keyword[0] for keyword in keywords]

def process_batch(df):
    """
    Process a batch of texts from DataFrame.
    Returns DataFrame with results or raises ValueError with descriptive message.
    """
    # Validate DataFrame
    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")
    
    if len(df.columns) < 1:
        raise ValueError("The CSV file must contain at least one column.")
    
    # Check if first column contains text data
    first_col = df.iloc[:, 0]
    if not first_col.dtype == object:
        raise ValueError("The first column must contain text data.")
    
    if first_col.isnull().any():
        raise ValueError("The text column contains empty cells. Please remove or fill them.")
    
    results = []
    total_rows = len(df)
    
    for idx, text in enumerate(first_col):
        try:
            # Ensure text is string
            text = str(text).strip()
            if not text:
                continue
                
            sentiment_result = analyze_sentiment(text)
            keywords = extract_keywords(text)
            sentiment_result['keywords'] = ', '.join(keywords)
            results.append(sentiment_result)
            
        except Exception as e:
            raise ValueError(f"Error processing row {idx + 1}: {str(e)}")
    
    if not results:
        raise ValueError("No valid text entries found in the CSV file.")
        
    return pd.DataFrame(results)

def create_pie_chart(df):
    """
    Create an interactive pie chart using plotly.
    Returns the figure object.
    """
    sentiment_counts = df['sentiment'].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0,  # Set to 0.5 for donut chart
        marker_colors=[SENTIMENT_COLORS[sent] for sent in sentiment_counts.index],
        textinfo='value+percent',
        hovertemplate="<b>%{label}</b><br>" +
                     "Count: %{value}<br>" +
                     "Percentage: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        showlegend=True,
        width=600,
        height=400
    )
    return fig

def create_donut_chart(df):
    """
    Create an interactive donut chart using plotly.
    Returns the figure object.
    """
    fig = create_pie_chart(df)
    fig.update_traces(hole=0.5)
    return fig

def create_line_chart(df):
    """
    Create a line chart showing sentiment trends.
    Assumes data has a timestamp column.
    """
    if 'timestamp' not in df.columns:
        df = df.copy()
        df['timestamp'] = pd.date_range(start='today', periods=len(df), freq='H')
    
    sentiment_pivot = pd.crosstab(df['timestamp'], df['sentiment'])
    
    fig = go.Figure()
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment in sentiment_pivot.columns:
            fig.add_trace(go.Scatter(
                x=sentiment_pivot.index,
                y=sentiment_pivot[sentiment],
                name=sentiment,
                line=dict(color=SENTIMENT_COLORS[sentiment]),
                hovertemplate="<b>%{x}</b><br>" +
                             f"{sentiment}: %{{y}}<extra></extra>"
            ))
    
    fig.update_layout(
        title="Sentiment Trends Over Time",
        xaxis_title="Time",
        yaxis_title="Count",
        hovermode='x unified',
        width=800,
        height=400
    )
    return fig

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

def export_to_pdf(df, visualizations):
    """
    Export analysis results and visualizations to PDF using reportlab.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, height-50, "Sentiment Analysis Report")
    
    # Add summary statistics
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-100, "Summary Statistics")
    c.setFont("Helvetica", 12)
    
    sentiment_counts = df['sentiment'].value_counts()
    y_position = height-130
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        count = sentiment_counts.get(sentiment, 0)
        c.drawString(50, y_position, f"{sentiment}: {count}")
        y_position -= 20
    
    # Add visualizations
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position-20, "Visualizations")
    
    # Add sentiment distribution plot
    if "Sentiment Distribution" in visualizations:
        fig = visualizations["Sentiment Distribution"]
        if isinstance(fig, go.Figure):
            plot_buf = convert_plotly_fig_to_bytes(fig)
            c.drawImage(plot_buf, 50, y_position-270, width=500, height=200)
    
    # Add word cloud
    if "Word Cloud" in visualizations:
        c.drawImage(visualizations["Word Cloud"], 50, y_position-540, width=500, height=200)
    
    c.save()
    buffer.seek(0)
    return buffer

def get_download_link(file_path, link_text):
    """
    Generate a download link for a file.
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{link_text}">Download {link_text}</a>' 