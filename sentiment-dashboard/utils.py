import pandas as pd
import tempfile
import pdfkit
from transformers import pipeline
from keybert import KeyBERT
import json
import base64
from io import BytesIO
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from sklearn.metrics import confusion_matrix, classification_report
from optimization import (
    ModelManager,
    timed_cache,
    handle_errors,
    optimize_memory_usage
)
from visualizations import convert_plotly_fig_to_bytes
from reportlab.platypus import Table, TableStyle

# Initialize models using ModelManager
sentiment_analyzer = ModelManager.get_model("sentiment")
keyword_model = ModelManager.get_model("keyword")

# Model limitations and thresholds
MODEL_LIMITATIONS = {
    'max_text_length': 512,  # BERT's maximum token length
    'confidence_threshold': 0.6,  # Minimum confidence for reliable predictions
    'supported_languages': ['English', 'Multilingual'],  # Model supports multiple languages
    'known_limitations': [
        'May struggle with sarcasm and irony',
        'Performance may vary with informal language',
        'Limited context understanding beyond 512 tokens',
        'May be biased towards training data distribution'
    ]
}

@handle_errors
@timed_cache(ttl=3600)
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

@handle_errors
@timed_cache(ttl=3600)
def explain_sentiment(text, sentiment_result):
    """
    Generate explanation for sentiment classification.
    """
    keywords = extract_keywords(text)
    confidence = sentiment_result['confidence']
    
    explanation = {
        'sentiment': sentiment_result['sentiment'],
        'confidence': confidence,
        'key_phrases': keywords,
        'reliability': 'High' if confidence >= MODEL_LIMITATIONS['confidence_threshold'] else 'Low',
        'limitations': []
    }
    
    # Add relevant limitations
    if len(text.split()) > MODEL_LIMITATIONS['max_text_length']:
        explanation['limitations'].append('Text exceeds maximum length, may affect accuracy')
    if confidence < MODEL_LIMITATIONS['confidence_threshold']:
        explanation['limitations'].append('Low confidence score, consider manual review')
    
    return explanation

@handle_errors
@timed_cache(ttl=3600)
def extract_keywords(text, top_n=5):
    """
    Extract top keywords from text using KeyBERT.
    """
    keywords = keyword_model.extract_keywords(
        text, 
                                           keyphrase_ngram_range=(1, 2),
                                           stop_words='english',
        top_n=top_n
    )
    return [keyword[0] for keyword in keywords]

@handle_errors
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
    
    # Add analysis results table
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position-20, "Analysis Results")
    c.setFont("Helvetica", 10)
    y_position -= 40
    
    # Prepare table data (limit rows for readability)
    max_rows = 15
    table_data = [df.columns.tolist()] + df.head(max_rows).values.tolist()
    # Convert all items to string for ReportLab
    table_data = [[str(cell) for cell in row] for row in table_data]
    
    # Create table
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), '#f0f0f0'),  # Light gray header
        ('TEXTCOLOR', (0, 0), (-1, 0), '#222'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), '#fff'),  # White body
        ('GRID', (0, 0), (-1, -1), 0.25, '#bbb'),
    ]))
    # Draw table
    table_width, table_height = table.wrapOn(c, width-100, height)
    table.drawOn(c, 50, y_position-table_height)
    y_position -= (table_height + 20)
    
    # Add visualizations
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position-20, "Visualizations")
    
    # Add sentiment distribution plot
    if "Sentiment Distribution" in visualizations:
        fig = visualizations["Sentiment Distribution"]
        if fig is not None and isinstance(fig, go.Figure):
            plot_buf = convert_plotly_fig_to_bytes(fig)
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp.write(plot_buf.getvalue())
                tmp_path = tmp.name
            c.drawImage(tmp_path, 50, y_position-270, width=500, height=200)
            # Clean up temporary file
            import os
            os.unlink(tmp_path)
    
    # Add word cloud
    if "Word Cloud" in visualizations:
        wordcloud_buf = visualizations["Word Cloud"]
        if wordcloud_buf is not None:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp.write(wordcloud_buf.getvalue())
                tmp_path = tmp.name
            c.drawImage(tmp_path, 50, y_position-540, width=500, height=200)
            # Clean up temporary file
            import os
            os.unlink(tmp_path)
    
    c.save()
    buffer.seek(0)
    return buffer

@handle_errors
def get_download_link(file_path, link_text):
    """
    Generate a download link for a file.
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{link_text}">Download {link_text}</a>' 