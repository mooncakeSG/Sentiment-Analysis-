import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st

# Define consistent color scheme
SENTIMENT_COLORS = {
    'Very Positive': '#00a65a',  # Dark Green
    'Positive': '#2ecc71',      # Light Green
    'Neutral': '#95a5a6',       # Gray
    'Negative': '#e74c3c',      # Light Red
    'Very Negative': '#c0392b'  # Dark Red
}

def create_sentiment_distribution(data, plot_type="bar", **kwargs):
    """
    Create enhanced sentiment distribution visualization with professional styling
    """
    try:
        if data is None or data.empty:
            return None
        
        # Professional color palette
        colors = {
            'Very Positive': '#059669',   # Emerald-600
            'Positive': '#10B981',        # Emerald-500
            'Neutral': '#6B7280',         # Gray-500
            'Negative': '#EF4444',        # Red-500
            'Very Negative': '#DC2626'    # Red-600
        }
        
        # Count sentiments
        sentiment_counts = data['sentiment'].value_counts()
        
        # Create color list for the plot
        plot_colors = [colors.get(sentiment, '#6B7280') for sentiment in sentiment_counts.index]
        
        if plot_type == "bar":
            fig = px.bar(
                x=sentiment_counts.index, 
                y=sentiment_counts.values,
                color=sentiment_counts.index,
                color_discrete_map=colors,
                title="📊 Sentiment Distribution Analysis",
                labels={'x': 'Sentiment Category', 'y': 'Number of Texts'}
            )
            
            # Enhanced styling
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#111827",
                title_font_size=20,
                title_x=0.5,
                title_font_family="Inter, sans-serif",
                margin=dict(t=80, b=60, l=60, r=60),
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    linecolor="#E5E7EB",
                    title_font_size=14,
                    title_font_family="Inter, sans-serif"
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#F3F4F6",
                    linecolor="#E5E7EB",
                    title_font_size=14,
                    title_font_family="Inter, sans-serif"
                )
            )
            
            # Add hover template
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                             "Count: %{y}<br>" +
                             "Percentage: %{customdata:.1%}<extra></extra>",
                customdata=sentiment_counts.values / sentiment_counts.sum()
            )
            
        elif plot_type == "pie":
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map=colors,
                title="🥧 Sentiment Distribution (Pie Chart)"
            )
            
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#111827",
                title_font_size=20,
                title_x=0.5,
                title_font_family="Inter, sans-serif",
                margin=dict(t=80, b=60, l=60, r=60)
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>" +
                             "Count: %{value}<br>" +
                             "Percentage: %{percent}<extra></extra>"
            )
            
        elif plot_type == "donut":
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map=colors,
                title="🍩 Sentiment Distribution (Donut Chart)",
                hole=0.4
            )
            
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#111827",
                title_font_size=20,
                title_x=0.5,
                title_font_family="Inter, sans-serif",
                margin=dict(t=80, b=60, l=60, r=60),
                annotations=[dict(text='Sentiment<br>Analysis', x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
            
        elif plot_type == "line":
            # For line plot, we'll show trend if there's a sequence
            fig = px.line(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="📈 Sentiment Trend Analysis",
                markers=True
            )
            
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#111827",
                title_font_size=20,
                title_x=0.5,
                title_font_family="Inter, sans-serif",
                margin=dict(t=80, b=60, l=60, r=60),
                xaxis=dict(
                    showgrid=False,
                    linecolor="#E5E7EB"
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#F3F4F6",
                    linecolor="#E5E7EB"
                )
            )
            
            fig.update_traces(
                line_color="#4F46E5",
                line_width=3,
                marker_color="#7C3AED",
                marker_size=8
            )
        
        return fig
        
    except Exception as e:
        print(f"Error creating sentiment distribution: {str(e)}")
        return None

def create_confidence_chart(data, **kwargs):
    """
    Create confidence distribution chart
    """
    try:
        if data is None or data.empty or 'confidence' not in data.columns:
            return None
        
        # Create confidence distribution
        fig = px.histogram(
            data, 
            x='confidence',
            title="📊 Confidence Score Distribution",
            labels={'confidence': 'Confidence Score', 'count': 'Number of Texts'},
            nbins=20
        )
        
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#111827",
            title_font_size=20,
            title_x=0.5,
            title_font_family="Inter, sans-serif",
            margin=dict(t=80, b=60, l=60, r=60),
            xaxis=dict(
                showgrid=False,
                linecolor="#E5E7EB",
                title_font_size=14,
                title_font_family="Inter, sans-serif"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#F3F4F6",
                linecolor="#E5E7EB",
                title_font_size=14,
                title_font_family="Inter, sans-serif"
            )
        )
        
        fig.update_traces(marker_color="#4F46E5")
        
        return fig
        
    except Exception as e:
        print(f"Error creating confidence chart: {str(e)}")
        return None

def create_keyword_importance(data, **kwargs):
    """
    Create keyword importance visualization
    """
    try:
        if data is None or data.empty:
            return None
        
        # This is a placeholder function since keyword importance would require
        # more complex processing. For now, return None to indicate not available
        return None
        
    except Exception as e:
        print(f"Error creating keyword importance chart: {str(e)}")
        return None

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

def optimize_chart_for_pdf(fig):
    """Optimize a plotly figure for PDF export with enhanced styling"""
    try:
        if fig is None:
            return None
        
        # Create a copy of the figure for modification
        pdf_fig = fig
        
        # Enhance for PDF export
        pdf_fig.update_layout(
            # White background for better PDF rendering
            plot_bgcolor="white",
            paper_bgcolor="white",
            
            # Better font settings for PDF
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#000000"  # Black text for better PDF contrast
            ),
            
            # Adjust title for PDF
            title=dict(
                font=dict(size=16, color="#000000"),
                x=0.5,
                y=0.95
            ),
            
            # Better margins for PDF
            margin=dict(t=60, b=40, l=40, r=40),
            
            # Remove legend background for cleaner look
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#CCCCCC",
                borderwidth=1
            )
        )
        
        # For pie charts, enhance text contrast
        if 'pie' in str(type(pdf_fig.data[0])).lower():
            pdf_fig.update_traces(
                textfont=dict(size=12, color="white"),
                textposition='inside',
                textinfo='percent+label',
                # Ensure strong colors for PDF
                marker=dict(
                    line=dict(color='white', width=2)
                )
            )
        
        return pdf_fig
        
    except Exception as e:
        print(f"Error optimizing chart for PDF: {str(e)}")
        return fig  # Return original if optimization fails

def convert_plotly_fig_to_bytes(fig):
    """Convert a plotly figure to bytes buffer for PDF export with enhanced quality"""
    try:
        if fig is None:
            return None
        
        # Create high-quality image with specific settings for PDF
        img_bytes = fig.to_image(
            format="png", 
            engine="kaleido",
            width=800,     # Higher resolution
            height=500,    # Better aspect ratio
            scale=2        # Higher DPI for sharper images
        )
        
        if img_bytes is None:
            return None
            
        buf = BytesIO(img_bytes)
        return buf
        
    except Exception as e:
        print(f"Error converting plotly figure to bytes: {str(e)}")
        return None 