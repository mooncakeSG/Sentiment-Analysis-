import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st

# Dark mode detection function
def detect_dark_mode():
    """
    Detect if user prefers dark mode
    Currently returns False but can be enhanced with browser detection
    """
    # For now, we'll use a simple session state approach
    # In a real app, you might detect this from browser preferences
    return st.session_state.get('dark_mode', False)

# Function to get theme-appropriate colors
def get_theme_colors(dark_mode=False):
    """Get color scheme based on theme preference"""
    if dark_mode:
        return {
            'bg_color': '#1f2937',
            'paper_bg': '#1f2937', 
            'text_color': '#f8fafc',
            'grid_color': '#374151',
            'line_color': '#4b5563'
        }
    else:
        return {
            'bg_color': 'white',
            'paper_bg': 'white',
            'text_color': '#111827',
            'grid_color': '#F3F4F6',
            'line_color': '#E5E7EB'
        }

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
    Create enhanced sentiment distribution visualization with professional styling and dark mode support
    """
    try:
        if data is None or data.empty:
            return None
        
        # Detect dark mode preference
        dark_mode = detect_dark_mode()
        theme_colors = get_theme_colors(dark_mode)
        
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
            
            # Enhanced styling with dark mode support
            fig.update_layout(
                template="plotly_dark" if dark_mode else "plotly",
                plot_bgcolor=theme_colors['bg_color'],
                paper_bgcolor=theme_colors['paper_bg'],
                font_color=theme_colors['text_color'],
                title_font_size=20,
                title_x=0.5,
                title_font_family="Inter, sans-serif",
                margin=dict(t=80, b=60, l=60, r=60),
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    linecolor=theme_colors['line_color'],
                    title_font_size=14,
                    title_font_family="Inter, sans-serif",
                    color=theme_colors['text_color']
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=theme_colors['grid_color'],
                    linecolor=theme_colors['line_color'],
                    title_font_size=14,
                    title_font_family="Inter, sans-serif",
                    color=theme_colors['text_color']
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
                template="plotly_dark" if dark_mode else "plotly",
                plot_bgcolor=theme_colors['bg_color'],
                paper_bgcolor=theme_colors['paper_bg'],
                font_color=theme_colors['text_color'],
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
                template="plotly_dark" if dark_mode else "plotly",
                plot_bgcolor=theme_colors['bg_color'],
                paper_bgcolor=theme_colors['paper_bg'],
                font_color=theme_colors['text_color'],
                title_font_size=20,
                title_x=0.5,
                title_font_family="Inter, sans-serif",
                margin=dict(t=80, b=60, l=60, r=60),
                annotations=[dict(text='Sentiment<br>Analysis', x=0.5, y=0.5, font_size=16, showarrow=False, font_color=theme_colors['text_color'])]
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
                template="plotly_dark" if dark_mode else "plotly",
                plot_bgcolor=theme_colors['bg_color'],
                paper_bgcolor=theme_colors['paper_bg'],
                font_color=theme_colors['text_color'],
                title_font_size=20,
                title_x=0.5,
                title_font_family="Inter, sans-serif",
                margin=dict(t=80, b=60, l=60, r=60),
                xaxis=dict(
                    showgrid=False,
                    linecolor=theme_colors['line_color'],
                    color=theme_colors['text_color']
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=theme_colors['grid_color'],
                    linecolor=theme_colors['line_color'],
                    color=theme_colors['text_color']
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
    Create confidence distribution chart with dark mode support
    """
    try:
        if data is None or data.empty or 'confidence' not in data.columns:
            return None
        
        # Detect dark mode preference
        dark_mode = detect_dark_mode()
        theme_colors = get_theme_colors(dark_mode)
        
        # Create confidence distribution
        fig = px.histogram(
            data, 
            x='confidence',
            title="📊 Confidence Score Distribution",
            labels={'confidence': 'Confidence Score', 'count': 'Number of Texts'},
            nbins=20
        )
        
        fig.update_layout(
            template="plotly_dark" if dark_mode else "plotly",
            plot_bgcolor=theme_colors['bg_color'],
            paper_bgcolor=theme_colors['paper_bg'],
            font_color=theme_colors['text_color'],
            title_font_size=20,
            title_x=0.5,
            title_font_family="Inter, sans-serif",
            margin=dict(t=80, b=60, l=60, r=60),
            xaxis=dict(
                showgrid=False,
                linecolor=theme_colors['line_color'],
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
    """Optimize a plotly figure for PDF export with enhanced styling and forced bright colors"""
    try:
        if fig is None:
            return None
        
        # Create a copy of the figure for modification
        pdf_fig = fig
        
        # Define bright, PDF-friendly colors that will show up clearly
        pdf_colors = {
            'Very Positive': '#00CC00',    # Bright Green
            'Positive': '#66FF66',         # Light Green
            'Neutral': '#FFD700',          # Gold/Yellow
            'Negative': '#FF6600',         # Orange
            'Very Negative': '#FF0000'     # Bright Red
        }
        
        # Enhance for PDF export
        pdf_fig.update_layout(
            # White background for better PDF rendering
            plot_bgcolor="white",
            paper_bgcolor="white",
            
            # Better font settings for PDF
            font=dict(
                family="Arial, sans-serif",
                size=14,
                color="#000000"  # Black text for better PDF contrast
            ),
            
            # Adjust title for PDF
            title=dict(
                font=dict(size=18, color="#000000", family="Arial"),
                x=0.5,
                y=0.95
            ),
            
            # Better margins for PDF
            margin=dict(t=70, b=50, l=50, r=50),
            
            # Force bright colors for PDF
            colorway=['#00CC00', '#66FF66', '#FFD700', '#FF6600', '#FF0000']
        )
        
        # For pie charts, enhance text contrast and force bright colors
        if hasattr(pdf_fig, 'data') and len(pdf_fig.data) > 0:
            trace = pdf_fig.data[0]
            if hasattr(trace, 'type') and trace.type == 'pie':
                # Force specific bright colors for pie chart segments
                colors_list = []
                if hasattr(trace, 'labels'):
                    for label in trace.labels:
                        colors_list.append(pdf_colors.get(str(label), '#808080'))
                
                pdf_fig.update_traces(
                    marker=dict(
                        colors=colors_list,
                        line=dict(color='#FFFFFF', width=3)  # White borders
                    ),
                    textfont=dict(size=14, color="#000000", family="Arial"),  # Black text
                    textposition='inside',
                    textinfo='percent+label',
                    insidetextorientation='radial'
                )
                
                # Update layout specifically for pie charts
                pdf_fig.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.02,
                        font=dict(size=12, color="#000000"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#000000",
                        borderwidth=1
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