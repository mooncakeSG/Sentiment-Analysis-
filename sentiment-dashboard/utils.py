import pandas as pd
import streamlit as st
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
from visualizations import convert_plotly_fig_to_bytes, optimize_chart_for_pdf
from reportlab.platypus import Table, TableStyle

# Initialize models using ModelManager
sentiment_analyzer = ModelManager.get_model("sentiment")
keyword_model = ModelManager.get_model("keyword")

# Model limitations and thresholds
MODEL_LIMITATIONS = {
    'max_text_length': 512,  # BERT's maximum token length
    'confidence_threshold': 0.55,  # Adjusted threshold for more balanced assessment
    'supported_languages': ['English', 'Multilingual'],  # Model supports multiple languages
    'known_limitations': [
        'May struggle with sarcasm and irony',
        'Performance may vary with informal language',
        'Limited context understanding beyond 512 tokens',
        'May be biased towards training data distribution'
    ],
    'high_confidence_threshold': 0.75,  # New threshold for high confidence
    'very_low_confidence_threshold': 0.40,  # New threshold for very low confidence
}

@handle_errors
@timed_cache(ttl=3600)
def analyze_sentiment(text):
    """
    Analyze sentiment of input text using transformers pipeline.
    Returns sentiment label and confidence score.
    
    Use cases:
    - Social media analysis
    - Customer feedback analysis
    - Product reviews classification
    - Brand monitoring
    - Market research
    - Customer service optimization
    - Competitive intelligence
    """
    result = sentiment_analyzer(text)[0]
    # Convert 1-5 score to detailed sentiment labels
    score = int(result['label'].split()[0])
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
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': round(result['score'], 3),
        'raw_score': score,
        'use_case': determine_use_case(text)  # New function to suggest use cases
    }

def determine_use_case(text):
    """
    Determine the most relevant use case for the analyzed text.
    """
    # Enhanced keywords with better coverage and weights
    use_cases = {
        'product_review': {
            'keywords': ['product', 'quality', 'price', 'feature', 'bought', 'purchased', 'taste', 'tastes', 'flavor', 'food', 'meal', 'dish', 'restaurant', 'delivery', 'order', 'item', 'goods', 'material', 'build', 'design', 'works', 'performance', 'value', 'money', 'worth', 'recommend', 'disappointed', 'satisfied', 'amazing', 'terrible', 'excellent', 'poor', 'cheap', 'expensive', 'fish', 'chicken', 'beef', 'pizza', 'burger', 'coffee', 'tea', 'bread', 'cake', 'sweet', 'sour', 'bitter', 'salty', 'spicy', 'fresh', 'stale', 'delicious', 'disgusting', 'yummy', 'nasty', 'cooked', 'raw', 'hot', 'cold', 'warm', 'crispy', 'soft', 'hard', 'juicy', 'dry'],
            'weight': 1.2  # Higher weight for product reviews
        },
        'social_media': {
            'keywords': ['post', 'posted', 'tweet', 'comment', 'like', 'share', 'follow', 'hashtag', 'instagram', 'facebook', 'twitter', 'snapchat', 'tiktok', 'social', 'viral', 'trending', 'story', 'stories'],
            'weight': 1.0
        },
        'customer_feedback': {
            'keywords': ['review', 'feedback', 'rating', 'experience', 'service', 'staff', 'team', 'visit', 'visited', 'customer', 'client', 'overall', 'satisfaction', 'recommend', 'suggestion'],
            'weight': 1.1
        },
        'brand_monitoring': {
            'keywords': ['brand', 'company', 'reputation', 'market', 'competitor', 'business', 'organization', 'corporate', 'enterprise', 'firm'],
            'weight': 1.0
        },
        'market_research': {
            'keywords': ['market', 'trend', 'industry', 'consumer', 'demand', 'analysis', 'research', 'study', 'survey', 'data', 'statistics'],
            'weight': 1.0
        },
        'customer_service': {
            'keywords': ['support', 'help', 'issue', 'problem', 'resolution', 'solve', 'fix', 'assistance', 'ticket', 'contact', 'complaint', 'refund', 'return'],
            'weight': 1.1
        },
        'competitive_intel': {
            'keywords': ['competitor', 'versus', 'vs', 'compared', 'alternative', 'better', 'worse', 'competition', 'rival', 'against'],
            'weight': 1.0
        }
    }
    
    text_lower = text.lower()
    use_case_scores = {}
    
    for case, case_data in use_cases.items():
        keywords = case_data['keywords']
        weight = case_data['weight']
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Apply weight and calculate score
        score = keyword_matches * weight
        use_case_scores[case] = score
    
    # If no keywords match, return default
    if all(score == 0 for score in use_case_scores.values()):
        return 'General Analysis'
    
    # Get the use case with highest weighted score
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

@handle_errors
@timed_cache(ttl=3600)
def explain_sentiment(text, sentiment_result):
    """
    Generate enhanced explanation for sentiment classification with improved confidence assessment.
    """
    keywords = extract_keywords(text)
    confidence = sentiment_result['confidence']
    text_length = len(text.split())
    
    # Enhanced confidence assessment
    reliability = 'High'
    if confidence >= MODEL_LIMITATIONS['high_confidence_threshold']:
        reliability = 'Very High'
    elif confidence >= MODEL_LIMITATIONS['confidence_threshold']:
        reliability = 'Good'
    elif confidence >= MODEL_LIMITATIONS['very_low_confidence_threshold']:
        reliability = 'Moderate'
    else:
        reliability = 'Low'
    
    explanation = {
        'sentiment': sentiment_result['sentiment'],
        'confidence': confidence,
        'key_phrases': keywords,
        'reliability': reliability,
        'limitations': []
    }
    
    # Intelligent limitation detection
    limitations = []
    
    # Text length analysis
    if text_length > MODEL_LIMITATIONS['max_text_length']:
        limitations.append('Text exceeds maximum length, analysis may be truncated')
    elif text_length < 5:
        limitations.append('Very short text, consider adding more context for better accuracy')
    
    # Confidence-based limitations with more nuanced messages
    if confidence < MODEL_LIMITATIONS['very_low_confidence_threshold']:
        limitations.append('Very low confidence - result may be unreliable, manual review strongly recommended')
    elif confidence < MODEL_LIMITATIONS['confidence_threshold']:
        # Be more specific about why confidence might be moderate
        if text_length < 10:
            limitations.append('Moderate confidence due to short text length')
        elif any(char in text for char in ['!', '?', '...', '!!!']):
            limitations.append('Moderate confidence - text contains complex punctuation or emphasis')
        else:
            limitations.append('Moderate confidence - consider additional context if available')
    
    # Content-based analysis for potential challenges
    text_lower = text.lower()
    
    # Detect potential sarcasm indicators
    sarcasm_indicators = ['oh great', 'wonderful', 'fantastic', 'perfect', 'just what i needed', 'oh sure']
    if any(indicator in text_lower for indicator in sarcasm_indicators) and confidence < 0.8:
        limitations.append('Possible sarcasm detected - sentiment may be opposite of literal meaning')
    
    # Detect mixed sentiment indicators
    mixed_indicators = ['but', 'however', 'although', 'though', 'except', 'despite']
    positive_words = ['good', 'great', 'love', 'like', 'amazing', 'excellent']
    negative_words = ['bad', 'hate', 'terrible', 'awful', 'horrible', 'poor']
    
    has_mixed = any(indicator in text_lower for indicator in mixed_indicators)
    has_positive = any(word in text_lower for word in positive_words)
    has_negative = any(word in text_lower for word in negative_words)
    
    if has_mixed and has_positive and has_negative:
        limitations.append('Mixed sentiment detected - review may contain both positive and negative aspects')
    
    # Detect informal language
    informal_indicators = ['lol', 'omg', 'wtf', 'tbh', 'imo', 'ngl', 'fr']
    if any(indicator in text_lower for indicator in informal_indicators):
        limitations.append('Informal language detected - sentiment interpretation may vary')
    
    # Detect questions (which are often neutral but may be misclassified)
    if text.count('?') > 0 and sentiment_result['sentiment'] != 'Neutral':
        limitations.append('Text contains questions - sentiment may be less definitive')
    
    # Only add limitations that are relevant
    explanation['limitations'] = limitations[:3]  # Limit to top 3 most relevant limitations
    
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
    Export analysis results and visualizations to PDF using reportlab with professional styling.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Define colors and styles
    header_color = colors.HexColor('#4F46E5')  # Changed to match app theme
    accent_color = colors.HexColor('#10B981')  # Green accent
    text_color = colors.HexColor('#1F2937')    # Dark gray for text
    light_gray = colors.HexColor('#F9FAFB')    # Very light gray
    white_bg = colors.white                    # Pure white background
    
    # Page 1: Title and Executive Summary
    def draw_header():
        # Header background
        c.setFillColor(header_color)
        c.rect(0, height-80, width, 80, fill=1, stroke=0)
        
        # Title
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(width/2, height-35, "Sentiment Analysis Report")
        
        # Subtitle
        c.setFont("Helvetica", 12)
        c.drawCentredString(width/2, height-55, "Professional Analysis Results")
        
        # Date and time
        from datetime import datetime
        current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        c.setFont("Helvetica", 10)
        c.drawCentredString(width/2, height-70, f"Generated on {current_time}")
    
    draw_header()
    
    # Executive Summary Section
    y_position = height - 120
    c.setFillColor(text_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, "Executive Summary")
    
    # Summary statistics - handle both uppercase and lowercase column names
    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'Sentiment'
    confidence_col = 'confidence' if 'confidence' in df.columns else 'Confidence'
    use_case_col = 'use_case' if 'use_case' in df.columns else 'Use_Case'
    
    # Handle different text column names
    text_col = None
    if 'text' in df.columns:
        text_col = 'text'
    elif 'Content_Preview' in df.columns:
        text_col = 'Content_Preview'
    elif 'Label' in df.columns:
        text_col = 'Label'
    else:
        # Fallback to first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                text_col = col
                break
    
    # Handle different keyword column names
    keywords_col = None
    if 'keywords' in df.columns:
        keywords_col = 'keywords'
    elif 'Key_Phrases' in df.columns:
        keywords_col = 'Key_Phrases'
    
    sentiment_counts = df[sentiment_col].value_counts()
    total_texts = len(df)
    
    # Calculate percentages
    sentiment_percentages = {
        sentiment: (count / total_texts) * 100 
        for sentiment, count in sentiment_counts.items()
    }
    
    # Determine dominant sentiment
    dominant_sentiment = sentiment_counts.index[0]
    dominant_percentage = sentiment_percentages[dominant_sentiment]
    
    # Executive summary text
    y_position -= 30
    c.setFont("Helvetica", 12)
    summary_text = [
        f"Analysis of {total_texts} text entries reveals the following sentiment distribution:",
        f"• {dominant_sentiment} sentiment dominates at {dominant_percentage:.1f}% of responses",
        f"• Average confidence score: {df[confidence_col].mean():.3f}",
        f"• Most common use case: {df[use_case_col].mode().iloc[0] if use_case_col in df.columns else 'General Analysis'}"
    ]
    
    for line in summary_text:
        c.drawString(50, y_position, line)
        y_position -= 20
    
    # Detailed Statistics Section
    y_position -= 30
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, "Detailed Statistics")
    
    # Create statistics table
    y_position -= 30
    
    # Sentiment distribution table
    sentiment_order = ['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative']
    
    # Table headers - improved readability
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(light_gray)
    c.rect(50, y_position-20, 450, 20, fill=1, stroke=1)
    c.setStrokeColor(colors.HexColor('#E5E7EB'))
    c.setFillColor(text_color)
    c.drawString(60, y_position-15, "Sentiment Class")
    c.drawString(200, y_position-15, "Count")
    c.drawString(280, y_position-15, "Percentage")
    c.drawString(380, y_position-15, "Confidence Range")
    
    y_position -= 20
    
    # Table rows
    c.setFont("Helvetica", 11)
    for sentiment in sentiment_order:
        count = sentiment_counts.get(sentiment, 0)
        percentage = sentiment_percentages.get(sentiment, 0)
        
        # Get confidence range for this sentiment
        sentiment_data = df[df[sentiment_col] == sentiment][confidence_col] if count > 0 else []
        if len(sentiment_data) > 0:
            conf_min = sentiment_data.min()
            conf_max = sentiment_data.max()
            conf_range = f"{conf_min:.3f} - {conf_max:.3f}"
        else:
            conf_range = "N/A"
        
        # Alternate row colors
        if sentiment_order.index(sentiment) % 2 == 0:
            c.setFillColor(colors.HexColor('#FAFAFA'))  # Very light gray instead of dark
            c.rect(50, y_position-15, 450, 15, fill=1, stroke=1)
        else:
            c.setFillColor(white_bg)  # Pure white background
            c.rect(50, y_position-15, 450, 15, fill=1, stroke=1)
        
        c.setFillColor(text_color)  # Ensure text is dark for contrast
        c.drawString(60, y_position-10, sentiment)
        c.drawString(200, y_position-10, str(count))
        c.drawString(280, y_position-10, f"{percentage:.1f}%")
        c.drawString(380, y_position-10, conf_range)
        y_position -= 15
    
    # Use Case Analysis (if available)
    if use_case_col in df.columns:
        y_position -= 30
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Use Case Distribution")
        
        use_case_counts = df[use_case_col].value_counts()
        y_position -= 20
        c.setFont("Helvetica", 11)
        
        for use_case, count in use_case_counts.head(5).items():
            percentage = (count / total_texts) * 100
            c.drawString(60, y_position, f"• {use_case}: {count} ({percentage:.1f}%)")
            y_position -= 15
    
    # Key Insights Section
    y_position -= 30
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "Key Insights")
    
    y_position -= 20
    c.setFont("Helvetica", 11)
    
    # Generate insights based on data
    insights = []
    
    # Sentiment insights
    positive_sentiment = sentiment_counts.get('Very Positive', 0) + sentiment_counts.get('Positive', 0)
    negative_sentiment = sentiment_counts.get('Very Negative', 0) + sentiment_counts.get('Negative', 0)
    neutral_sentiment = sentiment_counts.get('Neutral', 0)
    
    if positive_sentiment > negative_sentiment:
        insights.append(f"• Overall positive sentiment detected ({((positive_sentiment/total_texts)*100):.1f}% positive vs {((negative_sentiment/total_texts)*100):.1f}% negative)")
    elif negative_sentiment > positive_sentiment:
        insights.append(f"• Negative sentiment concerns identified ({((negative_sentiment/total_texts)*100):.1f}% negative vs {((positive_sentiment/total_texts)*100):.1f}% positive)")
    else:
        insights.append("• Balanced sentiment distribution observed")
    
    # Confidence insights
    avg_confidence = df[confidence_col].mean()
    if avg_confidence > 0.8:
        insights.append("• High confidence in predictions indicates reliable results")
    elif avg_confidence < 0.6:
        insights.append("• Lower confidence scores suggest need for manual review")
    
    # Variability insights
    confidence_std = df[confidence_col].std()
    if confidence_std > 0.2:
        insights.append("• High variability in confidence scores detected")
    
    for insight in insights:
        c.drawString(50, y_position, insight)
        y_position -= 15
    
    # Start new page for detailed results
    c.showPage()
    
    # Page 2: Detailed Analysis Results
    draw_header()
    
    y_position = height - 120
    c.setFillColor(text_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, "Detailed Analysis Results")
    
    # Prepare table data with better formatting
    y_position -= 30
    max_rows_per_page = 15  # Reduce rows to fit better
    
    # Column headers - make them even shorter
    headers = ['Text (30 chars)', 'Sentiment', 'Conf.', 'Keywords', 'Use Case']
    
    # Handle different text column names for different analysis types
    text_col = None
    if 'text' in df.columns:
        text_col = 'text'
    elif 'Content_Preview' in df.columns:
        text_col = 'Content_Preview'
    elif 'Label' in df.columns:
        text_col = 'Label'
    else:
        # Fallback to first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                text_col = col
                break
    
    # Handle different keyword column names
    keywords_col = None
    if 'keywords' in df.columns:
        keywords_col = 'keywords'
    elif 'Key_Phrases' in df.columns:
        keywords_col = 'Key_Phrases'
    
    # Create table data
    table_data = [headers]
    
    for idx, row in df.head(max_rows_per_page).iterrows():
        # Truncate text very aggressively for display - handle missing text column
        if text_col and text_col in row:
            text_display = str(row[text_col])[:30] + '...' if len(str(row[text_col])) > 30 else str(row[text_col])
            # Clean text to remove line breaks and special characters
            text_display = text_display.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        else:
            text_display = f"Row {idx+1}"  # Fallback if no text column
        
        # Get only the first keyword to fit better - handle missing keywords column
        if keywords_col and keywords_col in row and row[keywords_col]:
            keywords = str(row[keywords_col]).split(', ')[:1]
            keywords_display = keywords[0] if keywords else ''
            # Truncate keywords if too long
            if len(keywords_display) > 15:
                keywords_display = keywords_display[:12] + '...'
        else:
            keywords_display = ''  # No keywords available
        
        # Truncate use case even more aggressively
        use_case = row.get(use_case_col, 'General')
        if len(str(use_case)) > 12:
            use_case = str(use_case)[:9] + '...'
        
        # Truncate sentiment if needed
        sentiment = row[sentiment_col]
        if len(sentiment) > 10:
            sentiment_map = {
                'Very Positive': 'V.Pos',
                'Very Negative': 'V.Neg',
                'Positive': 'Pos',
                'Negative': 'Neg',
                'Neutral': 'Neutral'
            }
            sentiment = sentiment_map.get(sentiment, sentiment[:8])
        
        row_data = [
            text_display,
            sentiment,
            f"{row[confidence_col]:.2f}",  # Shorter confidence format
            keywords_display,
            str(use_case)
        ]
        
        table_data.append(row_data)
    
    # Create and style the table with much smaller column widths
    from reportlab.platypus import Table, TableStyle
    
    # Further reduce column widths to prevent overlap - total around 480
    col_widths = [150, 60, 40, 90, 70]  # Much more conservative widths
    
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        # Header styling - light header with dark text for better readability
        ('BACKGROUND', (0, 0), (-1, 0), light_gray),
        ('TEXTCOLOR', (0, 0), (-1, 0), text_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),  # Slightly larger for readability
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        
        # Data rows styling - ensure white background for readability
        ('BACKGROUND', (0, 1), (-1, -1), white_bg),
        ('TEXTCOLOR', (0, 1), (-1, -1), text_color),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),  # Larger font for better readability
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white_bg, colors.HexColor('#FAFAFA')]),  # Very light alternating
        
        # Grid styling - lighter grid lines
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),  # More padding for readability
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        
        # Text wrapping and overflow handling
        ('WORDWRAP', (0, 0), (-1, -1), 'LTR'),
        ('OVERFLOW', (0, 0), (-1, -1), 'TRUNCATE'),
    ]))
    
    # Draw table
    table_width, table_height = table.wrapOn(c, width-100, height)
    table.drawOn(c, 50, y_position-table_height)
    
    # Add note about truncation if there are more rows
    if len(df) > max_rows_per_page:
        y_position = y_position - table_height - 20
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColor(colors.grey)
        c.drawString(50, y_position, f"Note: Showing first {max_rows_per_page} results. Total records: {len(df)}")
        c.drawString(50, y_position-10, "Text, keywords, and use cases are heavily truncated for display.")
    
    # Start new page for visualizations
    c.showPage()
    
    # Page 3: Visualizations
    draw_header()
    
    y_position = height - 120
    c.setFillColor(text_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, "Data Visualizations")
    
    # Add sentiment distribution chart - handle any chart type dynamically
    chart_added = False
    for viz_name, fig in visualizations.items():
        if "Sentiment Distribution" in viz_name and fig is not None:
            y_position -= 40
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, viz_name)  # Use the dynamic name from the visualizations dict
            
            try:
                # Optimize chart for PDF export
                optimized_fig = optimize_chart_for_pdf(fig)
                plot_buf = convert_plotly_fig_to_bytes(optimized_fig)
                if plot_buf is not None:  # Check if conversion was successful
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp.write(plot_buf.getvalue())
                        tmp_path = tmp.name
                    
                    # Draw chart with border
                    chart_width, chart_height = 500, 250
                    c.setStrokeColor(colors.HexColor('#E5E7EB'))  # Use lighter border color
                    c.setLineWidth(1)
                    c.rect(50, y_position-chart_height-10, chart_width, chart_height, fill=0, stroke=1)
                    c.drawImage(tmp_path, 55, y_position-chart_height-5, width=chart_width-10, height=chart_height-10)
                    
                    import os
                    os.unlink(tmp_path)
                    y_position -= (chart_height + 40)
                    chart_added = True
                else:
                    # Chart conversion failed, show error message
                    c.setFont("Helvetica", 10)
                    c.setFillColor(colors.grey)
                    c.drawString(50, y_position-20, "Chart visualization temporarily unavailable")
                    y_position -= 40
            except Exception as e:
                # Handle any chart conversion errors
                c.setFont("Helvetica", 10)
                c.setFillColor(colors.grey)
                c.drawString(50, y_position-20, f"Chart error: {str(e)[:50]}...")
                y_position -= 40
            break  # Only add one chart type
    
    # Fallback if no sentiment distribution chart was found
    if not chart_added and "Sentiment Distribution" in visualizations:
        fig = visualizations["Sentiment Distribution"]
        if fig is not None:
            y_position -= 40
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "Sentiment Distribution Chart")
            
            try:
                # Optimize chart for PDF export
                optimized_fig = optimize_chart_for_pdf(fig)
                plot_buf = convert_plotly_fig_to_bytes(optimized_fig)
                if plot_buf is not None:  # Check if conversion was successful
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp.write(plot_buf.getvalue())
                        tmp_path = tmp.name
                    
                    # Draw chart with border
                    chart_width, chart_height = 500, 250
                    c.setStrokeColor(colors.HexColor('#E5E7EB'))
                    c.setLineWidth(1)
                    c.rect(50, y_position-chart_height-10, chart_width, chart_height, fill=0, stroke=1)
                    c.drawImage(tmp_path, 55, y_position-chart_height-5, width=chart_width-10, height=chart_height-10)
                    
                    import os
                    os.unlink(tmp_path)
                    y_position -= (chart_height + 40)
                else:
                    # Chart conversion failed, show error message
                    c.setFont("Helvetica", 10)
                    c.setFillColor(colors.grey)
                    c.drawString(50, y_position-20, "Chart visualization temporarily unavailable")
                    y_position -= 40
            except Exception as e:
                # Handle any chart conversion errors
                c.setFont("Helvetica", 10)
                c.setFillColor(colors.grey)
                c.drawString(50, y_position-20, f"Chart error: {str(e)[:50]}...")
                y_position -= 40
    
    # Add word cloud if available
    if "Word Cloud" in visualizations and y_position > 200:
        wordcloud_buf = visualizations["Word Cloud"]
        if wordcloud_buf is not None:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "Word Cloud Analysis")
            
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp.write(wordcloud_buf.getvalue())
                    tmp_path = tmp.name
                
                # Draw word cloud with border
                cloud_width, cloud_height = 500, 200
                c.setStrokeColor(colors.grey)
                c.setLineWidth(1)
                c.rect(50, y_position-cloud_height-30, cloud_width, cloud_height, fill=0, stroke=1)
                c.drawImage(tmp_path, 55, y_position-cloud_height-25, width=cloud_width-10, height=cloud_height-10)
                
                import os
                os.unlink(tmp_path)
            except Exception as e:
                # Handle any word cloud errors
                c.setFont("Helvetica", 10)
                c.setFillColor(colors.grey)
                c.drawString(50, y_position-20, f"Word cloud error: {str(e)[:50]}...")
    
    # Footer on last page
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.grey)
    footer_text = "Generated by Sentiment Analysis Dashboard - Tech Titanians"
    text_width = c.stringWidth(footer_text)
    c.drawString((width - text_width) / 2, 30, footer_text)
    
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

def handle_followup_question(question, text, sentiment_result, keywords):
    """
    Handle follow-up questions about sentiment analysis results.
    Provides explanations and insights based on the analysis.
    """
    question_lower = question.lower()
    sentiment = sentiment_result['sentiment']
    confidence = sentiment_result['confidence']
    
    # Common question patterns and responses
    if any(phrase in question_lower for phrase in ['why', 'labeled', 'classified', 'determined']):
        if 'negative' in question_lower and 'negative' in sentiment.lower():
            return generate_negative_explanation(text, sentiment_result, keywords)
        elif 'positive' in question_lower and 'positive' in sentiment.lower():
            return generate_positive_explanation(text, sentiment_result, keywords)
        elif 'neutral' in question_lower and 'neutral' in sentiment.lower():
            return generate_neutral_explanation(text, sentiment_result, keywords)
        else:
            return generate_general_explanation(text, sentiment_result, keywords)
    
    elif any(phrase in question_lower for phrase in ['keywords', 'words', 'caused', 'influenced']):
        return generate_keyword_explanation(text, sentiment_result, keywords)
    
    elif any(phrase in question_lower for phrase in ['confidence', 'sure', 'certain', 'reliable']):
        return generate_confidence_explanation(text, sentiment_result, keywords)
    
    elif any(phrase in question_lower for phrase in ['improve', 'better', 'accuracy']):
        return generate_improvement_suggestions(text, sentiment_result, keywords)
    
    elif any(phrase in question_lower for phrase in ['meaning', 'definition', 'what is']):
        return generate_definition_explanation(sentiment_result)
    
    else:
        return generate_general_explanation(text, sentiment_result, keywords)

def generate_negative_explanation(text, sentiment_result, keywords):
    """Generate explanation for negative sentiment classification."""
    sentiment = sentiment_result['sentiment']
    confidence = sentiment_result['confidence']
    
    explanation = f"**Why this text was labeled as {sentiment}:**\n\n"
    
    # Analyze negative indicators
    negative_words = [
        'terrible', 'awful', 'horrible', 'bad', 'poor', 'disappointing', 'frustrated',
        'angry', 'upset', 'hate', 'dislike', 'problem', 'issue', 'wrong', 'failed',
        'slow', 'expensive', 'waste', 'regret', 'disappointed', 'unsatisfied'
    ]
    
    found_negative_words = [word for word in negative_words if word in text.lower()]
    
    if found_negative_words:
        explanation += f"• **Negative language detected**: Words like '{', '.join(found_negative_words[:3])}' strongly indicate negative sentiment.\n"
    
    if keywords:
        explanation += f"• **Key phrases identified**: {', '.join(keywords[:3])} - these terms are associated with negative experiences.\n"
    
    explanation += f"• **Confidence level**: {confidence:.1%} - The model is {'highly confident' if confidence > 0.8 else 'moderately confident' if confidence > 0.6 else 'less confident'} in this classification.\n"
    
    if confidence < 0.7:
        explanation += "• **Note**: Lower confidence may indicate mixed sentiment or ambiguous language.\n"
    
    return explanation

def generate_positive_explanation(text, sentiment_result, keywords):
    """Generate explanation for positive sentiment classification."""
    sentiment = sentiment_result['sentiment']
    confidence = sentiment_result['confidence']
    
    explanation = f"**Why this text was labeled as {sentiment}:**\n\n"
    
    # Analyze positive indicators
    positive_words = [
        'excellent', 'amazing', 'fantastic', 'great', 'good', 'love', 'perfect',
        'wonderful', 'outstanding', 'brilliant', 'impressive', 'satisfied', 'happy',
        'pleased', 'recommend', 'best', 'awesome', 'incredible', 'superb'
    ]
    
    found_positive_words = [word for word in positive_words if word in text.lower()]
    
    if found_positive_words:
        explanation += f"• **Positive language detected**: Words like '{', '.join(found_positive_words[:3])}' strongly indicate positive sentiment.\n"
    
    if keywords:
        explanation += f"• **Key phrases identified**: {', '.join(keywords[:3])} - these terms are associated with positive experiences.\n"
    
    explanation += f"• **Confidence level**: {confidence:.1%} - The model is {'highly confident' if confidence > 0.8 else 'moderately confident' if confidence > 0.6 else 'less confident'} in this classification.\n"
    
    return explanation

def generate_neutral_explanation(text, sentiment_result, keywords):
    """Generate explanation for neutral sentiment classification."""
    confidence = sentiment_result['confidence']
    
    explanation = f"**Why this text was labeled as Neutral:**\n\n"
    explanation += "• **Balanced language**: The text contains neither strongly positive nor negative indicators.\n"
    explanation += "• **Factual content**: The text appears to be informational or descriptive rather than emotional.\n"
    
    if keywords:
        explanation += f"• **Key phrases identified**: {', '.join(keywords[:3])} - these terms are generally neutral in sentiment.\n"
    
    explanation += f"• **Confidence level**: {confidence:.1%}\n"
    
    if confidence < 0.6:
        explanation += "• **Note**: Low confidence may indicate the text has subtle emotional undertones that are difficult to classify.\n"
    
    return explanation

def generate_keyword_explanation(text, sentiment_result, keywords):
    """Generate explanation about keywords that influenced the result."""
    sentiment = sentiment_result['sentiment']
    
    explanation = f"**Keywords that influenced the {sentiment} classification:**\n\n"
    
    if keywords:
        for i, keyword in enumerate(keywords[:5], 1):
            explanation += f"{i}. **{keyword}**: "
            
            # Analyze each keyword's sentiment contribution
            if any(neg_word in keyword.lower() for neg_word in ['bad', 'poor', 'terrible', 'awful', 'problem', 'issue']):
                explanation += "Associated with negative experiences\n"
            elif any(pos_word in keyword.lower() for pos_word in ['good', 'great', 'excellent', 'amazing', 'perfect', 'love']):
                explanation += "Associated with positive experiences\n"
            else:
                explanation += "Context-dependent term that influenced the classification\n"
        
        explanation += f"\n• **How it works**: The AI model analyzes these keywords along with their context, sentence structure, and surrounding words to determine overall sentiment.\n"
    else:
        explanation += "• No specific keywords were extracted, but the model analyzed the overall language patterns and context.\n"
    
    return explanation

def generate_confidence_explanation(text, sentiment_result, keywords):
    """Generate explanation about confidence scores."""
    confidence = sentiment_result['confidence']
    sentiment = sentiment_result['sentiment']
    
    explanation = f"**Confidence Score Explanation ({confidence:.1%}):**\n\n"
    
    if confidence > 0.8:
        explanation += "• **High Confidence**: The model is very certain about this classification.\n"
        explanation += "• **Strong indicators**: Clear sentiment markers were detected in the text.\n"
        explanation += "• **Reliability**: This result is highly reliable for decision-making.\n"
    elif confidence > 0.6:
        explanation += "• **Moderate Confidence**: The model is reasonably certain about this classification.\n"
        explanation += "• **Mixed signals**: Some ambiguous language may be present.\n"
        explanation += "• **Reliability**: This result is generally reliable but consider context.\n"
    else:
        explanation += "• **Lower Confidence**: The model is less certain about this classification.\n"
        explanation += "• **Ambiguous language**: The text may contain mixed sentiment or be difficult to interpret.\n"
        explanation += "• **Recommendation**: Consider manual review or additional context.\n"
    
    explanation += f"\n**Factors affecting confidence:**\n"
    explanation += f"• Text length: {'Adequate' if len(text.split()) > 5 else 'Short (may affect accuracy)'}\n"
    explanation += f"• Language clarity: {'Clear' if confidence > 0.7 else 'Potentially ambiguous'}\n"
    explanation += f"• Sentiment strength: {'Strong {sentiment.lower()} indicators' if confidence > 0.8 else 'Moderate indicators'}\n"
    
    return explanation

def generate_improvement_suggestions(text, sentiment_result, keywords):
    """Generate suggestions for improving analysis accuracy."""
    confidence = sentiment_result['confidence']
    
    explanation = "**Suggestions for Better Analysis:**\n\n"
    
    if len(text.split()) < 5:
        explanation += "• **Add more context**: Longer texts generally provide more accurate results.\n"
    
    if confidence < 0.7:
        explanation += "• **Provide additional context**: More background information could improve accuracy.\n"
        explanation += "• **Consider manual review**: Low confidence results should be verified.\n"
    
    explanation += "• **Use clear language**: Explicit sentiment words help the model make better predictions.\n"
    explanation += "• **Avoid mixed messages**: Separate positive and negative points for clearer analysis.\n"
    explanation += "• **Include specific details**: Concrete examples help improve classification accuracy.\n"
    
    return explanation

def generate_definition_explanation(sentiment_result):
    """Generate explanation of sentiment definitions."""
    sentiment = sentiment_result['sentiment']
    
    definitions = {
        'Very Positive': 'Extremely positive sentiment - indicates strong satisfaction, enthusiasm, or praise',
        'Positive': 'Generally positive sentiment - indicates satisfaction, approval, or favorable opinion',
        'Neutral': 'Neither positive nor negative - factual, informational, or balanced content',
        'Negative': 'Generally negative sentiment - indicates dissatisfaction, criticism, or unfavorable opinion',
        'Very Negative': 'Extremely negative sentiment - indicates strong dissatisfaction, anger, or harsh criticism'
    }
    
    explanation = f"**Sentiment Definition:**\n\n"
    explanation += f"• **{sentiment}**: {definitions.get(sentiment, 'Sentiment classification based on emotional tone')}\n\n"
    
    explanation += "**5-Class Sentiment Scale:**\n"
    for sent, definition in definitions.items():
        marker = "👈 **Current**" if sent == sentiment else ""
        explanation += f"• **{sent}**: {definition} {marker}\n"
    
    return explanation

def generate_general_explanation(text, sentiment_result, keywords):
    """Generate a general explanation of the analysis."""
    sentiment = sentiment_result['sentiment']
    confidence = sentiment_result['confidence']
    
    explanation = f"**Analysis Summary:**\n\n"
    explanation += f"• **Sentiment**: {sentiment}\n"
    explanation += f"• **Confidence**: {confidence:.1%}\n"
    explanation += f"• **Key Terms**: {', '.join(keywords[:3]) if keywords else 'None extracted'}\n\n"
    
    explanation += "**How the analysis works:**\n"
    explanation += "• The AI model analyzes word choice, context, and language patterns\n"
    explanation += "• It considers the overall emotional tone and meaning\n"
    explanation += "• Keywords and phrases are weighted based on their sentiment associations\n"
    explanation += f"• The final classification is based on the strongest sentiment indicators\n"
    
    return explanation

def validate_text_input(text):
    """
    Validate text input for sentiment analysis.
    Returns None if valid, error message string if invalid.
    """
    if not text:
        return "❌ Text is empty. Please enter some text to analyze."
    
    text_stripped = text.strip()
    if not text_stripped:
        return "❌ Text contains only whitespace. Please enter meaningful text."
    
    if len(text_stripped) < 3:
        return "⚠️ Text is too short for meaningful analysis. Please enter at least 3 characters."
    
    if len(text_stripped) > 5000:
        return "⚠️ Text is too long (over 5000 characters). Please shorten it for better accuracy."
    
    # Check for suspicious content
    if text_stripped.count('\n') > 50:
        return "⚠️ Text has too many line breaks. Please use cleaner formatting."
    
    # Check if text is mostly numbers or special characters
    alphanumeric_chars = sum(c.isalnum() or c.isspace() for c in text_stripped)
    if alphanumeric_chars / len(text_stripped) < 0.5:
        return "⚠️ Text contains mostly special characters or numbers. Please use natural language text."
    
    return None

def validate_file_content(df, file_type):
    """
    Validate uploaded file content.
    Returns tuple (is_valid, error_message, warning_message)
    """
    try:
        # Basic empty check
        if df is None:
            return False, "❌ Failed to process file. File may be corrupted.", None
        
        if df.empty:
            return False, "❌ Uploaded file is empty. Please upload a file with content.", None
        
        if file_type == "csv":
            # CSV-specific validation
            if df.shape[1] < 1:
                return False, "❌ CSV file must contain at least one column.", None
            
            # Check if first column exists and has data
            first_col = df.iloc[:, 0]
            if first_col.isnull().all():
                return False, "❌ First column is completely empty. Please ensure your text data is in the first column.", None
            
            # Check for missing values
            null_count = first_col.isnull().sum()
            if null_count > 0:
                warning_msg = f"⚠️ Found {null_count} empty cells in text column. These rows will be skipped."
                # Remove null values
                df.dropna(subset=[df.columns[0]], inplace=True)
                if df.empty:
                    return False, "❌ No valid text entries found after removing empty cells.", None
                return True, None, warning_msg
            
            # Validate text content in first column
            invalid_rows = []
            for idx, text in enumerate(first_col):
                if isinstance(text, str):
                    validation_error = validate_text_input(text)
                    if validation_error:
                        invalid_rows.append(f"Row {idx + 2}: {validation_error}")  # +2 for header and 1-based indexing
                else:
                    invalid_rows.append(f"Row {idx + 2}: Contains non-text data")
            
            if invalid_rows:
                if len(invalid_rows) > 5:
                    warning_msg = f"⚠️ Found {len(invalid_rows)} rows with issues. First 5: " + "; ".join(invalid_rows[:5])
                else:
                    warning_msg = "⚠️ Issues found: " + "; ".join(invalid_rows)
                return True, None, warning_msg
        
        elif file_type == "txt":
            # TXT-specific validation
            if len(df) == 0:
                return False, "❌ TXT file contains no valid lines. Please check your file format.", None
            
            # Check for very long lines that might indicate wrong format
            long_lines = sum(1 for text in df.iloc[:, 0] if len(str(text)) > 1000)
            if long_lines > len(df) * 0.5:
                warning_msg = f"⚠️ Found {long_lines} very long lines. Consider splitting them for better analysis."
                return True, None, warning_msg
        
        # Check total size
        if len(df) > 10000:
            return False, "❌ File contains too many entries (>10,000). Please split into smaller files.", None
        elif len(df) > 1000:
            warning_msg = f"⚠️ Large file detected ({len(df)} entries). Processing may take longer."
            return True, None, warning_msg
        
        return True, None, None
        
    except Exception as e:
        return False, f"❌ Error validating file content: {str(e)}", None

def safe_sentiment_analysis(text):
    """
    Safely perform sentiment analysis with comprehensive error handling.
    """
    try:
        # Validate input first
        validation_error = validate_text_input(text)
        if validation_error:
            return {
                'error': validation_error,
                'text': text,
                'sentiment': None,
                'confidence': 0.0,
                'raw_score': 0,
                'use_case': 'Validation Error'
            }
        
        # Attempt sentiment analysis
        result = analyze_sentiment(text)
        
        # Validate result
        if not result or 'sentiment' not in result:
            return {
                'error': "❌ Model failed to analyze text. Please try again.",
                'text': text,
                'sentiment': None,
                'confidence': 0.0,
                'raw_score': 0,
                'use_case': 'Analysis Error'
            }
        
        # Check if confidence is reasonable
        if result.get('confidence', 0) < 0.1:
            result['warning'] = "⚠️ Very low confidence score. Consider reviewing the text or providing more context."
        
        return result
        
    except Exception as e:
        return {
            'error': f"❌ Unexpected error during analysis: {str(e)}",
            'text': text,
            'sentiment': None,
            'confidence': 0.0,
            'raw_score': 0,
            'use_case': 'System Error'
        }

def safe_keyword_extraction(text):
    """
    Safe keyword extraction with error handling and user feedback.
    """
    import streamlit as st
    
    try:
        if not text or len(text.strip()) < 3:
            return []
        
        keywords = extract_keywords(text)
        return keywords if keywords else []
        
    except Exception as e:
        st.warning(f"⚠️ Keyword extraction failed: {str(e)}")
        return []

def display_error_with_help(error_msg, error_type="general", suggestions=None):
    """
    Display user-friendly error messages with helpful suggestions.
    """
    import streamlit as st
    
    # Map error types to emojis and colors
    error_config = {
        "validation": {"emoji": "⚠️", "color": "orange"},
        "file": {"emoji": "📁", "color": "red"},
        "processing": {"emoji": "🔧", "color": "red"},
        "network": {"emoji": "🌐", "color": "red"},
        "general": {"emoji": "❌", "color": "red"}
    }
    
    config = error_config.get(error_type, error_config["general"])
    
    st.error(f"{config['emoji']} {error_msg}")
    
    # Default suggestions based on error type
    default_suggestions = {
        "validation": [
            "Check your text length (3-5000 characters)",
            "Ensure text contains natural language",
            "Remove excessive special characters or formatting"
        ],
        "file": [
            "Verify your file format (.csv or .txt)",
            "Check file encoding (UTF-8 recommended)",
            "Ensure file size is under 10MB",
            "Remove empty rows or columns"
        ],
        "processing": [
            "Try with a smaller dataset",
            "Refresh the page and try again",
            "Check your internet connection"
        ],
        "network": [
            "Check your internet connection",
            "Try again in a few moments",
            "Contact support if issue persists"
        ],
        "general": [
            "Refresh the page (Ctrl+F5 or Cmd+R)",
            "Try with different input data",
            "Contact support if problem continues"
        ]
    }
    
    # Use provided suggestions or defaults
    help_suggestions = suggestions or default_suggestions.get(error_type, default_suggestions["general"])
    
    with st.expander("💡 Troubleshooting Tips", expanded=True):
        st.markdown("**Try these solutions:**")
        for i, suggestion in enumerate(help_suggestions, 1):
            st.markdown(f"{i}. {suggestion}")

def display_success_with_details(message, details=None):
    """
    Display success messages with optional details.
    """
    import streamlit as st
    
    st.success(f"✅ {message}")
    
    if details:
        with st.expander("📊 Details", expanded=False):
            for key, value in details.items():
                st.write(f"**{key}:** {value}")

def display_warning_with_action(message, action_suggestions=None):
    """
    Display warning messages with actionable suggestions.
    """
    import streamlit as st
    
    st.warning(f"⚠️ {message}")
    
    if action_suggestions:
        st.info("**Recommended actions:**")
        for suggestion in action_suggestions:
            st.write(f"• {suggestion}")

def validate_and_process_file(uploaded_file):
    """
    Comprehensive file validation and processing with detailed error handling.
    """
    import streamlit as st
    import pandas as pd
    
    try:
        # File size validation
        max_size_mb = 10
        if uploaded_file.size > max_size_mb * 1024 * 1024:
            display_error_with_help(
                f"File is too large ({uploaded_file.size / 1024 / 1024:.1f} MB). Maximum allowed size is {max_size_mb} MB.",
                "file",
                ["Split your file into smaller chunks", "Remove unnecessary columns", "Compress your data"]
            )
            return None, None
        
        # File type detection
        filename = uploaded_file.name.lower()
        if not (filename.endswith(".csv") or filename.endswith(".txt")):
            display_error_with_help(
                "Unsupported file type. Please upload a CSV or TXT file.",
                "file",
                ["Save your file as .csv or .txt format", "Check file extension is correct", "Convert from other formats (Excel, Word, etc.)"]
            )
            return None, None
        
        # Process based on file type
        df = None
        file_type = None
        
        if filename.endswith(".csv"):
            df, file_type = _process_csv_file(uploaded_file)
        elif filename.endswith(".txt"):
            df, file_type = _process_txt_file(uploaded_file)
        
        if df is None:
            return None, None
        
        # Validate content
        is_valid, error_msg, warning_msg = validate_file_content(df, file_type)
        
        if not is_valid:
            display_error_with_help(error_msg, "validation")
            return None, None
        
        if warning_msg:
            display_warning_with_action(warning_msg)
        
        # Success with details
        details = {
            "File Type": file_type.upper(),
            "Total Entries": len(df),
            "File Size": f"{uploaded_file.size / 1024:.1f} KB",
            "Encoding": "UTF-8"
        }
        display_success_with_details(f"File processed successfully!", details)
        
        return df, file_type
        
    except Exception as e:
        display_error_with_help(
            f"Unexpected error processing file: {str(e)}",
            "processing",
            ["Check file format and encoding", "Try saving file in UTF-8 format", "Contact support with error details"]
        )
        return None, None

def _process_csv_file(uploaded_file):
    """Helper function to process CSV files with encoding fallback."""
    import pandas as pd
    import streamlit as st
    
    try:
        # Try UTF-8 encoding first
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        return df, "csv"
    except UnicodeDecodeError:
        try:
            # Fallback to Latin-1 encoding
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')
            st.info("ℹ️ File encoding detected as Latin-1")
            return df, "csv"
        except Exception as e:
            display_error_with_help(
                f"Could not read CSV file: {str(e)}",
                "file",
                ["Save file with UTF-8 encoding", "Check for special characters", "Verify CSV formatting (commas, quotes)"]
            )
            return None, None
    except pd.errors.EmptyDataError:
        display_error_with_help(
            "CSV file appears to be empty or corrupted.",
            "file",
            ["Check file has content", "Verify file isn't corrupted", "Try re-saving the file"]
        )
        return None, None
    except pd.errors.ParserError as e:
        display_error_with_help(
            f"CSV parsing error: {str(e)}",
            "file",
            ["Check CSV formatting (proper commas, quotes)", "Remove special characters", "Verify file structure"]
        )
        return None, None
    except Exception as e:
        display_error_with_help(f"Unexpected error reading CSV: {str(e)}", "file")
        return None, None

def _process_txt_file(uploaded_file):
    """Helper function to process TXT files with encoding fallback."""
    import pandas as pd
    import streamlit as st
    
    try:
        # Try UTF-8 encoding first
        uploaded_file.seek(0)
        raw_text = uploaded_file.read().decode("utf-8")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        
        if not lines:
            display_error_with_help(
                "TXT file is empty or contains no valid text lines.",
                "file",
                ["Ensure file has content", "Check each line contains text", "Verify file isn't corrupted"]
            )
            return None, None
        
        df = pd.DataFrame(lines, columns=["text"])
        return df, "txt"
        
    except UnicodeDecodeError:
        try:
            # Fallback to Latin-1 encoding
            uploaded_file.seek(0)
            raw_text = uploaded_file.read().decode("latin-1")
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            df = pd.DataFrame(lines, columns=["text"])
            st.info("ℹ️ File encoding detected as Latin-1")
            return df, "txt"
        except Exception as e:
            display_error_with_help(
                f"Could not read TXT file: {str(e)}",
                "file",
                ["Save file with UTF-8 encoding", "Check for special characters", "Verify file contains text"]
            )
            return None, None
    except Exception as e:
        display_error_with_help(f"Error processing TXT file: {str(e)}", "file")
        return None, None 