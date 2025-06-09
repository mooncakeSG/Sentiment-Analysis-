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

# Initialize models
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
keyword_model = KeyBERT()

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

def plot_sentiment_distribution(df):
    """
    Create sentiment distribution plot using matplotlib.
    Returns the figure object.
    """
    plt.figure(figsize=(10, 6))
    
    # Use custom colors for sentiments
    colors = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
    
    # Create the count plot with custom colors
    sns.countplot(data=df, x='sentiment', palette=colors)
    
    # Customize the plot
    plt.title('Sentiment Distribution', fontsize=14, pad=20)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add value labels on top of each bar
    for i in plt.gca().containers:
        plt.gca().bar_label(i, padding=3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    return buf

def export_to_pdf(df, plot_buf):
    """
    Export results to PDF using pdfkit.
    Returns the path to the generated PDF file.
    """
    # Create temporary HTML file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8') as f:
        # Convert plot to base64
        plot_base64 = base64.b64encode(plot_buf.getvalue()).decode()
        
        # Create HTML content
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Sentiment Analysis Results</h1>
            <h2>Sentiment Distribution</h2>
            <img src="data:image/png;base64,{plot_base64}">
            <h2>Detailed Results</h2>
            {df.to_html(index=False)}
        </body>
        </html>
        """
        f.write(html_content)
        html_path = f.name
    
    # Convert HTML to PDF
    pdf_path = html_path.replace('.html', '.pdf')
    pdfkit.from_file(html_path, pdf_path)
    return pdf_path

def get_download_link(file_path, link_text):
    """
    Generate a download link for a file.
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{link_text}">Download {link_text}</a>' 