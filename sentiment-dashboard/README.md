# Sentiment Analysis Dashboard

A powerful Streamlit-based dashboard for analyzing sentiment in text using state-of-the-art natural language processing models.

## Features

- Single text analysis with sentiment classification and keyword extraction
- Batch analysis through CSV file upload
- Five-class sentiment analysis (Very Positive, Positive, Neutral, Negative, Very Negative) with confidence scores
- Automatic use case detection for:
  - Social media analysis
  - Customer feedback analysis
  - Product reviews classification
  - Brand monitoring
  - Market research
  - Customer service optimization
  - Competitive intelligence
- Keyword extraction using KeyBERT
- Interactive visualization of sentiment distribution
- Export results in multiple formats (CSV, JSON, PDF)
- Modern, responsive user interface

## Use Cases

### Social Media Analysis
Monitor and analyze social media posts, comments, and reactions to understand public sentiment about your brand, products, or services.

### Customer Feedback Analysis
Process customer reviews, surveys, and feedback forms to gain insights into customer satisfaction and areas for improvement.

### Product Reviews Classification
Automatically categorize and analyze product reviews to understand customer sentiment and identify common praise or concerns.

### Brand Monitoring
Track brand mentions and sentiment across various channels to maintain brand reputation and respond to customer sentiment.

### Market Research
Analyze market trends, consumer opinions, and competitor reviews to inform business strategy and decision-making.

### Customer Service Optimization
Evaluate customer service interactions to improve response quality and customer satisfaction.

### Competitive Intelligence
Monitor competitor mentions and analyze customer sentiment about competing products or services.

## Prerequisites

- Python 3.8 or higher
- wkhtmltopdf (required for PDF export)

### Installing wkhtmltopdf

- **Windows**: Download and install from [wkhtmltopdf downloads](https://wkhtmltopdf.org/downloads.html)
- **Mac**: `brew install wkhtmltopdf`
- **Linux**: `sudo apt-get install wkhtmltopdf`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sentiment-dashboard
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Use the dashboard by either:
   - Entering text directly in the "Single Text Analysis" tab
   - Uploading a CSV file in the "Batch Analysis" tab

## Supported File Formats

### CSV Files
For batch analysis, prepare your CSV file with a single column containing the text to analyze. The first row should be the column header. Example:

```csv
text
This is a great product!
The service was terrible.
I'm neutral about this experience.
```

### TXT Files
For simple text files, put one sentence or text entry per line. Example:

```txt
This is a great product!
The service was terrible.
I'm neutral about this experience.
```

The system automatically detects the file format and processes it accordingly.

## Deployment

The app is ready to be deployed on Render. Use the following command as the start command:

```bash
streamlit run app.py --server.port=$PORT --server.enableCORS=false
```

## License

MIT

## Acknowledgments

- Hugging Face Transformers
- KeyBERT
- Streamlit
- The open-source NLP community 