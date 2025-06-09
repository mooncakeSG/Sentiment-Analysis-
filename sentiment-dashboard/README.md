# Sentiment Analysis Dashboard

A powerful Streamlit-based dashboard for analyzing sentiment in text using state-of-the-art natural language processing models.

## Features

- Single text analysis with sentiment classification and keyword extraction
- Batch analysis through CSV file upload
- Multi-class sentiment analysis (Positive, Neutral, Negative) with confidence scores
- Keyword extraction using KeyBERT
- Interactive visualization of sentiment distribution
- Export results in multiple formats (CSV, JSON, PDF)
- Modern, responsive user interface

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

## CSV File Format

For batch analysis, prepare your CSV file with a single column containing the text to analyze. The first row should be the column header. Example:

```csv
text
This is a great product!
The service was terrible.
I'm neutral about this experience.
```

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