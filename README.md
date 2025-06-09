# Sentiment Analysis Dashboard

An interactive dashboard built with Streamlit for analyzing sentiment in text using state-of-the-art natural language processing.

## Features

- Single text sentiment analysis
- Batch processing via CSV upload
- Multiple visualization options:
  - Bar charts
  - Pie charts
  - Donut charts
  - Line charts (for temporal analysis)
  - Word clouds
- Export capabilities:
  - CSV export
  - JSON export
  - PDF reports with visualizations

## Technologies Used

- Hugging Face Transformers for sentiment analysis
- KeyBERT for keyword extraction
- Streamlit for the web interface
- Plotly for interactive visualizations
- WordCloud for word cloud generation
- ReportLab for PDF generation

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd Interactive-Dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the dashboard:
```bash
cd sentiment-dashboard
streamlit run app.py
```

## Usage

1. **Single Text Analysis**
   - Enter text in the input field
   - View sentiment analysis results and visualizations
   - Export results in various formats

2. **Batch Analysis**
   - Upload a CSV file with a text column
   - View aggregate results and visualizations
   - Export batch analysis results

## Project Structure

```
sentiment-dashboard/
├── app.py              # Main Streamlit application
├── utils.py            # Utility functions
└── requirements.txt    # Project dependencies
```

## Dependencies

See `requirements.txt` for a complete list of dependencies. 