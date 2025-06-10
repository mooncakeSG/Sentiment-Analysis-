# 🔬 Sentiment Analysis Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful, AI-driven sentiment analysis dashboard built with Streamlit that provides comprehensive text sentiment analysis using state-of-the-art natural language processing models. Perfect for businesses, researchers, and developers who need to understand sentiment patterns in text data.

![Sentiment Analysis Dashboard](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

## 🚀 Features

### Core Functionality
- **🎯 Single Text Analysis**: Analyze individual texts with real-time sentiment classification
- **📊 Batch Processing**: Upload CSV or TXT files for large-scale sentiment analysis
- **🎨 Interactive Visualizations**: Beautiful charts and word clouds for data insights
- **📁 Multiple Export Formats**: Export results as CSV, JSON, or PDF reports
- **🔍 Keyword Extraction**: Automatic keyword identification using KeyBERT
- **📱 Responsive Design**: Modern, mobile-friendly user interface

### Advanced Features
- **🧠 Five-Class Sentiment Analysis**: Very Positive, Positive, Neutral, Negative, Very Negative
- **📈 Confidence Scoring**: Get confidence levels for each sentiment prediction
- **🎯 Automatic Use Case Detection**: Smart detection for social media, reviews, feedback analysis
- **⚡ Optimized Performance**: Cached models and batch processing for speed
- **🛡️ Error Handling**: Robust error handling with helpful user guidance
- **🔧 Memory Optimization**: Efficient memory usage for large datasets

## 🎯 Use Cases

| Use Case | Description | Benefits |
|----------|-------------|----------|
| **Social Media Monitoring** | Analyze posts, comments, and reactions | Track brand sentiment, viral trends |
| **Customer Feedback Analysis** | Process reviews, surveys, support tickets | Improve products and services |
| **Market Research** | Understand consumer opinions and trends | Data-driven business decisions |
| **Brand Monitoring** | Track brand mentions across platforms | Reputation management |
| **Product Review Analysis** | Categorize and analyze product feedback | Quality improvement insights |
| **Customer Service Optimization** | Evaluate service interactions | Enhance customer satisfaction |
| **Competitive Intelligence** | Monitor competitor sentiment | Strategic positioning |

## 🛠️ Installation

### Prerequisites
- **Python 3.8+**
- **wkhtmltopdf** (for PDF export functionality)

#### Installing wkhtmltopdf
```bash
# Windows - Download from: https://wkhtmltopdf.org/downloads.html

# macOS
brew install wkhtmltopdf

# Ubuntu/Debian
sudo apt-get install wkhtmltopdf

# CentOS/RHEL
sudo yum install wkhtmltopdf
```

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/mooncakeSG/Sentiment-Analysis-.git
cd Sentiment-Analysis-
```

2. **Set up virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
cd sentiment-dashboard
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

## 📖 Usage

### Single Text Analysis
1. Navigate to the "Single Text Analysis" tab
2. Enter your text in the input field
3. Click "Analyze Sentiment"
4. View results including sentiment, confidence score, and keywords

### Batch Analysis
1. Go to the "Batch Analysis" tab
2. Upload a CSV or TXT file
3. Review the data preview
4. Click "Analyze Batch"
5. Download results in your preferred format

### File Format Requirements

#### CSV Files
```csv
text
"This product is amazing!"
"The service was disappointing."
"Neutral experience overall."
```

#### TXT Files
```txt
This product is amazing!
The service was disappointing.
Neutral experience overall.
```

## 📊 Output Formats

- **CSV**: Structured data with sentiment scores and metadata
- **JSON**: Machine-readable format for API integration
- **PDF**: Professional reports with visualizations

## 🏗️ Project Structure

```
Sentiment-Analysis-/
├── sentiment-dashboard/
│   ├── app.py                 # Main Streamlit application
│   ├── utils.py              # Core utility functions
│   ├── optimization.py       # Performance optimizations
│   ├── visualizations.py     # Chart and visualization components
│   ├── requirements.txt      # Python dependencies
│   ├── .streamlit/           # Streamlit configuration
│   ├── sample_data/          # Sample datasets and examples
│   │   ├── README.md
│   │   ├── analysis_guide.md
│   │   └── sample_*.csv
│   └── README.md            # Detailed dashboard documentation
└── README.md                # This file
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the `sentiment-dashboard` directory:

```env
# Optional: Configure model settings
HUGGINGFACE_MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest
KEYWORD_MODEL_NAME=all-MiniLM-L6-v2

# Optional: Performance settings
BATCH_SIZE=32
MAX_FILE_SIZE=50MB
```

### Streamlit Configuration
The app includes optimized Streamlit settings in `.streamlit/config.toml`:

```toml
[theme]
base = "light"
primaryColor = "#4F46E5"

[server]
maxUploadSize = 50
```

## 🚀 Deployment

### Render Deployment
The application is ready for deployment on Render with these settings:

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
streamlit run app.py --server.port=$PORT --server.enableCORS=false
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY sentiment-dashboard/ .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y wkhtmltopdf

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🧪 Testing

Run the test suite:
```bash
cd sentiment-dashboard
python -m pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Hugging Face](https://huggingface.co/)** - For the transformer models
- **[KeyBERT](https://github.com/MaartenGr/KeyBERT)** - For keyword extraction
- **[Streamlit](https://streamlit.io/)** - For the web framework
- **[Plotly](https://plotly.com/)** - For interactive visualizations

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mooncakeSG/Sentiment-Analysis-/issues)
- **Documentation**: Check the `sentiment-dashboard/README.md` for detailed usage
- **Samples**: Explore the `sentiment-dashboard/sample_data/` directory

## 🔄 Version History

- **v1.0.0** - Initial release with core sentiment analysis
- **v1.1.0** - Added batch processing and visualizations
- **v1.2.0** - Performance optimizations and error handling
- **v1.3.0** - Enhanced UI/UX and export capabilities

---

<div align="center">

**[⭐ Star this repository](https://github.com/mooncakeSG/Sentiment-Analysis-)** if you found it helpful!

Made with ❤️ by the Tech Titanians

</div> 