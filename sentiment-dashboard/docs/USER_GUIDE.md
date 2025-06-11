# Sentiment Analysis Dashboard - User Guide

## Table of Contents
- [Getting Started](#getting-started)
- [Feature Overview](#feature-overview)
- [Single Text Analysis](#single-text-analysis)
- [Batch Analysis](#batch-analysis)
- [Comparative Analysis](#comparative-analysis)
- [Understanding Results](#understanding-results)
- [Export and Reporting](#export-and-reporting)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [FAQ](#frequently-asked-questions)

## Getting Started

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 5GB free space for models and dependencies
- **Internet**: Required for initial setup and model downloads

### Quick Start
1. **Launch the Application**
   ```bash
   streamlit run app.py
   ```
   
2. **Access the Dashboard**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - The dashboard will load automatically

3. **First Time Setup**
   - The application will download required models (1-2 minutes)
   - Once loaded, all features will be available

## Feature Overview

### Main Components
The dashboard consists of three primary analysis modes:

| Feature | Purpose | Best For |
|---------|---------|----------|
| **Single Text Analysis** | Analyze individual texts | Quick sentiment checks, testing |
| **Batch Analysis** | Process multiple texts at once | Large datasets, bulk processing |
| **Comparative Analysis** | Compare 2-5 texts side-by-side | A/B testing, competitive analysis |

### Core Capabilities
- **5-Class Sentiment Scale**: Very Negative → Negative → Neutral → Positive → Very Positive
- **Confidence Scoring**: Reliability indicators for each prediction
- **Keyword Extraction**: Identify sentiment-driving terms
- **Interactive Visualizations**: Charts, graphs, and word clouds
- **Multiple Export Formats**: CSV, JSON, PDF reports

## Single Text Analysis

### Step-by-Step Guide

#### 1. Navigate to Single Text Analysis
- Click on the **"📝 Single Text Analysis"** tab
- This is the default view when you open the application

#### 2. Enter Your Text
- Use the large text area to input your content
- **Minimum**: 3 characters
- **Maximum**: 5,000 characters
- **Character counter** shows current length

#### 3. Analyze the Text
- Click the **"Analyze Sentiment"** button
- Wait for processing (typically <2 seconds)
- Results will appear below

#### 4. Review Results
The analysis provides:
- **Sentiment Classification**: Primary sentiment category
- **Confidence Score**: Reliability percentage
- **Keyword Extraction**: Sentiment-driving terms
- **Detailed Explanation**: AI-generated insights

### Example Walkthrough

**Input Text**: 
```
"I absolutely love this new smartphone! 
The camera quality is amazing and the battery life exceeds my expectations. 
However, the price is a bit steep for my budget."
```

**Expected Results**:
- **Sentiment**: Positive
- **Confidence**: 82%
- **Keywords**: love, amazing, exceeds, expectations, steep
- **Explanation**: Mixed sentiment with positive aspects (love, amazing) outweighing concerns (steep price)

### Advanced Features

#### Follow-Up Questions
After analysis, you can ask specific questions about the results:
- "Why was this classified as positive?"
- "What words contributed most to this sentiment?"
- "How reliable is this prediction?"

#### Confidence Interpretation
| Confidence Range | Interpretation | Recommended Action |
|-----------------|----------------|-------------------|
| 90-100% | Very High | Trust the result |
| 75-89% | High | Generally reliable |
| 60-74% | Moderate | Consider context |
| Below 60% | Low | Manual review recommended |

## Batch Analysis

### Preparing Your Data

#### Supported File Formats
- **CSV Files**: Comma-separated values
- **TXT Files**: Plain text, one entry per line

#### CSV Format Requirements
```csv
text,category,source
"Great product, highly recommend!",review,website
"Poor customer service experience",feedback,email
"The new feature works perfectly",testimonial,survey
```

#### TXT Format Example
```
Great product, highly recommend!
Poor customer service experience
The new feature works perfectly
```

### Step-by-Step Process

#### 1. Navigate to Batch Analysis
- Click the **"📚 Batch Analysis"** tab

#### 2. Upload Your File
- Click **"Choose file"** or drag and drop
- Select your CSV or TXT file
- File size limit: 50MB
- Maximum texts: 10,000 entries

#### 3. Configure Processing
- **Column Selection** (CSV): Choose the text column
- **Preview**: Review first few entries
- **Validation**: Check for formatting issues

#### 4. Start Analysis
- Click **"Start Batch Analysis"**
- Progress bar shows completion status
- Processing time varies by size (typically 1-5 minutes)

#### 5. Review Results
Three tabs provide different views:
- **📊 Results Table**: Detailed results with all texts
- **📈 Visualizations**: Charts and word clouds
- **📤 Export Options**: Download results

### Sample Batch Analysis Results

| Original Text | Sentiment | Confidence | Keywords |
|--------------|-----------|------------|----------|
| "Excellent service!" | Very Positive | 0.95 | excellent, service |
| "Could be better" | Neutral | 0.71 | better |
| "Terrible experience" | Very Negative | 0.92 | terrible, experience |

### Visualization Options

#### Chart Types Available
1. **Bar Chart**: Classic sentiment distribution
2. **Pie Chart**: Percentage breakdown
3. **Donut Chart**: Modern percentage view
4. **Line Chart**: Trend visualization

#### Word Cloud Features
- **Size**: Frequency-based word sizing
- **Color**: Sentiment-based coloring
- **Filtering**: Automatic removal of common words
- **Export**: Save as high-resolution image

## Comparative Analysis

### Use Cases
- **A/B Testing**: Compare different versions of content
- **Competitive Analysis**: Evaluate competitor messaging
- **Campaign Optimization**: Test multiple marketing messages
- **Product Comparison**: Analyze different product descriptions

### Step-by-Step Guide

#### 1. Access Comparative Analysis
- Click the **"🔄 Comparative Analysis"** tab

#### 2. Enter Multiple Texts
- **Minimum**: 2 texts required
- **Maximum**: 5 texts supported
- Each text can be up to 5,000 characters

#### 3. Run Comparison
- Click **"Compare Texts"**
- Analysis runs on all texts simultaneously
- Results appear in side-by-side format

#### 4. Analyze Results
The comparison provides:
- **Side-by-side sentiment scores**
- **Confidence comparisons**
- **Keyword differences**
- **Recommendations**

### Example Comparison

**Text A**: "Our new product is revolutionary and will change the industry!"
**Text B**: "This product has some interesting features but needs improvement."

**Results**:
| Metric | Text A | Text B |
|--------|--------|--------|
| Sentiment | Very Positive | Neutral |
| Confidence | 0.89 | 0.76 |
| Key Positive | revolutionary, change | interesting, features |
| Key Negative | - | needs, improvement |

**Recommendation**: Text A is more positively received but may seem overly promotional. Text B is more balanced but less engaging.

## Understanding Results

### Sentiment Classifications

#### 5-Class System Explained
1. **Very Negative** (0.0-0.2): Strong negative sentiment
   - Examples: "Terrible", "Awful", "Hate it"
   
2. **Negative** (0.2-0.4): Moderate negative sentiment
   - Examples: "Disappointing", "Not good", "Poor quality"
   
3. **Neutral** (0.4-0.6): Balanced or no clear sentiment
   - Examples: "It's okay", "Average", "No opinion"
   
4. **Positive** (0.6-0.8): Moderate positive sentiment
   - Examples: "Good", "Satisfied", "Recommend"
   
5. **Very Positive** (0.8-1.0): Strong positive sentiment
   - Examples: "Excellent", "Amazing", "Love it"

### Confidence Scores

#### What They Mean
- **High Confidence (>0.8)**: Model is very certain about the classification
- **Medium Confidence (0.6-0.8)**: Model is reasonably confident
- **Low Confidence (<0.6)**: Model is uncertain, manual review suggested

#### Factors Affecting Confidence
- **Text Length**: Very short or very long texts may have lower confidence
- **Language Complexity**: Technical or ambiguous language reduces confidence
- **Mixed Sentiment**: Texts with both positive and negative elements
- **Sarcasm/Irony**: Difficult for models to detect

### Keyword Extraction

#### How It Works
- **KeyBERT Algorithm**: Identifies semantically important terms
- **Sentiment Context**: Highlights words that drive sentiment
- **Frequency Weighting**: More important words appear more prominently

#### Interpreting Keywords
- **Size**: Larger keywords are more important
- **Color**: May indicate sentiment association
- **Relevance**: Keywords should relate to the main sentiment drivers

## Export and Reporting

### Available Export Formats

#### CSV Export
- **Content**: All analysis results in spreadsheet format
- **Columns**: Text, Sentiment, Confidence, Keywords
- **Use Case**: Further analysis in Excel or other tools

#### JSON Export
- **Content**: Structured data with all metadata
- **Format**: Machine-readable for API integration
- **Use Case**: Programming and automation

#### PDF Reports
- **Content**: Comprehensive formatted report
- **Includes**: Visualizations, summaries, detailed results
- **Use Case**: Presentation and sharing

### Customizing Reports

#### PDF Report Sections
1. **Executive Summary**: Key findings and statistics
2. **Methodology**: Analysis approach and model information
3. **Results Overview**: High-level insights
4. **Detailed Analysis**: Individual text results
5. **Visualizations**: Charts and word clouds
6. **Recommendations**: Actionable insights

#### Report Customization Options
- **Chart Type Selection**: Choose visualization style
- **Filtering**: Include/exclude specific results
- **Branding**: Add company information (when customized)

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Problems
**Symptoms**: Error messages during startup, blank screens
**Solutions**:
- Ensure internet connection for initial download
- Check available disk space (5GB required)
- Restart the application
- Clear browser cache

#### 2. File Upload Issues
**Symptoms**: Upload fails, file not recognized
**Solutions**:
- Check file format (CSV or TXT only)
- Verify file size (<50MB)
- Ensure CSV has proper headers
- Try different file encoding (UTF-8 recommended)

#### 3. Processing Errors
**Symptoms**: Analysis fails, timeout errors
**Solutions**:
- Reduce batch size (<1000 texts recommended)
- Check text length (<5000 characters per text)
- Ensure stable internet connection
- Restart if memory issues occur

#### 4. Visualization Problems
**Symptoms**: Charts not displaying, white screens
**Solutions**:
- Refresh the page
- Try different chart types
- Check browser JavaScript settings
- Update browser to latest version

#### 5. Export Failures
**Symptoms**: Download doesn't start, corrupted files
**Solutions**:
- Ensure results are fully loaded
- Try different export format
- Check browser download settings
- Restart analysis if needed

### Error Messages Explained

| Error Message | Meaning | Solution |
|--------------|---------|----------|
| "Text too short" | Input less than 3 characters | Add more content |
| "Text too long" | Input exceeds 5000 characters | Reduce text length |
| "Model not loaded" | Sentiment analysis model unavailable | Restart application |
| "Processing timeout" | Analysis took too long | Reduce batch size |
| "Invalid file format" | Unsupported file type | Use CSV or TXT files |

## Best Practices

### For Optimal Results

#### Text Preparation
1. **Clean Text**: Remove excessive formatting, special characters
2. **Appropriate Length**: 10-500 words work best
3. **Clear Language**: Avoid overly technical jargon
4. **Context**: Include sufficient context for accurate analysis

#### Batch Processing
1. **File Organization**: Use clear column headers in CSV files
2. **Size Management**: Process large datasets in smaller batches
3. **Quality Control**: Review sample results before full processing
4. **Data Backup**: Keep original files as backup

#### Interpretation Guidelines
1. **Consider Confidence**: Always check confidence scores
2. **Context Matters**: Understand domain-specific language
3. **Multiple Perspectives**: Use comparative analysis for important decisions
4. **Human Oversight**: Review low-confidence predictions manually

### Performance Optimization

#### For Better Speed
- **Close Other Applications**: Free up system memory
- **Stable Internet**: Ensure reliable connection
- **Optimal Batch Size**: Use 50-100 texts per batch
- **Regular Restarts**: Restart app after large processing jobs

#### For Better Accuracy
- **Quality Input**: Use well-written, clear text
- **Appropriate Domain**: Model works best on social media/review-style text
- **Length Optimization**: Aim for 50-200 words per text
- **Language Consistency**: Use consistent English

## Frequently Asked Questions

### General Questions

**Q: How accurate is the sentiment analysis?**  
A: The system achieves 84% accuracy on benchmark datasets. Accuracy varies by text type and complexity.

**Q: Can I use this for languages other than English?**  
A: The current model is optimized for English. Limited support for other languages may work but with reduced accuracy.

**Q: Is my data secure?**  
A: Yes, all processing happens locally on your machine. No data is sent to external servers.

**Q: How many texts can I process at once?**  
A: The system supports up to 10,000 texts per batch, though smaller batches (50-500) process faster.

### Technical Questions

**Q: Why are some confidence scores low?**  
A: Low confidence typically indicates ambiguous text, mixed sentiment, or content outside the model's training domain.

**Q: Can I customize the sentiment categories?**  
A: The current version uses a fixed 5-class system. Customization requires code modifications.

**Q: How long does processing take?**  
A: Single texts: <2 seconds. Batch processing: 0.1-0.5 seconds per text depending on system performance.

**Q: What happens if the analysis fails?**  
A: The system includes error handling to gracefully manage failures and provide informative error messages.

### Usage Questions

**Q: What's the best file format for batch analysis?**  
A: CSV is recommended for structured data with multiple columns. TXT works well for simple text lists.

**Q: Can I analyze social media posts?**  
A: Yes, the model is specifically trained on social media content and performs well on tweets, posts, and comments.

**Q: How do I interpret mixed sentiment results?**  
A: Mixed sentiment often appears as "Neutral" with moderate confidence. Review keywords to understand the balance.

**Q: Can I save my analysis sessions?**  
A: Currently, sessions are temporary. Use the export features to save results permanently.

---

## Support and Resources

### Getting Help
- **Documentation**: Refer to this guide and API documentation
- **Sample Data**: Use provided sample packs for testing
- **Error Messages**: Check the troubleshooting section
- **Community**: Report issues on the project GitHub page

### Additional Resources
- [Sample Pack Guide](../SAMPLE_PACK_GUIDE.md): Pre-configured test datasets
- [API Documentation](API_DOCUMENTATION.md): Technical implementation details
- [Accuracy Report](ACCURACY_REPORT.md): Performance evaluation and limitations

---

*User Guide Version: 1.0*  
*Last Updated: January 2025*  
*For Technical Support: Contact the development team* 