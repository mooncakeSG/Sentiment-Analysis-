# 🔬 Sentiment Analysis Dashboard - Sample Guide

## Overview
This guide provides sample data and instructions for testing all three types of analysis available in the Sentiment Analysis Dashboard.

---

## 📝 Single Text Analysis

### How to Use:
1. Navigate to the **"📝 Single Text Analysis"** tab
2. Copy any sample text from `single_text_samples.txt`
3. Paste it into the text area
4. The analysis will run automatically

### What You'll See:
- **Sentiment Classification**: Very Positive, Positive, Neutral, Negative, or Very Negative
- **Confidence Score**: How certain the AI is about its prediction (0-100%)
- **Key Insights**: Important keywords and phrases that influenced the result
- **Analysis Details**: Processing time, model information, and reliability indicators

### Sample Categories Included:
- 🎉 **Very Positive**: Outstanding experiences, exceptional products
- 😊 **Positive**: Good experiences, satisfied customers
- 😐 **Neutral**: Factual information, balanced opinions
- 😔 **Negative**: Disappointing experiences, complaints
- 😢 **Very Negative**: Extremely poor experiences, angry feedback
- 🎯 **Mixed Sentiment**: Balanced reviews with pros and cons
- 💼 **Business Use Cases**: Professional communication, reports
- 📱 **Social Media**: Posts, comments, social interactions
- ✨ **Creative Content**: Literary, descriptive writing
- 📊 **Analytical Content**: Research, statistics, technical writing

---

## 📚 Batch Analysis

### CSV File Analysis:
1. Navigate to the **"📚 Batch Analysis"** tab
2. Download `batch_analysis_sample.csv` from the sample_data folder
3. Upload the CSV file using the file uploader
4. Click "Analyze" to process all texts at once

### TXT File Analysis:
1. Navigate to the **"📚 Batch Analysis"** tab
2. Download `batch_analysis_sample.txt` from the sample_data folder
3. Upload the TXT file using the file uploader
4. Click "Analyze" to process all texts at once

### What You'll See:
- **Results Table**: All texts with their sentiment classifications and confidence scores
- **Summary Statistics**: Overall sentiment distribution, average confidence, positive percentage
- **Visualizations**: 
  - Choose from Bar, Pie, Donut, or Line charts
  - Interactive charts with hover details
  - Word cloud generation from all analyzed texts
- **Export Options**:
  - 📊 **CSV**: Spreadsheet-ready data
  - 🔗 **JSON**: API-ready structured data
  - 📋 **PDF**: Professional report with your selected chart type

### Sample Data Features:
- **30 diverse texts** covering multiple industries and use cases
- **Balanced sentiment distribution** across all categories
- **Real-world scenarios**: Product reviews, customer service, social media, business communications
- **Various text lengths** from short posts to detailed reviews
- **Multiple domains**: E-commerce, entertainment, hospitality, technology, education, and more

---

## 🔄 Comparative Analysis

### How to Use:
1. Navigate to the **"🔄 Comparative Analysis"** tab
2. Set the number of texts to compare (2-5)
3. Enter different texts in each text area
4. Click "Compare Texts" to analyze all at once

### Recommended Sample Combinations:

#### Product Comparison:
```
Text 1: "This product is absolutely amazing! Best purchase I've ever made. Works perfectly and arrived quickly."
Text 2: "The product has some good features like fast performance and nice design, but it also has issues with battery life and the price is quite high."
Text 3: "Completely disappointed with this purchase. Poor quality and doesn't work properly."
```

#### Service Comparison:
```
Text 1: "Outstanding customer support! They resolved my issue quickly and professionally."
Text 2: "The service was okay, nothing special but got the job done."
Text 3: "Terrible customer service. They were rude and unhelpful. Complete waste of time!"
```

#### Review Evolution:
```
Text 1: "Excited about trying this new restaurant everyone's talking about!"
Text 2: "The restaurant was decent but not great. Some good moments but overall just okay."
Text 3: "Disappointed with our dinner. The food was cold and service was slow."
```

### What You'll See:
- **Side-by-side comparison** of all sentiment classifications
- **Confidence scores** for each analysis
- **Detailed explanations** for each text's classification
- **Individual visualizations** for each text
- **Word clouds** showing key terms for each text

---

## 🎯 Testing Different Features

### Chart Types Testing:
1. Upload the CSV sample file
2. In the Visualizations tab, try each chart type:
   - **Bar Chart**: Best for comparing sentiment counts
   - **Pie Chart**: Shows sentiment distribution as percentages
   - **Donut Chart**: Modern alternative to pie chart
   - **Line Chart**: Shows sentiment as a trend line

### PDF Export Testing:
1. Complete a batch analysis
2. Select different chart types
3. Notice how the PDF download button updates: "📋 Download PDF (Pie Chart)"
4. Download PDFs with different chart types to see the difference

### Error Handling Testing:
Try these scenarios to test the robust error handling:
- Upload a file with invalid formats
- Enter text that's too short (< 3 characters)
- Enter text that's too long (> 5000 characters)
- Upload an empty file
- Enter special characters or emojis

---

## 📊 Expected Results Distribution

Based on the sample data, you should expect approximately:
- **25% Very Positive/Positive**: Excellent experiences, recommendations
- **20% Neutral**: Factual information, balanced content
- **25% Negative/Very Negative**: Complaints, poor experiences
- **30% Mixed**: Nuanced opinions, constructive feedback

---

## 🔍 Advanced Testing Scenarios

### Industry-Specific Analysis:
- **E-commerce**: Product reviews, delivery feedback
- **Hospitality**: Hotel reviews, restaurant experiences
- **Technology**: App reviews, software feedback
- **Education**: Course evaluations, training feedback
- **Entertainment**: Movie reviews, concert experiences

### Sentiment Complexity Testing:
- **Sarcasm Detection**: "Oh great, another delay. Just what I needed today."
- **Mixed Emotions**: "Love the product but hate the customer service."
- **Subtle Sentiment**: "The meeting was... interesting."
- **Cultural Context**: Polite criticism vs. direct feedback

---

## 💡 Tips for Best Results

1. **Text Quality**: Use natural language for best accuracy
2. **Length**: 50-500 characters work best for most analyses
3. **Context**: Include relevant context for better keyword extraction
4. **File Formats**: CSV files should have text in the first column
5. **Batch Size**: For large datasets, consider processing in chunks

---

## 🚀 Ready to Start?

1. Start with **Single Text Analysis** using samples from `single_text_samples.txt`
2. Try **Batch Analysis** with the provided CSV or TXT files
3. Experiment with **Comparative Analysis** using the suggested combinations
4. Test different **chart types** and **export options**
5. Explore the **Q&A feature** by asking questions about your results

Happy analyzing! 🎉 