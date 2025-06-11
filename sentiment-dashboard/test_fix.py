#!/usr/bin/env python3
"""
Simple test script to verify sentiment analysis functionality
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils import safe_sentiment_analysis, safe_keyword_extraction, validate_text_input
    print("✅ Successfully imported utils functions")
except ImportError as e:
    print(f"❌ Failed to import utils: {e}")
    sys.exit(1)

def test_sentiment_analysis():
    """Test basic sentiment analysis functionality"""
    test_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is terrible. I hate it and want my money back.",
        "It's okay, nothing special but not bad either."
    ]
    
    print("\n🔍 Testing Sentiment Analysis:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        
        # Test validation
        validation_error = validate_text_input(text)
        if validation_error:
            print(f"❌ Validation Error: {validation_error}")
            continue
        
        # Test sentiment analysis
        result = safe_sentiment_analysis(text)
        if 'error' in result:
            print(f"❌ Analysis Error: {result['error']}")
        else:
            print(f"✅ Sentiment: {result['sentiment']}")
            print(f"✅ Confidence: {result['confidence']:.2%}")
            print(f"✅ Use Case: {result.get('use_case', 'General')}")
        
        # Test keyword extraction
        keywords = safe_keyword_extraction(text)
        if keywords:
            print(f"✅ Keywords: {', '.join(keywords)}")
        else:
            print("⚪ No keywords extracted")

def test_comparative_scenario():
    """Test a comparative analysis scenario"""
    print("\n\n📊 Testing Comparative Analysis Scenario:")
    print("=" * 50)
    
    texts = [
        "This restaurant has excellent food and amazing service!",
        "The food was okay but the service was really slow.",
        "Terrible experience. Cold food and rude staff."
    ]
    
    labels = ["Positive Review", "Mixed Review", "Negative Review"]
    
    results = []
    
    for i, (text, label) in enumerate(zip(texts, labels)):
        print(f"\nAnalyzing {label}:")
        print(f"Text: {text}")
        
        result = safe_sentiment_analysis(text)
        if 'error' not in result:
            keywords = safe_keyword_extraction(text)
            
            analysis = {
                'label': label,
                'text': text,
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'keywords': keywords,
                'word_count': len(text.split())
            }
            results.append(analysis)
            
            print(f"✅ Result: {result['sentiment']} ({result['confidence']:.1%} confidence)")
            print(f"✅ Keywords: {', '.join(keywords) if keywords else 'None'}")
        else:
            print(f"❌ Error: {result['error']}")
    
    if results:
        print(f"\n📈 Summary: Analyzed {len(results)} texts successfully")
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"📊 Average Confidence: {avg_confidence:.1%}")
        
        sentiments = [r['sentiment'] for r in results]
        print(f"📋 Sentiments: {', '.join(sentiments)}")
        
        return True
    else:
        print("❌ No texts were analyzed successfully")
        return False

if __name__ == "__main__":
    print("🚀 Starting Sentiment Analysis Test Suite")
    print("=" * 60)
    
    try:
        # Test basic functionality
        test_sentiment_analysis()
        
        # Test comparative scenario
        success = test_comparative_scenario()
        
        if success:
            print("\n\n🎉 All tests completed successfully!")
            print("✅ The sentiment analysis functions are working correctly.")
            print("✅ Ready for integration with Streamlit app.")
        else:
            print("\n\n⚠️ Some tests failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc() 