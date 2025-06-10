# Test Use Case Detection Accuracy
from utils import determine_use_case

def test_use_case_accuracy():
    """Test the accuracy of use case detection with various examples."""
    
    test_cases = [
        # Product Reviews
        ("This fish tastes bad", "Product Review Classification"),
        ("The phone quality is excellent", "Product Review Classification"),
        ("I bought this laptop and it's amazing", "Product Review Classification"),
        ("Poor build quality for the price", "Product Review Classification"),
        
        # Social Media
        ("Just posted about my day", "Social Media Analysis"),
        ("Check out my new tweet", "Social Media Analysis"),
        ("Love this Instagram post", "Social Media Analysis"),
        ("Follow me for more updates", "Social Media Analysis"),
        
        # Customer Service
        ("I need help with my account", "Customer Service Optimization"),
        ("Can you resolve this issue?", "Customer Service Optimization"),
        ("Support was very helpful", "Customer Service Optimization"),
        ("Having problems with the system", "Customer Service Optimization"),
        
        # Customer Feedback
        ("The service experience was great", "Customer Feedback Analysis"),
        ("My feedback on your restaurant", "Customer Feedback Analysis"),
        ("Here's my review of the hotel", "Customer Feedback Analysis"),
        ("Rating your delivery service", "Customer Feedback Analysis"),
        
        # Brand Monitoring
        ("This company has great reputation", "Brand Monitoring"),
        ("Brand image needs improvement", "Brand Monitoring"),
        ("Competitor analysis shows...", "Brand Monitoring"),
        ("Market position is strong", "Brand Monitoring"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    print("🧪 USE CASE DETECTION ACCURACY TEST")
    print("=" * 50)
    
    for text, expected in test_cases:
        actual = determine_use_case(text)
        is_correct = actual == expected
        correct += is_correct
        
        status = "✅" if is_correct else "❌"
        print(f"{status} '{text[:30]}...' → {actual}")
        if not is_correct:
            print(f"   Expected: {expected}")
        print()
    
    accuracy = (correct / total) * 100
    print(f"📊 ACCURACY: {correct}/{total} = {accuracy:.1f}%")
    
    if accuracy < 80:
        print("🔧 NEEDS IMPROVEMENT!")
        suggest_improvements()
    else:
        print("✅ GOOD ACCURACY!")

def suggest_improvements():
    """Suggest improvements for better use case detection."""
    print("\n💡 IMPROVEMENT SUGGESTIONS:")
    print("1. Add more specific keywords for each category")
    print("2. Use weighted scoring for different keyword types")
    print("3. Consider context and word combinations")
    print("4. Add fallback logic for edge cases")

if __name__ == "__main__":
    test_use_case_accuracy() 