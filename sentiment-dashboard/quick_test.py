from utils import determine_use_case

# Test the specific example from the screenshot
test_text = "This fish tastes bad"
result = determine_use_case(test_text)

print(f"Text: '{test_text}'")
print(f"Detected Use Case: {result}")

# Test a few more examples
test_cases = [
    "This fish tastes bad",
    "The phone quality is excellent", 
    "Just posted a new tweet",
    "I need help with my order",
    "Great customer service experience"
]

print("\n" + "="*50)
print("MULTIPLE TEST CASES:")
print("="*50)

for text in test_cases:
    case = determine_use_case(text)
    print(f"'{text}' → {case}") 