"""
Quick fix for indentation error in app.py
"""

# Read the file and fix the problematic section
with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Fix the problematic line around 1070
problematic_text = """                            )
                            
                                                         # Fallback to deployment-safe processing
                             try:"""

fixed_text = """                            )
                            
                            # Fallback to deployment-safe processing
                            try:"""

# Replace the problematic section
content = content.replace(problematic_text, fixed_text)

# Write back the corrected file
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed indentation error in app.py") 