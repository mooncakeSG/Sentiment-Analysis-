import streamlit as st

st.title("Debug Test")

if st.button("Test Button"):
    st.write("Button clicked!")
    st.success("Success message")
    
texts = ["Sample text 1", "Sample text 2"]
if len(texts) >= 2:
    st.write(f"Found {len(texts)} texts")
    if st.button("Run Analysis Test"):
        st.write("Analysis button clicked!")
        for i, text in enumerate(texts):
            st.write(f"Text {i+1}: {text}")
else:
    st.write("Not enough texts") 