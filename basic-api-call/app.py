import streamlit as st
from main import ask_gemini

st.title("Basic Gemini API Call")

st.write("Ask a question and get a response from Google Gemini.")


question = st.text_area(
    "Your Question",
    placeholder="Enter Your Question Here..."
)

token_length = st.slider(
    "Maximum output tokens",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100
)

if st.button("Ask Gemini"):
    if not question.strip():
        st.warning("Please enter a question here...")
    else:
        with st.spinner("Gemini is thinking..."):
            answer = ask_gemini(question, token_length)

        st.subheader("Gemini's Answer")
        st.write(answer)
