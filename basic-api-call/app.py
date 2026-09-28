import streamlit as st
from main import ask_gemini

st.title("Basic Gemini API Call")

st.write("Ask a question and get a response from Google Gemini.")

question = st.text_area(
    "Your Question",
    placeholder="Enter Your Question Here..."
)

if st.button("Ask Gemini"):
    if not question.strip():
        st.warning("Please enter a question here...")
    else:
        with st.spinner("Gemini is thinking..."):
            answer = ask_gemini(question)

        st.subheader("Gemini's Answer")
        st.write(answer)