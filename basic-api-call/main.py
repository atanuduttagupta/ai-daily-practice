"""
PA Day 1 - Track A (Provider APIs): Subtopic 1 - Basic completion call
Provider: Google Gemini (free tier, no credit card needed)

Goal: make a single, minimal call to an LLM API and print the response.
No framework, no agent, no RAG yet - just the raw request/response shape.
Every framework (LangChain, LangGraph, CrewAI, ...) sits on top of a call
that looks like this one underneath.

Setup:
    1. Get a free key at https://aistudio.google.com (sign in with Google,
       click "Get API key")
    2. pip install -r requirements.txt
    3. export GEMINI_API_KEY="your-key-here"
       (Windows cmd: set GEMINI_API_KEY=...   PowerShell: $env:GEMINI_API_KEY="...")
    4. python main.py
"""

import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("MODEL")

def ask_gemini(question: str) -> str:
    """Send one question to Gemini and return the text of the reply."""
    if not GEMINI_API_KEY:
        print("ERROR: Set the API Key first")
        sys.exit(1)

    client = genai.Client(api_key = GEMINI_API_KEY)

    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(max_output_tokens=1000)
    )

    return response.text

def main():

    question = input(f'Ask a Question to Gemini {MODEL}: ');

    print(f"MODEL: {MODEL}")
    print(f"Question: {question}")

    answer = ask_gemini(question)

    print("Gemini's answer: ")
    print(answer)

if __name__ == "__main__":
    main();
