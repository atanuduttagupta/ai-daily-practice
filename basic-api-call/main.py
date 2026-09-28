"""
PA Day 1 - Track A (Provider APIs): Subtopic 1 - Basic completion call
Provider: Any (free tier, no credit card needed)

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
from groq import Groq
from groq import APIConnectionError
from groq import APIStatusError
from groq import AuthenticationError

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = os.environ.get("MODEL")

class LLMError(Exception):
    """Base class for expected LLM application errors."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM configuration is missing or invalid."""


class LLMQuotaError(LLMError):
    """Raised when the provider quota or rate limit is exceeded."""


class LLMUnavailableError(LLMError):
    """Raised when the provider is temporarily unavailable."""


class LLMAuthenticationError(LLMError):
    """Raised when the API authentication fails."""


class LLMRequestError(LLMError):
    """Raised when the provider rejects the request."""


def ask_llm(question: str, max_output_tokens: int = 1000) -> str:
    """Send a question to the configured LLM provider."""

    if not LLM_PROVIDER:
        raise LLMConfigurationError(
            "LLM provider is not configured."
        )

    if LLM_PROVIDER.lower() == "groq":
        return ask_groq(question, max_output_tokens)

    raise LLMConfigurationError(
        f"Unsupported LLM provider: {LLM_PROVIDER}"
    )

def ask_groq(question: str, max_output_tokens: int = 1000) -> str:
    """Send a question to Groq and return the generated response."""

    if not GROQ_API_KEY:
        raise LLMConfigurationError(
            "Groq API key is not configured."
        )

    try:

        client = Groq(api_key = GROQ_API_KEY)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            max_tokens=max_output_tokens,
        )

        return response.choices[0].message.content

    except AuthenticationError as error:
        raise LLMAuthenticationError(
            "The AI service authentication failed."
        ) from error

    except APIConnectionError as error:
        raise LLMUnavailableError(
            "The AI service could not be reached."
        ) from error

    except APIStatusError as error:
        if error.status_code == 429:
            raise LLMQuotaError(
                "The AI service usage limit has been reached."
            ) from error

        if error.status_code >= 500:
            raise LLMUnavailableError(
                "The AI service is temporarily unavailable."
            ) from error

        raise LLMRequestError(
            "The AI service could not process the request."
        ) from error

def main():

    question = input(f'Ask a Question to {MODEL}: ');

    print(f"MODEL: {MODEL}")
    print(f"Question: {question}")

    answer = ask_llm(question)

    print("LLM's answer: ")
    print(answer)

if __name__ == "__main__":
    main();
