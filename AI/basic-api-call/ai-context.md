# AI Context — Basic LLM API Call

## Project Identity

- Project ID: `genai-001`
- Title: Basic LLM API Call
- Domain: Generative AI
- Project Type: `generative_ai`
- Difficulty: beginner
- Status: completed

## Purpose

This project demonstrates the basic interaction between a Python application and a Large Language Model (LLM) through an API.

The project started as a basic Gemini API experiment and was later refactored into a provider-neutral LLM API application.

The current implementation uses Groq as the configured LLM provider.

The project demonstrates:

- Sending a user question to an LLM
- Receiving generated text
- Configuring the LLM provider
- Configuring the model
- Configuring the maximum output token limit
- Handling common API errors
- Providing a Streamlit user interface

## Core Learning Objective

Understand the basic application flow:

User Question → Python Application → LLM API → LLM → Generated Response

The project also demonstrates how an application can avoid hard-coding a specific LLM provider.

## Current LLM Configuration

The provider and model are configured through environment variables rather than being hard-coded into the application.

Current configuration:

- LLM provider: Groq
- Model: configured through the `MODEL` environment variable
- API key: configured through `GROQ_API_KEY`
- Provider selection: configured through `LLM_PROVIDER`

Example local configuration:

```text
LLM_PROVIDER=groq
GROQ_API_KEY=<secret>
MODEL=openai/gpt-oss-20b
```

The actual API key must never be committed to GitHub.

For Streamlit Community Cloud, the API key and configuration are stored as Streamlit Secrets.

## Technologies

- Python
- Groq API
- Groq Python SDK
- python-dotenv
- Streamlit

## Application Architecture

```text
User
  |
  v
Streamlit UI
  |
  v
ask_llm()
  |
  v
Configured LLM Provider
  |
  v
Groq API
  |
  v
GPT-OSS 20B
  |
  v
Generated Response
  |
  v
Streamlit UI
```

The Streamlit layer does not directly depend on the Groq API.

Instead, the application calls:

```python
ask_llm(question, max_output_tokens)
```

The provider configuration determines which implementation is used.

## Provider Abstraction

The application uses a provider-neutral function:

```python
ask_llm()
```

The current provider-specific implementation is:

```python
ask_groq()
```

Conceptually:

```text
ask_llm()
    |
    +-- Groq → ask_groq()
    |
    +-- Future provider → provider-specific implementation
```

This design allows another LLM provider to be added later without requiring the Streamlit UI to know provider-specific details.

## Configuration

The application reads configuration from environment variables.

Important configuration values:

- `LLM_PROVIDER`
- `GROQ_API_KEY`
- `MODEL`

This avoids hard-coding provider names, model names, and API credentials in the source code.

## Maximum Output Tokens

The Streamlit application provides a slider for maximum output tokens.

Current range:

```text
Minimum: 100
Maximum: 2000
Default: 1000
Step: 100
```

The selected value is passed to the LLM request.

Conceptually:

```text
User selects token limit
        |
        v
Streamlit
        |
        v
ask_llm(question, token_length)
        |
        v
LLM API
```

The token limit controls the maximum output requested from the model. It does not represent the API request quota.

## API Error Handling

The application converts provider-specific API failures into application-level exceptions.

Defined application errors include:

- `LLMConfigurationError`
- `LLMQuotaError`
- `LLMUnavailableError`
- `LLMAuthenticationError`
- `LLMRequestError`

### Quota / Rate Limit

Provider HTTP status `429` is converted into:

```text
LLMQuotaError
```

The Streamlit application displays a friendly message rather than exposing the raw provider exception.

Example situation:

```text
API quota exhausted
        |
        v
LLMQuotaError
        |
        v
Friendly Streamlit warning
```

### Service Unavailability

Provider HTTP `5xx` responses and connection failures are handled as service availability problems.

They are converted into:

```text
LLMUnavailableError
```

The user receives a friendly message indicating that the AI service is temporarily unavailable.

### Authentication Errors

Authentication failures are converted into:

```text
LLMAuthenticationError
```

The application does not expose sensitive API information to the user.

### Request Errors

Other provider request failures are converted into:

```text
LLMRequestError
```

This keeps provider-specific exception details out of the normal user interface.

## Security

API credentials must never be stored in source code.

Local development uses:

```text
.env
```

The `.env` file is excluded through `.gitignore`.

The deployed Streamlit application uses Streamlit Secrets.

The following must never be committed:

```text
GROQ_API_KEY
```

or any other actual API credential.

## Streamlit Application

The project provides an interactive Streamlit interface.

The user can:

1. Enter a question.
2. Select the maximum output token limit.
3. Submit the question.
4. Receive a generated response.
5. See a friendly message when common API errors occur.

Hosted application:

https://ai-daily-practice-basic-api-call.streamlit.app/

## GitHub Repository

https://github.com/atanuduttagupta/ai-daily-practice

## What This Project Teaches

This project is intentionally small.

Its main purpose is to understand the fundamental mechanics of connecting an application to an LLM API before introducing more advanced AI architecture.

Key concepts:

- LLM API
- API request
- API response
- Prompt
- Model selection
- Provider configuration
- Environment variables
- API credentials
- Maximum output tokens
- Streamlit
- API error handling
- Quota handling
- Service availability
- Provider abstraction

## What Is Not Implemented

This project does not currently implement:

- Retrieval-Augmented Generation (RAG)
- Embeddings
- Vector databases
- Hybrid search
- Knowledge graphs
- GraphRAG
- Agents
- Tool calling
- Agent memory
- Multi-agent systems
- Fine-tuning
- Model training
- LLM evaluation pipelines
- MLOps pipelines
- Production authentication
- Persistent conversation history

These are outside the scope of this beginner project.

## Future Extensions

Possible future extensions include:

- Support for additional LLM providers
- Conversational chat history
- Prompt templates
- Structured output
- Model comparison
- Streaming responses
- Retry and backoff strategies
- More detailed API observability
- RAG integration
- Tool calling
- Agent-based workflows

These are future possibilities, not current project capabilities.

## AI Retrieval Context

Useful concepts and keywords for future AI Project Atlas retrieval:

- Basic LLM API
- LLM API integration
- Generative AI
- Large Language Model
- Python LLM application
- Groq API
- GPT-OSS 20B
- LLM provider abstraction
- Provider configuration
- Model configuration
- Prompting
- Text generation
- Maximum output tokens
- API quota
- Rate limiting
- HTTP 429
- HTTP 5xx
- API availability
- Authentication errors
- Streamlit LLM application
- Environment variables
- API key security
- Error handling

## AI Project Atlas Relationship

This project is a small Generative AI learning project intended to eventually be integrated into AI Project Atlas.

Stable project identity:

```text
genai-001
```

The project demonstrates a foundational Generative AI capability:

```text
Python Application
        ↓
LLM API
        ↓
Generated Text
```

It can serve as a foundation for later projects involving:

```text
LLM API
   ↓
Prompt Engineering
   ↓
Structured Output
   ↓
RAG
   ↓
Tool Use
   ↓
Agents
```

## Project Philosophy

The project favors simple, readable implementation over unnecessary abstraction.

Configuration should be externalized where practical so that provider and model changes do not require rewriting the application.

The application should also fail gracefully when external AI services are unavailable or when API usage limits are reached.
