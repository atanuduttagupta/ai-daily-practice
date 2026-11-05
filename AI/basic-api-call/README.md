# Basic LLM API Call

A small beginner-friendly Generative AI project demonstrating how a Python application can call a Large Language Model (LLM) through an API and expose the interaction through a Streamlit web interface.

## Project Details

- **Project ID:** `genai-001`
- **Domain:** Generative AI
- **Project Type:** `generative_ai`
- **Difficulty:** Beginner
- **Current LLM Provider:** Groq
- **Current Model:** `openai/gpt-oss-20b`
- **Python SDK:** Groq Python SDK
- **Web UI:** Streamlit
- **Hosted Demo:** https://ai-daily-practice-basic-api-call.streamlit.app/

## What This Project Demonstrates

```text
User Question
     ↓
Streamlit UI
     ↓
ask_llm()
     ↓
Configured LLM Provider
     ↓
Groq API
     ↓
GPT-OSS 20B
     ↓
Generated Response
     ↓
Streamlit UI
```

The application is provider-configurable rather than hard-coded around a single LLM vendor.

## Current LLM

The current implementation uses **OpenAI GPT-OSS 20B served through Groq**.

Model ID:

```text
openai/gpt-oss-20b
```

Groq documents GPT-OSS 20B as a 20B-parameter open-weight Mixture-of-Experts model. It supports text input/output, has a 131,072-token context window, and a maximum output of 65,536 tokens. This project intentionally exposes a much smaller configurable output limit through its Streamlit slider.

Official model documentation:
https://console.groq.com/docs/model/openai/gpt-oss-20b

## Why Groq?

Groq provides a straightforward API and official Python SDK for calling hosted LLMs.

Official Groq quickstart:
https://console.groq.com/docs/quickstart

## Features

- Ask a question to an LLM.
- Configure the maximum output token limit.
- Select the LLM provider through configuration.
- Select the model through configuration.
- Keep API credentials outside the source code.
- Run locally using a `.env` file.
- Run on Streamlit Community Cloud using Streamlit Secrets.
- Gracefully handle common API failures.

## Maximum Output Tokens

The Streamlit application provides a slider:

```text
Minimum: 100
Maximum: 2000
Default: 1000
Step: 100
```

The selected value is passed to the LLM request as the maximum number of output tokens requested by the application.

This setting is separate from the provider's API quota and rate limits.

## API Error Handling

The application converts provider-specific errors into application-level errors.

Examples include:

- `LLMConfigurationError`
- `LLMQuotaError`
- `LLMUnavailableError`
- `LLMAuthenticationError`
- `LLMRequestError`

For example:

```text
API quota / rate limit
        ↓
LLMQuotaError
        ↓
Friendly Streamlit warning
```

Connection failures and provider 5xx responses are handled as temporary service availability problems.

The application does not intentionally expose raw provider exceptions or API credentials to normal users.

## Project Structure

```text
basic-api-call/
├── app.py
├── main.py
├── project.yaml
├── ai-context.md
├── README.md
├── requirements.txt
├── .gitignore
└── .env                  # Local only - NOT committed
```

## Requirements

- Python 3.12+ recommended
- A Groq API key
- Internet connection for API calls

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/atanuduttagupta/ai-daily-practice.git
cd ai-daily-practice/basic-api-call
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## Obtain a Groq API Key

The application requires a Groq API key.

1. Create or sign in to a Groq account.
2. Open the Groq Console.
3. Open the API Keys section.
4. Create an API key.
5. Copy the key and keep it private.

Groq's official quickstart recommends using an environment variable for the API key rather than putting the key directly into source code.

Official Groq Quickstart:
https://console.groq.com/docs/quickstart

API key page:
https://console.groq.com/keys

**Never commit the actual API key to GitHub.**

## `.env` Configuration

The real `.env` file is intentionally **not included in this repository** because it contains a private API credential.

Create a local file named:

```text
.env
```

Use this template:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key-here
MODEL=openai/gpt-oss-20b
```

Replace `your-groq-api-key-here` with your actual Groq API key.

**Do not share your real `.env` file publicly.**

The `.env` file is excluded through `.gitignore`.

## Run the Python Example

From the `basic-api-call` directory:

```powershell
python main.py
```

The application sends a sample question to the configured LLM and prints the generated response.

## Run the Streamlit Application

From the `basic-api-call` directory:

```powershell
streamlit run app.py
```

Streamlit will provide a local URL where the application can be opened in a browser.

## Hosted Demo

The project is deployed on Streamlit Community Cloud:

https://ai-daily-practice-basic-api-call.streamlit.app/

The deployed application uses Streamlit Secrets rather than the local `.env` file.

## Streamlit Secrets

For Streamlit Community Cloud, configure the equivalent values in the application's Secrets settings:

```toml
LLM_PROVIDER = "groq"
GROQ_API_KEY = "your-groq-api-key-here"
MODEL = "openai/gpt-oss-20b"
```

The actual key should only be entered into Streamlit Secrets and must never be committed to GitHub.

## Configuration Design

The application uses:

```text
LLM_PROVIDER
GROQ_API_KEY
MODEL
```

instead of hard-coding provider and model details throughout the application.

The Streamlit layer calls:

```python
ask_llm(question, max_output_tokens)
```

The current provider-specific implementation is:

```python
ask_groq(question, max_output_tokens)
```

This creates a simple path for adding another provider later.

## Learning Outcomes

After completing this project, the learner should understand:

- What an LLM API is.
- How a Python application sends a prompt to an LLM.
- How generated text is returned by an API.
- How API credentials should be managed.
- How environment variables can externalize configuration.
- How model selection can be configured.
- What maximum output tokens control.
- Why API quota and token limits are different concepts.
- How to handle common API failures.
- How to expose an LLM application through Streamlit.
- Why provider-specific implementation should be separated from the application UI.

## What Is Not Included

This is intentionally a small foundational project.

It does not currently implement:

- RAG
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
- MLOps
- Persistent conversation history

## Possible Future Extensions

Possible extensions include:

- Additional LLM providers
- Conversational chat history
- Prompt templates
- Structured output
- Model comparison
- Streaming responses
- Retry and exponential backoff
- More detailed observability
- RAG
- Tool calling
- Agent workflows

## GitHub Repository

https://github.com/atanuduttagupta/ai-daily-practice

## AI Project Atlas

This project is intended to eventually be integrated into **AI Project Atlas**.

Stable project ID:

```text
genai-001
```

The project represents a foundational Generative AI capability:

```text
Python Application
       ↓
LLM API
       ↓
Generated Text
```

It can provide a foundation for later projects involving prompt engineering, structured output, RAG, tool use, and agentic AI.

## Security Reminder

Never commit:

```text
GROQ_API_KEY=your-real-key
```

or any other actual API credential.

The repository should contain only the `.env` template shown above, not the real `.env` file.

## Official References

- Groq Quickstart: https://console.groq.com/docs/quickstart
- Groq API Keys: https://console.groq.com/keys
- GPT-OSS 20B on Groq: https://console.groq.com/docs/model/openai/gpt-oss-20b
- Groq Supported Models: https://console.groq.com/docs/models
