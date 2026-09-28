# AI Context

## Project Identity

- Project ID: genai-001
- Project Title: Basic Gemini API Call
- Domain: Generative AI
- Project Type: Generative AI
- Difficulty: Beginner
- Status: Completed

## Project Purpose

This project demonstrates the most basic interaction between a Python
application and a Large Language Model (LLM) API.

The application sends a text question to Google Gemini and prints the
generated response.

The project intentionally uses a direct provider API call without
frameworks such as LangChain, LangGraph, or CrewAI.

## Learning Objective

The primary objective is to understand the basic request/response
pattern used when interacting with an LLM API.

The project is intended as a foundation for understanding how higher-level
AI frameworks ultimately interact with an underlying model API.

## Technologies

- Python
- Google Gemini API
- google-genai
- python-dotenv

## Model

The project currently uses:

- Model: gemini-3.5-flash

The exact model may change in future experiments without changing the
fundamental purpose of this project.

## Architecture

The application follows this basic flow:

User question
    ↓
Python application
    ↓
Google Gemini API
    ↓
Gemini model
    ↓
Generated text response
    ↓
Python application prints response

## Core Implementation

The project contains a function that accepts a question as a string,
sends it to Gemini, and returns the generated response as a string.

The basic interaction is conceptually:

ask_gemini(question)
    ↓
client.models.generate_content(...)
    ↓
Gemini response
    ↓
response.text

The generation configuration includes a maximum output-token setting.

## Key Concepts Demonstrated

- Calling an LLM API from Python
- API authentication using an environment variable
- Sending a text prompt to an LLM
- Receiving generated text from an LLM
- Basic generation configuration
- Handling API responses
- Understanding output-token limits

## Important Learning Observation

During development, a low `max_output_tokens` value caused the response
to terminate early with:

    finish_reason = MAX_TOKENS

The model consumed part of the generation budget for internal thinking,
leaving insufficient tokens for the visible response.

Increasing the configured output-token limit allowed the requested
three-sentence response to complete.

This demonstrates that the configured token budget can affect the
completeness of an LLM response.

## What This Project Does NOT Implement

This is intentionally a basic LLM API experiment.

It does not implement:

- RAG
- Vector databases
- Embeddings
- Agents
- Tool use
- Function calling
- Agent memory
- Multi-agent systems
- Fine-tuning
- Model training
- Evaluation pipelines
- MLOps

## Future Extension

Possible future extensions include:

- Building a Streamlit interface
- Supporting conversational interaction
- Adding prompt templates
- Comparing different Gemini models
- Adding structured output
- Adding retrieval or RAG in a later project

These extensions are not part of the current implementation.

## AI Retrieval Context

This project should be retrieved when the user asks about:

- Basic Gemini API usage
- Calling Gemini from Python
- Simple LLM API examples
- Google Gemini experiments
- LLM request/response patterns
- Basic prompting experiments
- `google-genai`
- `generate_content`
- `max_output_tokens`
- Beginner Generative AI projects
- Simple provider API experiments

This project is particularly relevant when the user is looking for an
introductory example before moving to frameworks, RAG, or agentic AI.

## Relationship to AI Project Atlas

This is a small daily learning project intended to become part of the
AI Project Atlas.

Its `project.yaml` provides structured project metadata, while this file
provides additional semantic context for future AI retrieval and the
Atlas AI Project Assistant.