"""
PA - Day 8 - Track H (MLOps & Serving): Subtopic 1 - FastAPI basics

A small in-memory Prompt Store API used to practise:
- FastAPI routes and HTTP methods
- Path and query parameters
- Pydantic request validation
- HTTP status codes and error handling
- Automatic OpenAPI / Swagger documentation

No LLM API key or external database is required.

Run:
    python -m pip install -r requirements.txt
    uvicorn main:app --reload

Open:
    http://127.0.0.1:8000/docs

Test:
    pytest -q
"""

from typing import Any

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field


app = FastAPI(
    title="Prompt Store API",
    description=(
        "A learning API demonstrating FastAPI routes, request validation, "
        "status codes, and automatic API documentation."
    ),
    version="0.2.0",
)


# In-memory storage. Data is lost whenever the application process restarts.
PROMPTS: dict[int, dict[str, Any]] = {}
_next_id = 1


class PromptIn(BaseModel):
    """Request model used when creating a prompt."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Prompt text (1 to 2,000 characters).",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Illustrative model temperature; no model is called.",
    )


class PromptOut(PromptIn):
    """Response model returned after a prompt is stored."""

    id: int


class TokenEstimateIn(BaseModel):
    """Request model for the rough token-estimation endpoint."""

    text: str = Field(..., min_length=1)


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    """Return a welcome message."""
    return {
        "message": "Prompt Store API is running. Open /docs to explore the API."
    }


@app.get("/health", tags=["General"])
def health() -> dict[str, str]:
    """Basic health-check endpoint."""
    return {"status": "ok"}


@app.post(
    "/prompts",
    response_model=PromptOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Prompts"],
)
def create_prompt(body: PromptIn) -> dict[str, Any]:
    """Validate and store a prompt, then return the created record."""
    global _next_id

    record = {"id": _next_id, **body.model_dump()}
    PROMPTS[_next_id] = record
    _next_id += 1

    return record


@app.get(
    "/prompts",
    response_model=list[PromptOut],
    tags=["Prompts"],
)
def list_prompts(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of prompts to return (1 to 100).",
    ),
) -> list[dict[str, Any]]:
    """Return stored prompts, limited by the query parameter."""
    return list(PROMPTS.values())[:limit]


@app.get(
    "/prompts/{prompt_id}",
    response_model=PromptOut,
    tags=["Prompts"],
)
def get_prompt(prompt_id: int) -> dict[str, Any]:
    """Return one prompt by its ID, or raise HTTP 404 if it is missing."""
    if prompt_id not in PROMPTS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prompt not found",
        )

    return PROMPTS[prompt_id]


@app.post("/estimate-tokens", tags=["Utilities"])
def estimate_tokens(body: TokenEstimateIn) -> dict[str, int]:
    """
    Return a rough token estimate based on character count.

    This is only a learning approximation (about four English characters
    per token). It is not a model tokenizer and must not be used for
    authoritative token counts or billing.
    """
    character_count = len(body.text)
    estimated_token_count = max(1, character_count // 4)

    return {
        "characters": character_count,
        "estimated_tokens": estimated_token_count,
    }
