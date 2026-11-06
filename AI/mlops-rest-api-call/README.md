# Prompt Store API --- FastAPI Basics

**Learning track:** PA --- Day 8\
**Track:** H --- MLOps & Serving\
**Subtopic:** 1 --- FastAPI Basics\
**Level:** Beginner

## 1. Project Overview

This project builds a small REST API using **FastAPI**. It acts as an
in-memory Prompt Store: clients can submit text prompts, list stored
prompts, retrieve a prompt by its ID, and request a rough token-count
estimate.

The goal is to learn the building blocks used to expose software and
future AI features over HTTP: routes, HTTP methods, path and query
parameters, request validation with Pydantic, status codes, automatic
API documentation, and basic API testing.

The project deliberately uses a stand-in prompt store instead of calling
an LLM. No model credentials, paid API, or external database are
required.

## 2. What Is an API?

An **Application Programming Interface (API)** is a defined way for one
software component to interact with another.

A web API makes this interaction available over a network, commonly
using HTTP. A client sends a request to an endpoint; the server
processes it and returns a response.

-   **Client:** The browser, script, application, or service sending a
    request.
-   **Server:** The application receiving and processing the request.
-   **Endpoint:** A URL and HTTP method through which an operation is
    exposed.
-   **Request:** The method, URL, headers, and optionally a body sent by
    the client.
-   **Response:** The status code, headers, and optionally a body
    returned by the server.

The API contract describes available endpoints, accepted inputs, and
response formats.

### 2.1 Client--server model

For example, a client sends prompt text to this API. The server
validates the data, stores the prompt in memory, and returns a response
containing the assigned ID. The client does not need to know how the
server stores the record internally; it interacts through the API
contract.

## 3. REST and HTTP Fundamentals

REST (Representational State Transfer) is an architectural style
commonly used for web APIs. REST-style APIs typically expose resources
through URLs and use standard HTTP methods to describe operations.

This project exposes a `prompts` resource.

  HTTP method   Common meaning                            Project example
  ------------- ----------------------------------------- -----------------
  `GET`         Retrieve a resource or collection         `GET /prompts`
  `POST`        Submit data, often to create a resource   `POST /prompts`

The project implements prompt creation and retrieval. It does not
currently implement update or delete operations.

### 3.1 Routes and endpoints

A route connects an HTTP method and URL pattern to a Python function.

``` python
@app.get("/health")
def health():
    return {"status": "ok"}
```

When a client sends `GET /health`, FastAPI calls `health()` and converts
its returned dictionary into a JSON response.

### 3.2 Path parameters

A path parameter is part of the URL path and commonly identifies a
particular resource.

Example: `GET /prompts/3`

``` python
@app.get("/prompts/{prompt_id}")
def get_prompt(prompt_id: int):
    ...
```

FastAPI parses the path value according to the declared type. If the ID
is not present in the in-memory store, the endpoint raises an HTTP 404
error.

### 3.3 Query parameters

A query parameter is supplied after `?` in the URL. It is commonly used
to filter, sort, or limit a collection.

Example: `GET /prompts?limit=5`

The project declares a default limit of 10 and constrains the value to
1--100:

``` python
def list_prompts(limit: int = Query(10, ge=1, le=100)):
    ...
```

`ge` means greater than or equal to; `le` means less than or equal to.

### 3.4 Request bodies and JSON

A request body carries data submitted by the client. JSON is a common
format for web API request and response bodies.

Example body for `POST /prompts`:

``` json
{
  "text": "Explain retrieval-augmented generation.",
  "temperature": 0.7
}
```

FastAPI can parse this JSON into a declared Pydantic model.

## 4. FastAPI

**FastAPI** is a Python framework for building APIs. It uses Python type
annotations to describe inputs and outputs and to support request
parsing, validation, and API schema generation.

``` python
app = FastAPI(title="Prompt Store API", version="0.1.0")
```

The `app` object is the application instance. Decorators such as
`@app.get()` and `@app.post()` register endpoint functions.

### 4.1 Why type annotations matter

A declaration such as `prompt_id: int` tells FastAPI that the path
parameter should be an integer. FastAPI uses this declaration when
parsing requests and generating the API schema.

Type annotations make the API contract explicit and reduce repetitive
input-handling code.

### 4.2 ASGI and Uvicorn

FastAPI is an ASGI-compatible framework. **ASGI (Asynchronous Server
Gateway Interface)** defines how an asynchronous Python web application
communicates with a web server.

**Uvicorn** is an ASGI server that runs the application and accepts
incoming HTTP requests.

``` bash
uvicorn main:app --reload
```

-   `main` identifies the Python module (`main.py`).
-   `app` identifies the FastAPI application object inside that module.
-   `--reload` restarts the development server when source files change.
    It is intended for development, not production deployment.

## 5. Pydantic and Request Validation

**Pydantic** is a Python data-validation library. FastAPI integrates it
to validate structured request data using model declarations.

``` python
class PromptIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
```

This model specifies:

-   `text` is required because `...` marks it as required.
-   `text` must contain 1--2,000 characters.
-   `temperature` is optional because it has a default value of `0.7`.
-   `temperature` must be between 0.0 and 2.0, inclusive.

If the request body violates these constraints, FastAPI returns a
validation response before the endpoint's business logic runs.

### 5.1 Input models and output models

``` python
class PromptOut(PromptIn):
    id: int
```

`PromptOut` inherits the prompt fields and adds an ID. The
`response_model` parameter tells FastAPI which response structure to
validate and document.

Separating input and output models is useful because data accepted from
a client may differ from data returned by the server. The server assigns
the prompt ID, so the client does not supply it when creating a prompt.

## 6. HTTP Status Codes and Error Handling

HTTP status codes communicate the outcome of a request.

  ----------------------------------------------------------------------------
  Code                         Category                Meaning in this project
  ---------------------------- ----------------------- -----------------------
  `201 Created`                Success                 A prompt was created

  `404 Not Found`              Client error            The requested prompt ID
                                                       does not exist

  `422 Unprocessable Entity`   Client error            Request data failed
                                                       validation
  ----------------------------------------------------------------------------

### 6.1 Creation status

The create endpoint declares `status_code=status.HTTP_201_CREATED`, so
successful prompt creation returns HTTP 201.

### 6.2 Raising an HTTP error

When a prompt ID is absent, the endpoint raises:

``` python
raise HTTPException(status_code=404, detail="Prompt not found")
```

This stops normal endpoint processing and returns an HTTP error response
with a detail message.

### 6.3 Automatic validation errors

If a client submits empty `text`, an invalid `temperature`, or an
invalid `limit`, FastAPI and Pydantic generate a validation error
response. The endpoint does not need to implement each check manually.

## 7. Project Implementation

### 7.1 In-memory prompt store

The application uses a Python dictionary:

``` python
PROMPTS: dict[int, dict] = {}
_next_id = 1
```

Each prompt is stored under an integer ID. `_next_id` supplies the next
ID when a prompt is created.

This is intentionally simple for learning. It is **not persistent
storage**: the dictionary is reset when the server process restarts. It
is also not a substitute for a database in a multi-worker or production
application.

### 7.2 Implemented endpoints

  ------------------------------------------------------------------------
  Endpoint                 Method                  Purpose
  ------------------------ ----------------------- -----------------------
  `/`                      GET                     Returns a welcome
                                                   message

  `/health`                GET                     Returns a basic health
                                                   status

  `/prompts`               POST                    Validates and stores a
                                                   prompt; returns its ID

  `/prompts`               GET                     Lists stored prompts,
                                                   with a configurable
                                                   limit

  `/prompts/{prompt_id}`   GET                     Retrieves one prompt by
                                                   ID, or returns 404

  `/estimate-tokens`       POST                    Returns character count
                                                   and a rough token
                                                   estimate

  `/docs`                  GET                     Opens interactive API
                                                   documentation
  ------------------------------------------------------------------------

### 7.3 Token estimation

The `/estimate-tokens` endpoint uses this rough rule:

``` python
estimated_tokens = max(1, len(text) // 4)
```

It estimates approximately one token per four characters for English
text. This is **not an exact tokenizer**. Actual token counts depend on
the model's tokenizer and content, including language, punctuation,
numbers, and code.

Do not use this estimate as an authoritative token count or billing
calculation.

## 8. OpenAPI and Swagger UI

FastAPI generates an **OpenAPI schema** from application routes,
parameter declarations, Pydantic models, and response definitions.

OpenAPI is a machine-readable description of an API. It can describe
endpoints, methods, parameters, request bodies, response structures, and
status codes.

FastAPI provides an interactive Swagger UI page at:

`http://127.0.0.1:8000/docs`

The page allows developers to inspect endpoints and send test requests
from a browser. The documentation is generated from the same
declarations used by the application, helping keep the API contract and
documentation aligned.

## 9. Testing the API

The project includes seven tests. Run them with:

``` bash
pytest -q
```

The README's example checks include:

-   Submit an empty `text` value to `POST /prompts` and observe
    validation error `422`.
-   Request `GET /prompts/999` and observe `404` when that ID is not
    stored.
-   Request `GET /prompts?limit=0` and observe query-parameter
    validation.

Testing verifies expected responses and error behaviour without relying
only on manual browser interaction.

## 10. How to Run the Project

### 10.1 Install dependencies

``` bash
pip install -r requirements.txt
```

The current `requirements.txt` contains:

-   `fastapi` --- API framework.
-   `uvicorn` --- ASGI server.
-   `httpx` --- HTTP client dependency used by the test setup.
-   `pytest` --- test runner.

### 10.2 Start the development server

``` bash
uvicorn main:app --reload
```

### 10.3 Open API documentation

Visit `http://127.0.0.1:8000/docs` and try the endpoints using Swagger
UI.

### 10.4 Run the tests

In another terminal, from the project directory:

``` bash
pytest -q
```

## 11. Connection to AI Serving and MLOps

An AI capability needs more than a model call to become usable by other
applications. It also needs an interface that accepts requests,
validates inputs, handles errors, and returns predictable responses.

The patterns practised here can later be applied to:

-   **LLM inference:** accept a prompt and return a model response.
-   **Embedding generation:** accept text and return vector
    representations.
-   **RAG retrieval:** accept a question and return relevant passages.
-   **Agent execution:** accept a task and return an execution result.
-   **Evaluation:** accept model outputs and return evaluation results.

These are possible future applications of the API concepts; they are not
implemented in this Prompt Store project.

## 12. Current Scope and Limitations

This is a learning API, not a production prompt-management service.

-   Data is stored only in process memory and is lost when the server
    restarts.
-   There is no database integration.
-   There is no authentication or authorization.
-   There is no LLM call or model inference.
-   Token estimation is approximate.
-   The API supports prompt creation and retrieval, not update or
    deletion.
-   The development server uses `--reload`; production deployment
    requires an appropriate configuration.

These limitations are deliberate so the project can focus on FastAPI
fundamentals.

## 13. Key Learning

The central lesson is that **Python type declarations, Pydantic models,
and FastAPI route definitions work together to create a structured HTTP
API**.

FastAPI uses these declarations to parse and validate incoming data,
produce appropriate responses, and generate interactive documentation.

This is a foundation for serving AI features: model or retrieval logic
can later be placed behind API routes, while the API layer provides a
consistent interface for client applications.

## 14. Next Topic

The next topic in this learning track is **Dockerizing an application**:
packaging the API so it can run consistently across environments.
