# FastAPI Fundamentals

## What is FastAPI?

FastAPI is a modern Python web framework for building APIs. It is built on top of Starlette
(for the web-handling parts) and Pydantic (for the data-validation parts). The two headline
features are speed of development and automatic interactive documentation.

Because FastAPI relies on standard Python type hints, most editors and type checkers can
understand your API code without any special plugins, and FastAPI itself uses those same
hints to validate request data, serialize responses, and generate an OpenAPI schema.

## Creating a Basic Application

A minimal FastAPI application looks like this:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

Running `uvicorn main:app --reload` starts a development server. FastAPI automatically
serves interactive API docs at `/docs` (Swagger UI) and `/redoc` (ReDoc), generated from
the function signatures and type hints.

## Path Parameters and Query Parameters

Path parameters are declared as part of the route, e.g. `/items/{item_id}`. Any function
parameter that isn't part of the path is treated as a query parameter automatically. Type
hints (`int`, `str`, `bool`, etc.) tell FastAPI how to convert and validate the incoming
value, and it will return a 422 Unprocessable Entity response automatically if the value
does not match the expected type.

## Request Bodies

Request bodies are usually declared using Pydantic models. FastAPI reads the JSON body of
the incoming request, validates it against the model, converts it into a Python object, and
passes it into your function:

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None

@app.post("/items/")
def create_item(item: Item):
    return item
```

If the client sends invalid data (missing field, wrong type), FastAPI responds with a 422
status code and a JSON body describing exactly which field failed validation and why.

## Dependency Injection

FastAPI has a built-in dependency injection system based on the `Depends` function. A
dependency is just a callable (often a function) that FastAPI will call for you before your
path operation runs, and whose return value gets passed into your function as an argument.
This is commonly used for shared logic such as database sessions, authentication checks, or
pagination parameters, so that this logic doesn't need to be repeated in every route.

```python
from fastapi import Depends

def get_query_params(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

@app.get("/items/")
def list_items(params: dict = Depends(get_query_params)):
    return params
```

## Error Handling

Route handlers can raise `HTTPException` to return a specific HTTP status code and detail
message:

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return fake_db[item_id]
```

You can also register custom exception handlers with `@app.exception_handler(...)` to
control the response format for specific exception types across the whole application.

## Async Support

Path operation functions can be declared with `async def` instead of `def`. FastAPI will run
`async def` functions directly on the event loop, while regular `def` functions are run in a
thread pool so that blocking code doesn't block the whole server. As a rule of thumb: use
`async def` when everything inside the function is awaitable (e.g. an async database
driver or async HTTP client); otherwise a plain `def` is usually simpler and safer.

## Middleware

Middleware wraps every request/response cycle and is useful for cross-cutting concerns like
CORS, logging, or timing. FastAPI (via Starlette) provides `app.add_middleware(...)` for
built-in middleware classes such as `CORSMiddleware`, and also supports custom middleware
functions decorated with `@app.middleware("http")`.
