# Pydantic Data Validation Guide

## What is Pydantic?

Pydantic is a Python library for data validation and settings management using Python type
annotations. You describe the "shape" of your data as a class, and Pydantic enforces that
shape at runtime, converting and validating input data as it goes. It is the validation
engine used internally by FastAPI for request and response models.

## Defining a Model

Models are defined by subclassing `BaseModel` and declaring fields with type annotations:

```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class User(BaseModel):
    id: int
    name: str
    email: str
    signup_ts: Optional[datetime] = None
    is_active: bool = True
```

Instantiating `User(**data)` with a dictionary of raw data (for example, JSON parsed from a
request) will validate every field, coerce compatible types (like a numeric string being
converted to `int`), and raise a `ValidationError` if something doesn't match.

## Validation Errors

When validation fails, Pydantic raises a `ValidationError` containing a list of every field
that failed, along with a human-readable message and an error "type" code. This structured
error format is what FastAPI turns into its 422 response bodies. You can catch and inspect
these errors directly:

```python
from pydantic import ValidationError

try:
    User(id="not-an-int", name="Ada", email="ada@example.com")
except ValidationError as e:
    print(e.json())
```

## Field Customization

The `Field` function lets you attach extra metadata and constraints to individual fields,
such as default values, aliases, minimum/maximum length, or numeric bounds:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0, description="Price in USD")
    tags: list[str] = Field(default_factory=list)
```

The `...` (Ellipsis) as the first argument marks a field as required with no default value.

## Custom Validators

For validation logic that goes beyond simple type/constraint checks, Pydantic supports
custom validator functions. In Pydantic v2 these are declared with the `field_validator`
decorator (in v1 the equivalent was `validator`):

```python
from pydantic import BaseModel, field_validator

class SignupForm(BaseModel):
    password: str
    password_confirm: str

    @field_validator("password_confirm")
    @classmethod
    def passwords_match(cls, v, info):
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("passwords do not match")
        return v
```

## Nested Models

Pydantic models can be nested inside one another, and Pydantic will recursively validate
each layer:

```python
class Address(BaseModel):
    street: str
    city: str

class Customer(BaseModel):
    name: str
    address: Address
```

Passing a nested dictionary for `address` will automatically construct and validate an
`Address` instance.

## Serialization

Models can be converted back to plain Python data structures or JSON with `model_dump()` and
`model_dump_json()` (in Pydantic v1 these were `.dict()` and `.json()`). You can control
which fields are included with arguments like `exclude`, `include`, and
`exclude_unset`, which is useful for partial updates (PATCH-style endpoints) where you only
want to know which fields the client actually sent.

## Settings Management

`pydantic-settings` (a companion package) provides `BaseSettings`, a model subclass designed
to read configuration values from environment variables and `.env` files, with the same
validation and type coercion as a regular model. This is a common pattern for centralizing
application configuration (API keys, database URLs, feature flags) in one typed object.
