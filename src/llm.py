
from __future__ import annotations
from typing import Type, TypeVar
from pydantic import BaseModel

import os
from google import genai
from google.genai import types

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-2.0-flash-exp"  # if not available, switch to "gemini-2.0-flash"

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment variables.")
    return genai.Client(api_key=api_key)

def json_from_llm(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0
) -> T:
    """
    Uses Gemini structured outputs with a Pydantic schema.
    """
    client = get_client()

    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Content(role="user", parts=[
                types.Part(text=f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_prompt}")
            ])
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,  # Pydantic model supported by google-genai SDK
        ),
    )

    # google-genai returns parsed output under .parsed when schema is provided
    if getattr(resp, "parsed", None) is None:
        raise RuntimeError(f"Model did not return parsed structured output. Raw text: {getattr(resp, 'text', '')}")

    return resp.parsed
