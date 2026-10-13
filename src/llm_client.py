"""
Thin wrapper around an OpenAI-compatible API (Groq or OpenAI).

Rules for this file:
- NO mock or fallback answers. If the API cannot be reached, raise an error.
- The API key comes only from .env. It is never printed.
- Token counts come only from the API's own usage report.
"""
import os
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

load_dotenv()

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

if not API_KEY or API_KEY.startswith("your_"):
    raise RuntimeError(
        "API_KEY is missing or still the placeholder. "
        "Put your real key in the .env file (API_KEY=...)."
    )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_agent(prompt: str, temperature: float = 0.7, max_retries: int = 5) -> dict:
    """
    Call the model once. Returns {"text": str, "tokens": int}.
    Retries with exponential backoff on rate limits, network errors and 5xx errors.
    Any other error is raised so it can never be mistaken for a real answer.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            if response.usage is None:
                raise RuntimeError("API response had no token usage information.")
            return {
                "text": response.choices[0].message.content or "",
                "tokens": response.usage.total_tokens,
            }
        except (RateLimitError, APIConnectionError) as err:
            wait = min(2 ** attempt, 30)
            print(f"  [retry {attempt + 1}/{max_retries}] {type(err).__name__}; waiting {wait}s")
            time.sleep(wait)
        except APIStatusError as err:
            if err.status_code >= 500:
                wait = min(2 ** attempt, 30)
                print(f"  [retry {attempt + 1}/{max_retries}] server error {err.status_code}; waiting {wait}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"API call failed after {max_retries} attempts.")
