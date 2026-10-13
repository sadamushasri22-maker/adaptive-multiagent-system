"""
src/llm_client.py: Wrapper around OpenAI-compatible API (Groq or OpenAI).
Tracks token usage per call and handles rate limits with exponential backoff.
Includes an optional offline mock mode (MOCK_LLM=true) for local pipeline validation without burning API credits.
"""
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

# Load environment variables from .env file
load_dotenv()

_client = None


def get_client() -> OpenAI:
    """
    Lazily initializes and returns the OpenAI client.
    Ensures API_KEY is read securely from environment or .env without hardcoding.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("API_KEY")
    if not api_key or api_key == "your_groq_api_key_here" or api_key == "your_openai_api_key_here":
        raise ValueError(
            "API_KEY is missing or contains placeholder. Please set your real API_KEY in .env "
            "(or set MOCK_LLM=true in .env for offline dry-run testing)."
        )

    base_url = os.environ.get("BASE_URL", "https://api.openai.com/v1")
    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def _mock_call(prompt: str) -> dict:
    """
    Simulates model responses for dry-run offline testing when MOCK_LLM=true.
    Extracts numbers or simple code to let all pipeline modes run without network.
    """
    # If prompted for Python code in verifier or code agent:
    if "Python" in prompt or "```python" in prompt:
        # Check if asked to compute math
        if "compute the exact numeric answer" in prompt or "math problem" in prompt:
            # Find numbers in prompt
            nums = re.findall(r"\b\d+\b", prompt)
            val = nums[-1] if nums else "42"
            return {
                "text": f"```python\nprint({val})\n```",
                "tokens": 40,
            }
        # Otherwise code generation
        fn_match = re.search(r"def (\w+)", prompt) or re.search(r"`(\w+)\(", prompt)
        fn_name = fn_match.group(1) if fn_match else "solution"
        return {
            "text": f"```python\ndef {fn_name}(*args, **kwargs):\n    return True\n```",
            "tokens": 55,
        }
    # Math problem prompt:
    nums = re.findall(r"\b\d+\b", prompt)
    val = nums[-1] if nums else "42"
    return {
        "text": f"Let's solve step by step.\nResult is {val}.\nFINAL ANSWER: {val}",
        "tokens": 45,
    }


def call_agent(prompt: str, temperature: float = 0.7, max_retries: int = 5, initial_backoff: float = 1.0) -> dict:
    """
    Calls the LLM with exponential backoff on rate-limit (429) and network errors.
    Returns:
        {
            "text": str (model response text),
            "tokens": int (total tokens consumed by this call)
        }
    """
    if os.environ.get("MOCK_LLM", "").lower() in ("true", "1"):
        return _mock_call(prompt)

    client = get_client()
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    delay = initial_backoff
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            total_tokens = response.usage.total_tokens if response.usage else 0
            return {
                "text": response.choices[0].message.content or "",
                "tokens": total_tokens,
            }
        except (RateLimitError, APIConnectionError, APIStatusError) as e:
            status_code = getattr(e, "status_code", None)
            is_rate_limit = isinstance(e, RateLimitError) or status_code == 429
            is_connection = isinstance(e, APIConnectionError)

            if (is_rate_limit or is_connection) and attempt < max_retries - 1:
                print(f"[llm_client] Encountered {type(e).__name__} (status={status_code}). "
                      f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
