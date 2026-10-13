"""
src/agents.py: Solves math and code problems with low or high effort prompts.
Used as the fundamental building block across all pipeline modes.
"""
import re
from src.llm_client import call_agent
from src.verifier import normalize_math_answer

MATH_PROMPT = """Solve this math problem step by step, then give the final
numeric answer on its own line in the exact form:
FINAL ANSWER: <number>

Problem: {question}
"""

MATH_PROMPT_LOW = """Answer this math problem directly with minimal working.
End with the exact line:
FINAL ANSWER: <number>

Problem: {question}
"""

CODE_PROMPT_LOW = """Write a short Python function for this problem.
Return ONLY the function in a ```python code block, no comments.

Problem: {question}
"""

CODE_PROMPT = """Write a Python function that solves this problem.
Return ONLY the function code in a ```python code block, nothing else.

Problem: {question}
"""


def solve_math(question: str, effort: str = "high") -> dict:
    """
    Prompts the LLM to solve a math problem and extracts/normalizes the numeric answer.
    Handles currency symbols, commas, decimals, and negative signs via normalize_math_answer.
    """
    template = MATH_PROMPT if effort == "high" else MATH_PROMPT_LOW
    prompt = template.format(question=question)
    result = call_agent(prompt)

    # Search for "FINAL ANSWER: <value>"
    match = re.search(r"FINAL ANSWER:\s*([^\n\r]+)", result["text"], re.IGNORECASE)
    raw_answer = match.group(1).strip() if match else None

    # If no explicit marker was found, search for the last numeric token as fallback
    if raw_answer is None:
        numbers = re.findall(r"[-+]?[$]?[\d,]+(?:\.\d+)?", result["text"])
        raw_answer = numbers[-1] if numbers else None

    result["extracted_answer"] = normalize_math_answer(raw_answer)
    return result


def solve_code(question: str, effort: str = "high") -> dict:
    """
    Prompts the LLM to write a Python function and extracts the code block.
    """
    template = CODE_PROMPT if effort == "high" else CODE_PROMPT_LOW
    prompt = template.format(question=question)
    result = call_agent(prompt)

    # Extract Python code from markdown code fences
    match = re.search(r"```python\s*\n(.*?)```", result["text"], re.DOTALL)
    if not match:
        match = re.search(r"```\s*\n(.*?)```", result["text"], re.DOTALL)
    result["extracted_code"] = match.group(1) if match else result["text"]
    return result


def solve(question: str, domain: str, effort: str = "high") -> dict:
    """Dispatches to the domain-specific solver."""
    if domain == "math":
        return solve_math(question, effort)
    elif domain == "code":
        return solve_code(question, effort)
    raise ValueError(f"Unknown domain: {domain}")
