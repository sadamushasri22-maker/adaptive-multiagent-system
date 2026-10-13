"""
src/verifier.py: Verifiers and normalization for math and code.
Designed for viva presentation with clear step-by-step logic.

Key components:
1. normalize_math_answer: Shared normalization function for numbers (commas, $, decimals, negatives).
2. verify_math: Ground-truth comparison used ONLY for final scoring at the end.
3. solve_math_via_code: Independent LLM-generated Python calculator check without ground truth.
4. verify_code: Sandboxed code execution in a subprocess with a 5s timeout and typing/math imports.
5. get_code_behavior_signature: Evaluates candidate code outputs on test inputs to compare behavior.
"""
import ast
import json
import re
import subprocess
import sys
from typing import Optional, Tuple, List, Any
from src.llm_client import call_agent

# Prompt for the independent math verifier
INDEPENDENT_MATH_PROMPT = """Write a short, self-contained Python script to compute the exact numeric answer to this math problem.
The script must print ONLY the final numeric answer to stdout using print() and nothing else.
Return ONLY the executable Python code inside a ```python ``` code block.

Problem: {question}
"""


def normalize_math_answer(raw: Optional[str]) -> Optional[str]:
    """
    Normalizes a numeric math answer to a canonical string representation:
    - Strips whitespace, dollar signs ($), commas (,), and percentage signs (%)
    - Cleans spaces in negative numbers (e.g. '- 15' -> '-15')
    - Treats '72' and '72.0' as equal by numeric float/int conversion
    - Preserves decimal precision when non-zero (e.g. '72.5')
    - Handles simple fractions (e.g. '3/4' -> '0.75')
    Returns canonical string or None if unparseable.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    # Strip currency symbols, commas, and percentage signs
    text = text.replace("$", "").replace(",", "").replace("%", "").strip()

    # Strip trailing periods (e.g. "72.")
    if text.endswith("."):
        text = text[:-1].strip()

    # Handle negative prefix spacing like "- 72" -> "-72"
    text = re.sub(r"^-\s+", "-", text)

    # Try numeric conversion
    try:
        val = float(text)
        if val.is_integer():
            return str(int(val))
        # Format decimal without trailing zero clutter
        return f"{val:.8f}".rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        # Check if answer is a fraction like "3/4"
        if "/" in text:
            parts = text.split("/")
            if len(parts) == 2:
                try:
                    num = float(parts[0].strip())
                    den = float(parts[1].strip())
                    if den != 0:
                        val = num / den
                        if val.is_integer():
                            return str(int(val))
                        return f"{val:.8f}".rstrip("0").rstrip(".")
                except ValueError:
                    pass
        return text.strip().lower()


def verify_math(extracted_answer: Optional[str], ground_truth: str) -> bool:
    """
    Compares extracted math answer with ground truth using normalized values.
    CRITICAL: This function is used ONLY for final scoring of results, never
    during candidate selection in pipeline runs.
    """
    norm_ext = normalize_math_answer(extracted_answer)
    norm_gt = normalize_math_answer(ground_truth)
    if norm_ext is None or norm_gt is None:
        return False
    try:
        return abs(float(norm_ext) - float(norm_gt)) < 1e-6
    except ValueError:
        return norm_ext == norm_gt


def solve_math_via_code(question: str) -> dict:
    """
    Independent math check: asks LLM to write a Python snippet computing the answer,
    runs it in an isolated subprocess with a 5 second timeout, and extracts the answer.
    NO ground truth is used.
    Returns:
        dict with:
            'verified_answer': normalized answer string or None
            'tokens': token cost of the LLM call
            'success': True if execution succeeded and produced a number
    """
    prompt = INDEPENDENT_MATH_PROMPT.format(question=question)
    response = call_agent(prompt)
    raw_text = response.get("text", "")
    tokens = response.get("tokens", 0)

    # Extract code from markdown block
    match = re.search(r"```(?:python)?\n(.*?)```", raw_text, re.DOTALL)
    code = match.group(1) if match else raw_text

    # Prepend required imports
    wrapper_code = f"from typing import *\nimport math\n\n{code}\n"

    try:
        proc = subprocess.run(
            [sys.executable],
            input=wrapper_code,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if proc.returncode == 0:
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if lines:
                # Take the last printed line as the answer
                last_line = lines[-1]
                num_match = re.search(r"[-+]?[\d,]+(?:\.\d+)?", last_line)
                raw_ans = num_match.group(0) if num_match else last_line
                norm = normalize_math_answer(raw_ans)
                return {"verified_answer": norm, "tokens": tokens, "success": True}
    except (subprocess.TimeoutExpired, Exception):
        pass

    return {"verified_answer": None, "tokens": tokens, "success": False}


def verify_code(code: str, test_code: str, entry_point: str, timeout: float = 5.0) -> bool:
    """
    Runs generated code against test assertions in an isolated subprocess with a 5s timeout.
    Executes 'from typing import *' and 'import math' first.
    Returns True if tests pass (returncode 0), False otherwise.
    """
    full_script = f"""from typing import *
import math
import sys

# Candidate code
{code}

# Test code
{test_code}

try:
    check({entry_point})
except Exception:
    sys.exit(1)
"""
    try:
        proc = subprocess.run(
            [sys.executable],
            input=full_script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def extract_test_calls(test_code: str, entry_point: str) -> List[str]:
    """
    Extracts test call expressions from test code using Python's AST parser.
    Looks for calls to 'candidate(...)' or '{entry_point}(...)'.
    """
    calls = []
    try:
        tree = ast.parse(test_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None)
                if name in ("candidate", entry_point):
                    calls.append(ast.unparse(node))
    except Exception:
        pass
    return calls


def get_code_behavior_signature(code: str, test_code: str, entry_point: str, timeout: float = 5.0) -> tuple:
    """
    Executes candidate code on test inputs in an isolated subprocess.
    Returns a tuple signature of output representations for behavioral comparison.
    Two candidates with identical signatures produce identical outputs on the same test inputs.
    """
    test_calls = extract_test_calls(test_code, entry_point)
    test_calls_json = json.dumps(test_calls)

    # Indent candidate code for safe execution inside runner
    indented_code = "\n".join("    " + line for line in code.splitlines())
    escaped_test_code = json.dumps(test_code)

    runner_script = f"""from typing import *
import math
import sys
import json

try:
{indented_code}
except Exception as e:
    print(json.dumps([("syntax_error", type(e).__name__)]))
    sys.exit(0)

# Identify target function
try:
    candidate = {entry_point}
except Exception as e:
    print(json.dumps([("missing_entry_point", type(e).__name__)]))
    sys.exit(0)

test_calls = {test_calls_json}
results = []
if test_calls:
    for expr in test_calls:
        try:
            val = eval(expr)
            results.append(("ok", repr(val)))
        except Exception as e:
            results.append(("err", type(e).__name__))
else:
    # If no AST calls were found, run test_code's check(candidate)
    try:
        ns = {{"candidate": candidate, "{entry_point}": candidate, "math": math}}
        exec({escaped_test_code}, ns)
        if "check" in ns:
            ns["check"](candidate)
            results.append(("check", "passed"))
        else:
            results.append(("check", "no_check_fn"))
    except Exception as e:
        results.append(("check", type(e).__name__))

print(json.dumps(results))
"""
    try:
        proc = subprocess.run(
            [sys.executable],
            input=runner_script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout.strip())
            return tuple(tuple(item) for item in data)
    except (subprocess.TimeoutExpired, Exception):
        return (("timeout", "expired"),)

    return (("error", "execution_failed"),)
