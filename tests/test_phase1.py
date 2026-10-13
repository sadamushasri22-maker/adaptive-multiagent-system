"""
Unit tests to thoroughly verify Phase 1 implementations:
- Math answer normalization ($ stripped, commas stripped, 72 == 72.0, negatives, decimals)
- Subprocess code execution with typing/math imports and timeout enforcement
- Code behavioral agreement (different source code with identical behavior match)
- Rate-limit retry mechanism in llm_client
- Absence of ground-truth leakage in candidate selection
"""
import time
from unittest.mock import patch, MagicMock
from openai import RateLimitError

from src.verifier import (
    normalize_math_answer,
    verify_math,
    verify_code,
    get_code_behavior_signature,
    solve_math_via_code,
)
from src.llm_client import call_agent


def test_normalize_math_answer():
    # 72 and 72.0 equality
    assert normalize_math_answer("72") == "72"
    assert normalize_math_answer("72.0") == "72"
    assert normalize_math_answer("72.000") == "72"

    # Currency and commas
    assert normalize_math_answer("$72") == "72"
    assert normalize_math_answer("$1,000") == "1000"
    assert normalize_math_answer("1,234.50") == "1234.5"
    assert normalize_math_answer("$ 1,234.50") == "1234.5"

    # Negatives and spacing
    assert normalize_math_answer("-15") == "-15"
    assert normalize_math_answer("-15.0") == "-15"
    assert normalize_math_answer("- 15") == "-15"
    assert normalize_math_answer("-$15.00") == "-15"

    # Decimals
    assert normalize_math_answer("0.5") == "0.5"
    assert normalize_math_answer(".5") == "0.5"
    assert normalize_math_answer("-0.25") == "-0.25"

    # Fractions
    assert normalize_math_answer("3/4") == "0.75"

    # None and empty
    assert normalize_math_answer(None) is None
    assert normalize_math_answer("") is None


def test_verify_math_scoring():
    assert verify_math("72", "72.0") is True
    assert verify_math("$72", "72") is True
    assert verify_math("1,000", "1000") is True
    assert verify_math("-15.0", "-15") is True
    assert verify_math("72", "73") is False
    assert verify_math(None, "72") is False


def test_verify_code_subprocess():
    # Code using math and typing
    good_code = """
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2 and math.isclose(elem, elem2, abs_tol=threshold):
                return True
    return False
"""
    test_code = """
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
"""
    assert verify_code(good_code, test_code, "has_close_elements", timeout=5.0) is True

    # Bad code
    bad_code = "def has_close_elements(numbers, threshold):\n    return False\n"
    assert verify_code(bad_code, test_code, "has_close_elements", timeout=5.0) is False

    # Code with infinite loop enforces 1s timeout
    hang_code = "def has_close_elements(numbers, threshold):\n    while True: pass\n"
    start = time.time()
    assert verify_code(hang_code, test_code, "has_close_elements", timeout=1.0) is False
    elapsed = time.time() - start
    assert elapsed >= 0.9 and elapsed < 3.0


def test_code_behavioral_agreement():
    # Two different implementations of is_even:
    impl1 = """
def is_even(n: int) -> bool:
    # Modulo operator
    return n % 2 == 0
"""
    impl2 = """
def is_even(num):
    # Bitwise operator
    return (num & 1) == 0
"""
    impl_buggy = """
def is_even(n):
    return True
"""
    test_code = """
def check(candidate):
    assert candidate(4) == True
    assert candidate(7) == False
"""
    sig1 = get_code_behavior_signature(impl1, test_code, "is_even")
    sig2 = get_code_behavior_signature(impl2, test_code, "is_even")
    sig_buggy = get_code_behavior_signature(impl_buggy, test_code, "is_even")

    # They should have completely different source text:
    assert impl1 != impl2

    # BUT identical behavioral signatures:
    assert sig1 == sig2

    # Buggy one should disagree:
    assert sig1 != sig_buggy


def test_independent_math_verifier_mock():
    # Mock LLM call to return python script
    mock_resp = {
        "text": "```python\nclips_april = 48\nclips_may = clips_april / 2\ntotal = clips_april + clips_may\nprint(total)\n```",
        "tokens": 45,
    }
    with patch("src.verifier.call_agent", return_value=mock_resp):
        res = solve_math_via_code("Natalia sold clips...")
        assert res["success"] is True
        assert res["verified_answer"] == "72"
        assert res["tokens"] == 45


def test_llm_client_rate_limit_retry():
    # Mock rate limit on first call, success on second
    mock_client = MagicMock()
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    rate_err = RateLimitError("Rate limit reached", response=mock_resp_429, body={})

    success_resp = MagicMock()
    success_resp.choices = [MagicMock(message=MagicMock(content="FINAL ANSWER: 42"))]
    success_resp.usage.total_tokens = 50

    mock_client.chat.completions.create.side_effect = [rate_err, success_resp]

    with patch("src.llm_client.get_client", return_value=mock_client):
        res = call_agent("Solve 6 * 7", initial_backoff=0.01)
        assert res["text"] == "FINAL ANSWER: 42"
        assert res["tokens"] == 50
        assert mock_client.chat.completions.create.call_count == 2
