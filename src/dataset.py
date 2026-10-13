"""
dataset.py: Loads benchmark sample problems for math and code.
Reads from data/math_sample.json (GSM8K) and data/code_sample.json (HumanEval).
Falls back to minimal starter lists if data files have not been prepared yet.
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MATH_SAMPLE_FILE = os.path.join(BASE_DIR, "data", "math_sample.json")
CODE_SAMPLE_FILE = os.path.join(BASE_DIR, "data", "code_sample.json")

# Fallback starter problems if data files are missing
DEFAULT_MATH_PROBLEMS = [
    {"id": "m1", "question": "Natalia sold clips to 48 friends in April, "
     "then half as many in May. How many clips did she sell total?",
     "ground_truth": "72"},
    {"id": "m2", "question": "A train travels 60 miles in 1.5 hours. "
     "At the same speed, how many miles does it travel in 4 hours?",
     "ground_truth": "160"},
    {"id": "m3", "question": "If a shirt costs $25 and is discounted by "
     "20%, what is the new price?",
     "ground_truth": "20"},
]

DEFAULT_CODE_PROBLEMS = [
    {"id": "c1",
     "question": "Write a function `is_even(n)` that returns True if n "
     "is even, False otherwise.",
     "test": "def check(candidate):\n"
             "    assert candidate(4) == True\n"
             "    assert candidate(7) == False\n",
     "entry_point": "is_even"},
    {"id": "c2",
     "question": "Write a function `reverse_string(s)` that returns the "
     "reverse of the input string.",
     "test": "def check(candidate):\n"
             "    assert candidate('abc') == 'cba'\n"
             "    assert candidate('') == ''\n",
     "entry_point": "reverse_string"},
]


def load_math_problems():
    """Loads math sample problems from data/math_sample.json or default."""
    if os.path.exists(MATH_SAMPLE_FILE):
        with open(MATH_SAMPLE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_MATH_PROBLEMS


def load_code_problems():
    """Loads code sample problems from data/code_sample.json or default."""
    if os.path.exists(CODE_SAMPLE_FILE):
        with open(CODE_SAMPLE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CODE_PROBLEMS


MATH_PROBLEMS = load_math_problems()
CODE_PROBLEMS = load_code_problems()
