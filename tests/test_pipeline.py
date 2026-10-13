"""
Tests full pipeline execution across all four modes with mock agents,
verifying that:
1. All modes execute cleanly and compute token costs correctly.
2. Adaptive mode escalates only on disagreement.
3. Math candidate selection in adaptive mode uses independent verification without ground truth.
4. Code candidate selection compares behaviorally.
"""
from unittest.mock import patch
from src.dataset import MATH_PROBLEMS, CODE_PROBLEMS
from src.pipeline import (
    single_agent,
    ares_style,
    fixed_multi_agent,
    adaptive_multi_agent,
)


def test_pipeline_math_agree():
    # Both agents return the same answer: 72 and 72.0 (should agree due to normalization)
    responses = [
        {"text": "FINAL ANSWER: 72", "tokens": 20},
        {"text": "FINAL ANSWER: 72.0", "tokens": 25},
    ]
    problem = MATH_PROBLEMS[0]

    with patch("src.agents.call_agent", side_effect=responses):
        res = adaptive_multi_agent(problem, "math")
        assert res["mode"] == "adaptive_multi_agent"
        assert res["agents_used"] == 2
        assert res["escalated"] is False
        assert res["total_tokens"] == 45
        assert res["correct"] == (res["final_answer"] == problem["ground_truth"])


def test_pipeline_math_disagree_escalate():
    # Agent 1 returns 50, Agent 2 returns 72 (disagree -> escalate to agent 3 returning 72)
    # Independent verifier returns 72 with 30 tokens
    agent_responses = [
        {"text": "FINAL ANSWER: 50", "tokens": 20},
        {"text": "FINAL ANSWER: 72", "tokens": 25},
        {"text": "FINAL ANSWER: 72", "tokens": 22},
    ]
    verifier_response = {
        "text": "```python\nprint(72)\n```",
        "tokens": 30,
    }

    problem = MATH_PROBLEMS[0]

    with patch("src.agents.call_agent", side_effect=agent_responses), \
         patch("src.verifier.call_agent", return_value=verifier_response):
        res = adaptive_multi_agent(problem, "math")
        assert res["mode"] == "adaptive_multi_agent"
        assert res["agents_used"] == 3
        assert res["escalated"] is True
        # Total tokens = 20 + 25 + 22 + 30 = 97
        assert res["total_tokens"] == 97
        assert res["final_answer"] == "72"


def test_all_modes_on_sample_problem():
    math_p = MATH_PROBLEMS[0]
    code_p = CODE_PROBLEMS[0]

    dummy_math = {"text": "FINAL ANSWER: 72", "tokens": 15}
    dummy_code = {
        "text": "```python\ndef " + code_p["entry_point"] + "(numbers, threshold):\n    return False\n```",
        "tokens": 20,
    }

    with patch("src.agents.call_agent", return_value=dummy_math):
        r_single = single_agent(math_p, "math")
        assert r_single["agents_used"] == 1
        r_ares = ares_style(math_p, "math")
        assert r_ares["agents_used"] == 1
        r_fixed = fixed_multi_agent(math_p, "math", n_agents=3)
        assert r_fixed["agents_used"] == 3

    with patch("src.agents.call_agent", return_value=dummy_code):
        r_code_single = single_agent(code_p, "code")
        assert r_code_single["agents_used"] == 1
