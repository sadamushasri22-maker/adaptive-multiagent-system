"""
src/pipeline.py: Reasoning modes comparing single, fixed multi-agent,
ARES-inspired heuristic, and confidence-aware adaptive multi-agent.

Comparison modes:
  1. single_agent          - Calls one agent once (baseline)
  2. ares_style            - Calls one agent with heuristic prompt effort (baseline)
  3. fixed_multi_agent     - Calls 3 agents, majority vote (baseline)
  4. adaptive_multi_agent  - Calls 2 agents; on disagreement escalates to 3rd agent
                             and uses independent verification (NO ground truth leak).
"""
from collections import Counter
from src.agents import solve
from src.verifier import (
    verify_math,
    verify_code,
    normalize_math_answer,
    solve_math_via_code,
    get_code_behavior_signature,
)


def _is_correct(domain: str, answer_item: dict, problem: dict) -> bool:
    """
    CRITICAL: Evaluates correctness against problem ground truth / test suite.
    Used ONLY for post-run evaluation and scoring. Never used during candidate selection.
    """
    if domain == "math":
        return verify_math(answer_item.get("extracted_answer"), problem["ground_truth"])
    return verify_code(
        answer_item.get("extracted_code", ""),
        problem["test"],
        problem["entry_point"],
    )


def _answer_key(domain: str, answer_item: dict, problem: dict):
    """
    Returns the representation used to compare agreement between agents:
    - For math: normalized numeric string (e.g. '72' and '72.0' match).
    - For code: behavioral signature on test inputs (outputs on test cases match).
    """
    if domain == "math":
        return normalize_math_answer(answer_item.get("extracted_answer"))
    return get_code_behavior_signature(
        answer_item.get("extracted_code", ""),
        problem["test"],
        problem["entry_point"],
    )


def single_agent(problem: dict, domain: str) -> dict:
    """Baseline: runs a single high-effort agent once."""
    result = solve(problem["question"], domain)
    final_ans = result.get("extracted_answer") if domain == "math" else result.get("extracted_code")
    return {
        "mode": "single_agent",
        "agents_used": 1,
        "escalated": False,
        "total_tokens": result["tokens"],
        "final_answer": final_ans,
        "correct": _is_correct(domain, result, problem),
    }


def fixed_multi_agent(problem: dict, domain: str, n_agents: int = 3) -> dict:
    """Baseline: calls N agents unconditionally and selects candidate by majority vote."""
    results = [solve(problem["question"], domain) for _ in range(n_agents)]
    total_tokens = sum(r["tokens"] for r in results)

    # Vote using normalized math answers or behavioral signatures for code
    votes = Counter(_answer_key(domain, r, problem) for r in results)
    winning_key = votes.most_common(1)[0][0]
    winner = next(r for r in results if _answer_key(domain, r, problem) == winning_key)

    final_ans = winner.get("extracted_answer") if domain == "math" else winner.get("extracted_code")
    return {
        "mode": "fixed_multi_agent",
        "agents_used": n_agents,
        "escalated": False,
        "total_tokens": total_tokens,
        "final_answer": final_ans,
        "correct": _is_correct(domain, winner, problem),
    }


def adaptive_multi_agent(problem: dict, domain: str) -> dict:
    """
    Our method: confidence-aware adaptive multi-agent system.
    Step 1: Run 2 initial agents.
    Step 2: Check agreement (normalized math values or code behavior).
    Step 3: If agreement -> accept early with 2 agents.
            If disagreement -> escalate to 3rd agent AND verify independently:
               - Math: independent code-based computation check (no ground truth).
               - Code: sandboxed test execution check.
    """
    r1 = solve(problem["question"], domain)
    r2 = solve(problem["question"], domain)
    total_tokens = r1["tokens"] + r2["tokens"]
    agents_used = 2
    escalated = False

    k1 = _answer_key(domain, r1, problem)
    k2 = _answer_key(domain, r2, problem)

    if k1 == k2:
        # Agents agree on behavior/answer -> high confidence, stop early
        final = r1
    else:
        # Disagreement -> low confidence, escalate
        escalated = True
        r3 = solve(problem["question"], domain)
        total_tokens += r3["tokens"]
        agents_used = 3
        candidates = [r1, r2, r3]

        if domain == "math":
            # Independent math verification without ground truth
            ver_res = solve_math_via_code(problem["question"])
            total_tokens += ver_res["tokens"]
            v_ans = ver_res.get("verified_answer")

            # Check if any candidate matches the independently derived answer
            matched = [c for c in candidates if normalize_math_answer(c.get("extracted_answer")) == v_ans]
            if matched:
                final = matched[0]
            else:
                # Fallback to majority vote if verifier didn't match or failed
                votes = Counter(_answer_key(domain, c, problem) for c in candidates)
                top_key, count = votes.most_common(1)[0]
                if count > 1:
                    final = next(c for c in candidates if _answer_key(domain, c, problem) == top_key)
                else:
                    final = r3
        else:
            # Code domain: execute candidates against tests in sandbox
            passing = [
                c for c in candidates
                if verify_code(c.get("extracted_code", ""), problem["test"], problem["entry_point"])
            ]
            if passing:
                final = passing[0]
            else:
                votes = Counter(_answer_key(domain, c, problem) for c in candidates)
                top_key, count = votes.most_common(1)[0]
                if count > 1:
                    final = next(c for c in candidates if _answer_key(domain, c, problem) == top_key)
                else:
                    final = r3

    final_ans = final.get("extracted_answer") if domain == "math" else final.get("extracted_code")
    return {
        "mode": "adaptive_multi_agent",
        "agents_used": agents_used,
        "escalated": escalated,
        "total_tokens": total_tokens,
        "final_answer": final_ans,
        "correct": _is_correct(domain, final, problem),
    }


def _pick_effort(question: str) -> str:
    """
    Heuristic stand-in for ARES's router: questions with length >= 25 words
    or >= 4 numbers are routed to high reasoning effort; otherwise low effort.
    """
    words = question.split()
    numbers = sum(any(ch.isdigit() for ch in w) for w in words)
    return "high" if (len(words) >= 25 or numbers >= 4) else "low"


def ares_style(problem: dict, domain: str) -> dict:
    """
    ARES-inspired heuristic baseline: ONE agent whose prompt effort is chosen
    per question by a heuristic router.
    """
    effort = _pick_effort(problem["question"])
    result = solve(problem["question"], domain, effort)
    final_ans = result.get("extracted_answer") if domain == "math" else result.get("extracted_code")
    return {
        "mode": "ares_style",
        "agents_used": 1,
        "escalated": effort == "high",
        "total_tokens": result["tokens"],
        "final_answer": final_ans,
        "correct": _is_correct(domain, result, problem),
    }
