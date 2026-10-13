"""
run_pilot.py: Executes all evaluation modes on sample benchmark problems.
Writes results to results.csv and prints an accuracy and token summary.

Usage: python run_pilot.py
"""
import csv
from src.dataset import MATH_PROBLEMS, CODE_PROBLEMS
from src.pipeline import (
    single_agent,
    fixed_multi_agent,
    adaptive_multi_agent,
    ares_style,
)

MODES = {
    "single_agent": single_agent,
    "ares_style": ares_style,
    "fixed_multi_agent": fixed_multi_agent,
    "adaptive_multi_agent": adaptive_multi_agent,
}


def run_all():
    rows = []
    # Test 5 problems per domain (or available in sample)
    math_set = MATH_PROBLEMS[:5]
    code_set = CODE_PROBLEMS[:5]
    problem_sets = [("math", math_set), ("code", code_set)]

    for domain, problems in problem_sets:
        for problem in problems:
            for mode_name, mode_fn in MODES.items():
                print(f"Running {mode_name} on {problem['id']} ({domain})...")
                result = mode_fn(problem, domain)
                result["task_id"] = problem["id"]
                result["domain"] = domain
                rows.append(result)

    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "task_id",
            "domain",
            "mode",
            "agents_used",
            "escalated",
            "total_tokens",
            "correct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone. Results written to results.csv")
    print_summary(rows)


def print_summary(rows):
    from collections import defaultdict

    stats = defaultdict(lambda: {"n": 0, "correct": 0, "tokens": 0})
    for r in rows:
        s = stats[r["mode"]]
        s["n"] += 1
        s["correct"] += int(r["correct"])
        s["tokens"] += r["total_tokens"]

    print("\n--- Summary ---")
    for mode, s in stats.items():
        acc = 100 * s["correct"] / s["n"] if s["n"] > 0 else 0
        avg_tokens = s["tokens"] / s["n"] if s["n"] > 0 else 0
        print(f"{mode:22s}  accuracy={acc:5.1f}%  avg_tokens={avg_tokens:6.1f}")


if __name__ == "__main__":
    run_all()
