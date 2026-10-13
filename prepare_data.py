"""
prepare_data.py: Downloads and samples GSM8K and HumanEval benchmarks (seed 42).
Produces data/math_sample.json and data/code_sample.json for pilot and experiment runs.
"""
import gzip
import io
import json
import os
import random
import re
import urllib.request

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def prepare_math_data(sample_size: int = 5, seed: int = 42):
    """Downloads GSM8K test set, samples problems deterministically, and saves them."""
    print(f"Downloading GSM8K dataset from {GSM8K_URL}...")
    req = urllib.request.Request(GSM8K_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")

    problems = []
    for line in content.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        question = item["question"]
        raw_answer = item["answer"]
        # In GSM8K, the final numeric answer follows '####'
        if "####" in raw_answer:
            ground_truth = raw_answer.split("####")[-1].strip()
            # Clean ground truth number (remove commas if any)
            ground_truth = ground_truth.replace(",", "")
        else:
            match = re.search(r"[-+]?\d+(?:\.\d+)?", raw_answer)
            ground_truth = match.group(0) if match else raw_answer.strip()

        problems.append({
            "question": question,
            "ground_truth": ground_truth,
        })

    # Sample deterministically
    random.seed(seed)
    sampled = random.sample(problems, min(sample_size, len(problems)))
    # Add standardized task_ids
    for idx, item in enumerate(sampled):
        item["id"] = f"gsm8k_{idx + 1}"

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "math_sample.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2)
    print(f"Saved {len(sampled)} math problems to {out_path}")
    return sampled


def prepare_code_data(sample_size: int = 5, seed: int = 42):
    """Downloads HumanEval dataset, samples problems deterministically, and saves them."""
    print(f"Downloading HumanEval dataset from {HUMANEVAL_URL}...")
    req = urllib.request.Request(HUMANEVAL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        buf = io.BytesIO(resp.read())
        with gzip.GzipFile(fileobj=buf) as gz:
            content = gz.read().decode("utf-8")

    problems = []
    for line in content.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        problems.append({
            "id": item["task_id"],
            "question": item["prompt"],
            "entry_point": item["entry_point"],
            "test": item["test"],
        })

    # Sample deterministically
    random.seed(seed)
    sampled = random.sample(problems, min(sample_size, len(problems)))

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "code_sample.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2)
    print(f"Saved {len(sampled)} code problems to {out_path}")
    return sampled


def main():
    prepare_math_data(sample_size=5, seed=42)
    prepare_code_data(sample_size=5, seed=42)
    print("Benchmark preparation complete.")


if __name__ == "__main__":
    main()
