# Adaptive Multi-Agent Reasoning — starter pipeline

Minimal, runnable skeleton covering all three comparison modes from your
paper: single agent, fixed multi-agent, and your adaptive (agreement-based)
system — on a handful of sample math/code problems.

## Setup (do this first)

1. Get a free Groq API key: https://console.groq.com (fast + cheap, good
   for running many test queries without burning a budget)
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and paste your key into `.env`

## Run it

```
python run_pilot.py
```

This solves every sample problem with all three modes, prints a live
accuracy/token summary, and writes `results.csv` — one row per run, in
the same schema we discussed for your evaluation table.

## What to do next, in order

1. Run it as-is first — confirm the pipeline actually works end to end
   before changing anything.
2. Swap `src/dataset.py` for real GSM8K / HumanEval rows (a few dozen of
   each is enough for a real pilot).
3. **Known placeholder to fix before you report results**: math
   verification in `src/verifier.py` currently compares directly against
   the known ground truth to decide which candidate "wins" during
   escalation. That's fine for *scoring* accuracy at the end, but it's
   circular as an actual verifier, since a real system won't know the
   ground truth in advance. Replace it with something that re-derives
   the answer independently — e.g. have a separate agent call re-solve
   just the final arithmetic step, or extract and evaluate the
   expression with Python's `eval`/`sympy` — before using this as your
   "deterministic verification" claim in the paper. Code verification
   is fine as-is, since test cases are a legitimate oracle a real system
   can have access to.
4. Once results look sane on ~20-30 questions, scale up and add logging
   for latency if you want that metric too.
