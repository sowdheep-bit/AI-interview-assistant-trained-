"""
generate_grading_dataset.py
============================
Builds the actual training set for the grading task: for a stratified,
deduplicated sample of questions from data/LLM_Interview_Dataset-v2.csv,
generates a Perfect / Partial / Wrong candidate answer + numeric score for
each, using Gemini. The 30 GOLD_STANDARD questions from evaluate_model.py
are excluded so the held-out eval stays clean.

Requires network access + GEMINI_API_KEY. Run this locally / wherever that
key works — it will NOT run in a sandboxed, network-disabled environment.

Usage:
    python scripts/generate_grading_dataset.py --n-per-cell 60
    python scripts/generate_grading_dataset.py --resume   # continue after interruption
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "LLM_Interview_Dataset-v2.csv"
OUT_JSONL = ROOT / "data" / "grading_train.jsonl"
EVAL_SCRIPT = ROOT / "scripts" / "evaluate_model.py"

RANDOM_SEED = 42

# Same prefix-normalization regex evaluate_model.py uses, so our dedup logic
# matches theirs exactly.
PREFIX_RE = re.compile(
    r"^(Candidate prompt|Could you explain this|Define and explain"
    r"|Interview Question|Please answer the following)\s*:\s*",
    re.IGNORECASE,
)


def norm_q(q: str) -> str:
    return PREFIX_RE.sub("", str(q).strip()).strip().lower()


def load_gold_standard_questions() -> set:
    """Extract the 30 GOLD_STANDARD questions straight out of evaluate_model.py
    so this stays in sync even if that list changes later."""
    src = EVAL_SCRIPT.read_text()
    m = re.search(r"GOLD_STANDARD\s*=\s*\[(.*?)\n\]\n", src, re.DOTALL)
    if not m:
        raise RuntimeError("Could not find GOLD_STANDARD list in evaluate_model.py")
    block = m.group(1)
    # Each entry starts with ("question text",  -- grab the first string literal
    # on each tuple opening line.
    qs = re.findall(r'\(\s*"((?:[^"\\]|\\.)*)"\s*,', block)
    if len(qs) < 25:
        raise RuntimeError(f"Only parsed {len(qs)} gold questions, expected ~30 — check regex")
    return {norm_q(q) for q in qs}


def stratified_sample(rows, n_per_cell, excluded_norm_qs):
    """Dedup by normalized question text, then sample up to n_per_cell
    unique questions per (Domain, Difficulty) cell."""
    seen = {}
    for r in rows:
        key = norm_q(r["Question"])
        if key in excluded_norm_qs:
            continue
        cell = (r["Domain"], r["Difficulty_Level"])
        seen.setdefault(cell, {})
        # keep first occurrence of each unique question in this cell
        seen[cell].setdefault(key, r)

    rng = random.Random(RANDOM_SEED)
    sampled = []
    for cell, qmap in seen.items():
        pool = list(qmap.values())
        rng.shuffle(pool)
        sampled.extend(pool[:n_per_cell])
    return sampled


GEMINI_PROMPT = """You are helping build training data for an automated interview-answer grader.

Given an interview question and its ideal reference answer, generate THREE candidate answers a real human interviewee might give, at three distinct quality tiers, plus a fair score (0-10) for each under this rubric:
- 0 marks: gibberish, completely wrong, or empty.
- 1 to 4 marks: vague, lacking detail, or mostly incorrect.
- 5 to 7 marks: partially correct but missing key concepts.
- 8 to 10 marks: conceptually accurate and complete.

Question: {question}
Ideal Answer: {reference}

Rules:
- "perfect" tier: conceptually complete and accurate, but phrased DIFFERENTLY from the ideal answer (not a copy). Score 8-10.
- "partial" tier: gets the general idea but is vague, incomplete, or missing a key concept. Score 4-7.
- "wrong" tier: a plausible-sounding but substantively incorrect or badly confused answer (NOT gibberish, NOT empty, NOT "I don't know" — a real wrong attempt). Score 0-3.
- Vary phrasing, length, and style across items so the dataset isn't formulaic.
- Scores must be integers.

Respond with ONLY valid JSON, no markdown fences, no preamble, in exactly this shape:
{{"perfect": {{"answer": "...", "score": 9}}, "partial": {{"answer": "...", "score": 5}}, "wrong": {{"answer": "...", "score": 2}}}}
"""


def call_gemini(question, reference, api_key, model="gemini-2.0-flash", max_retries=5):
    """Minimal REST call to Gemini generateContent. Kept dependency-free
    (just `requests`) so this doesn't force a particular SDK version on you."""
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": GEMINI_PROMPT.format(question=question, reference=reference)}]}],
        "generationConfig": {"temperature": 0.9, "response_mime_type": "application/json"},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 429:
                wait = min(60, 2 ** attempt * 2)
                print(f"    rate limited, sleeping {wait}s ...", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            text = re.sub(r"^```json\s*|\s*```$", "", text.strip())
            parsed = json.loads(text)
            for tier in ("perfect", "partial", "wrong"):
                if tier not in parsed or "answer" not in parsed[tier] or "score" not in parsed[tier]:
                    raise ValueError(f"malformed response, missing {tier}")
            return parsed
        except Exception as e:
            wait = min(30, 2 ** attempt)
            print(f"    attempt {attempt+1} failed ({e}), retrying in {wait}s ...", file=sys.stderr)
            time.sleep(wait)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=100,
                     help="max unique questions sampled per Domain x Difficulty cell")
    ap.add_argument("--model", default="gemini-2.0-flash")
    ap.add_argument("--resume", action="store_true",
                     help="skip questions already present in the output file")
    ap.add_argument("--sleep", type=float, default=0.5,
                     help="seconds to sleep between calls (basic rate limiting)")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set in environment. Export it and re-run.")

    rows = list(csv.DictReader(open(DATA_CSV, encoding="utf-8")))
    gold_qs = load_gold_standard_questions()
    print(f"Loaded {len(rows)} rows, excluding {len(gold_qs)} gold-standard questions.")

    sample = stratified_sample(rows, args.n_per_cell, gold_qs)
    print(f"Sampled {len(sample)} unique questions across cells "
          f"(target up to {args.n_per_cell}/cell).")

    already_done = set()
    mode = "a"
    if args.resume and OUT_JSONL.exists():
        for line in open(OUT_JSONL, encoding="utf-8"):
            try:
                already_done.add(norm_q(json.loads(line)["question"]))
            except Exception:
                continue
        print(f"Resuming: {len(already_done)} questions already generated.")
    else:
        mode = "w"

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_failed = 0
    with open(OUT_JSONL, mode, encoding="utf-8") as out:
        for i, row in enumerate(sample):
            q, ref = row["Question"], row["Reference_Answer"]
            if norm_q(q) in already_done:
                continue

            result = call_gemini(q, ref, api_key, model=args.model)
            if result is None:
                n_failed += 1
                print(f"  [{i+1}/{len(sample)}] FAILED, skipping: {q[:60]}...")
                continue

            for tier_name, tier_label in (("perfect", "P"), ("partial", "M"), ("wrong", "W")):
                item = result[tier_name]
                score = int(item["score"])
                score = max(0, min(10, score))
                record = {
                    "question": q,
                    "reference": ref,
                    "candidate": item["answer"],
                    "score": score,
                    "tier": tier_label,
                    "domain": row["Domain"],
                    "difficulty": row["Difficulty_Level"],
                }
                out.write(json.dumps(record) + "\n")
                n_written += 1
            out.flush()

            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(sample)}] {n_written} examples written, {n_failed} q's failed")
            time.sleep(args.sleep)

    print(f"\nDone. {n_written} training examples written to {OUT_JSONL} "
          f"({n_failed} questions failed after retries).")


if __name__ == "__main__":
    main()
