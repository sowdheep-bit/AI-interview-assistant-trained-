"""
agent_batch_processor.py
========================
Helper utility for processing and validating agent-generated answer variation batches.
Appends validated batches to scratch/augmented_5x_cache.jsonl and updates datasets.
"""

import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "LLM_Interview_Dataset-v2.csv"
CACHE_JSONL = ROOT / "scratch" / "augmented_5x_cache.jsonl"
OUT_AUG_CSV = ROOT / "data" / "LLM_Interview_Dataset-Augmented-5x.csv"
OUT_UNIQUE_CSV = ROOT / "data" / "LLM_Interview_Dataset-Unique-5x.csv"
REPORT_TXT = ROOT / "reports" / "augmentation_5x_validation.txt"
EXPECTED_SHA256 = "3b465f782b30f036c83dfc978dfc9bf53480826444f39bf9730d7a582ae72fcf"
REQUIRED_COLUMNS = ["ID", "Domain", "Difficulty_Level", "Question", "Reference_Answer"]
NUM_VARIATIONS = 5


def calculate_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compute_jaccard_similarity(str1: str, str2: str) -> float:
    tokens1 = set(re.findall(r"\b\w+\b", str1.lower()))
    tokens2 = set(re.findall(r"\b\w+\b", str2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if union else 0.0


def check_diversity(answers: List[str], threshold: float = 0.82) -> Tuple[bool, str]:
    if len(answers) != NUM_VARIATIONS:
        return False, f"Expected {NUM_VARIATIONS} answers, got {len(answers)}"

    for i, ans in enumerate(answers):
        if not ans or not isinstance(ans, str) or len(ans.strip()) < 10:
            return False, f"Answer {i+1} is empty or too short"
        if re.match(r"^(the\s+answer\s+is|the\s+correct\s+answer\s+is|x\s+is\s+the\s+correct\s+answer)\b", ans.strip(), re.I):
            return False, f"Answer {i+1} uses forbidden formulaic opening"

    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            sim = compute_jaccard_similarity(answers[i], answers[j])
            if sim > threshold:
                return False, f"Answers {i+1} and {j+1} are too similar (Jaccard={sim:.2f} > {threshold})"

    return True, "OK"


def get_pending_questions(batch_size: int = 50) -> List[Dict]:
    rows = list(csv.DictReader(open(INPUT_CSV, encoding="utf-8")))
    unique_qs = []
    seen = set()
    for r in rows:
        q = r["Question"].strip()
        if q not in seen:
            seen.add(q)
            unique_qs.append(r)

    cached_qs = set()
    if CACHE_JSONL.exists():
        with open(CACHE_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cached_qs.add(json.loads(line)["question"].strip())

    pending = [r for r in unique_qs if r["Question"].strip() not in cached_qs]
    return pending[:batch_size]


def append_batch(batch_data: List[Dict]) -> Tuple[int, int, List[str]]:
    CACHE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    existing_cached_qs = set()
    if CACHE_JSONL.exists():
        with open(CACHE_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_cached_qs.add(json.loads(line)["question"].strip())

    valid_count = 0
    errors = []
    to_write = []

    for idx, item in enumerate(batch_data):
        q = item.get("question", "").strip()
        answers = item.get("answers", [])
        if not q:
            errors.append(f"Item {idx}: missing question")
            continue
        if q in existing_cached_qs:
            continue

        is_valid, reason = check_diversity(answers)
        if not is_valid:
            errors.append(f"Question '{q[:40]}...': {reason}")
            continue

        to_write.append({"question": q, "answers": answers})
        existing_cached_qs.add(q)
        valid_count += 1

    if to_write:
        with open(CACHE_JSONL, "a", encoding="utf-8") as f:
            for item in to_write:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()

    return valid_count, len(errors), errors


def rebuild_datasets():
    sha_start = calculate_sha256(INPUT_CSV)
    if sha_start != EXPECTED_SHA256:
        print(f"CRITICAL WARNING: Input SHA256 mismatch! {sha_start}")

    rows = list(csv.DictReader(open(INPUT_CSV, encoding="utf-8")))
    unique_q_map = {}
    for idx, r in enumerate(rows):
        q = r["Question"].strip()
        if q not in unique_q_map:
            unique_q_map[q] = {
                "question": q,
                "domain": r["Domain"].strip(),
                "difficulty": r["Difficulty_Level"].strip(),
                "reference": r["Reference_Answer"].strip(),
                "original_id": r["ID"],
            }

    cached_answers = {}
    if CACHE_JSONL.exists():
        with open(CACHE_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cached_answers[rec["question"].strip()] = rec["answers"]

    OUT_AUG_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_UNIQUE_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Write unique CSV
    unique_count = 0
    with open(OUT_UNIQUE_CSV, "w", encoding="utf-8", newline="") as f_unq:
        writer_unq = csv.DictWriter(f_unq, fieldnames=REQUIRED_COLUMNS)
        writer_unq.writeheader()
        for q, info in unique_q_map.items():
            if q in cached_answers and len(cached_answers[q]) == NUM_VARIATIONS:
                for ans in cached_answers[q]:
                    writer_unq.writerow({
                        "ID": info["original_id"],
                        "Domain": info["domain"],
                        "Difficulty_Level": info["difficulty"],
                        "Question": q,
                        "Reference_Answer": ans,
                    })
                    unique_count += 1

    # Write augmented CSV (preserving original order)
    aug_count = 0
    blank_count = 0
    with open(OUT_AUG_CSV, "w", encoding="utf-8", newline="") as f_aug:
        writer_aug = csv.DictWriter(f_aug, fieldnames=REQUIRED_COLUMNS)
        writer_aug.writeheader()
        for r in rows:
            q = r["Question"].strip()
            if q in cached_answers and len(cached_answers[q]) == NUM_VARIATIONS:
                for ans in cached_answers[q]:
                    row_dict = {
                        "ID": r["ID"],
                        "Domain": r["Domain"],
                        "Difficulty_Level": r["Difficulty_Level"],
                        "Question": r["Question"],
                        "Reference_Answer": ans,
                    }
                    for k, v in row_dict.items():
                        if not str(v).strip():
                            blank_count += 1
                    writer_aug.writerow(row_dict)
                    aug_count += 1

    sha_end = calculate_sha256(INPUT_CSV)
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    report_text = f"""=== 5x DATASET AUGMENTATION REPORT ===

Input file:
data/LLM_Interview_Dataset-v2.csv

Original SHA256:
{sha_start}

Final input SHA256:
{sha_end}

Original rows:
{len(rows)}

Unique questions:
{len(unique_q_map)}

Required variations per unique question:
5

Generated unique QA pairs:
{unique_count}

Expected unique QA pairs:
23600

Actual unique QA pairs:
{unique_count}

Expected reconstructed rows:
30620

Actual reconstructed rows:
{aug_count}

Questions with fewer than 5 answers:
{len(unique_q_map) - len(cached_answers)}

Blank required fields:
{blank_count}

Original file modified:
{"NO" if sha_start == sha_end == EXPECTED_SHA256 else "YES"}

Overall validation:
{"PASS" if unique_count == 23600 and aug_count == 30620 and sha_start == sha_end == EXPECTED_SHA256 else "IN_PROGRESS"}
"""
    with open(REPORT_TXT, "w", encoding="utf-8") as rf:
        rf.write(report_text)

    return unique_count, aug_count, len(cached_answers), len(unique_q_map)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--get-pending", type=int, default=0, help="Output next N pending questions as JSON")
    parser.add_argument("--process-file", type=str, default="", help="Process batch JSON file and append to cache")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild final output CSVs and report")
    args = parser.parse_args()

    if args.get_pending > 0:
        pending = get_pending_questions(args.get_pending)
        print(json.dumps(pending, indent=2, ensure_ascii=False))
    elif args.process_file:
        with open(args.process_file, "r", encoding="utf-8") as f:
            batch = json.load(f)
        valid, err_cnt, errs = append_batch(batch)
        print(f"Batch processed: {valid} valid items written, {err_cnt} errors.")
        if errs:
            print("Errors:", errs[:5])
        unq, aug, cached, tot = rebuild_datasets()
        print(f"Current progress: {cached}/{tot} questions ({unq} QA pairs, {aug} augmented rows)")
    elif args.rebuild:
        unq, aug, cached, tot = rebuild_datasets()
        print(f"Datasets rebuilt: {cached}/{tot} questions ({unq} QA pairs, {aug} augmented rows)")
