"""
generate_augmented_dataset_5x.py
================================
Production-grade, resumable dataset augmentation pipeline for LLM Interview Dataset.
Generates exactly 5 distinct, high-quality, factually accurate answer variations
for every unique question in data/LLM_Interview_Dataset-v2.csv.

Maintains absolute file protection of original dataset (verified via SHA256).

Outputs:
- data/LLM_Interview_Dataset-Augmented-5x.csv (Reconstructed full dataset: 30,620 rows)
- data/LLM_Interview_Dataset-Unique-5x.csv    (Clean unique training dataset: 23,600 rows)
- scratch/augmented_5x_cache.jsonl           (Checkpoint/resume cache)
- reports/augmentation_5x_validation.txt     (Validation and audit report)
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp

EXPECTED_INPUT_SHA256 = "3b465f782b30f036c83dfc978dfc9bf53480826444f39bf9730d7a582ae72fcf"
REQUIRED_COLUMNS = ["ID", "Domain", "Difficulty_Level", "Question", "Reference_Answer"]
NUM_VARIATIONS = 5

DEFAULT_MODEL_POOL = [
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]

SYSTEM_PROMPT = """You are an expert technical interviewer and AI educator creating high-quality dataset variations for fine-tuning LLMs on technical and professional interviews.

Given a Question, Domain, Difficulty Level, and ground truth Reference Answer, generate EXACTLY 5 distinct, high-quality, factually correct answer variations.

Adhere strictly to these 5 distinct explanation styles:
1. Direct and Precise: A concise, authoritative answer that directly addresses the question without fluff.
2. Beginner Friendly: A clear, accessible explanation using intuitive analogies or simplified terms while remaining accurate.
3. Detailed / Technical: An in-depth technical explanation with precise terminology, underlying mechanics, architectural context, or formal definitions.
4. Example Based: An explanation anchored around a concrete scenario, real-world use case, or relevant code snippet.
5. Practical / Alternative Explanation: A practical, production-oriented, or alternative perspective on how/why this applies in industry.

CRITICAL RULES:
- FACTUAL FIDELITY: All 5 answers must be 100% factually correct and preserve the ground truth meaning of the reference answer. No hallucinations or unsupported claims.
- NO SUPERFICIAL PARAPHRASES: Do not just swap synonyms. Variations must differ in structure, depth, perspective, and framing.
- NO FORMULAIC ANSWERS: Do NOT start answers with formulaic templates like "The answer is...", "The correct answer is...", or "In response to your question...".
- Respond ONLY with valid JSON matching the exact schema below.

JSON Schema:
{
  "question": "<exact question text>",
  "answers": [
    "<Variation 1: Direct and Precise>",
    "<Variation 2: Beginner Friendly>",
    "<Variation 3: Detailed / Technical>",
    "<Variation 4: Example Based>",
    "<Variation 5: Practical / Alternative Explanation>"
  ]
}
"""

USER_PROMPT_TEMPLATE = """Question: {question}
Domain: {domain}
Difficulty: {difficulty}
Reference Answer: {reference}

Generate the 5 distinct answer variations in valid JSON format:"""


def calculate_sha256(filepath: Path) -> str:
    """Calculate the SHA256 checksum of a file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_jaccard_similarity(str1: str, str2: str) -> float:
    """Compute token-level Jaccard similarity between two strings."""
    tokens1 = set(re.findall(r"\b\w+\b", str1.lower()))
    tokens2 = set(re.findall(r"\b\w+\b", str2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if union else 0.0


def check_diversity(answers: List[str], threshold: float = 0.85) -> Tuple[bool, str]:
    """Verify that all 5 answers are mutually diverse and valid."""
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


def is_hard_quota_exhaustion(status_code: int, response_text: str) -> bool:
    """
    Determines if an HTTP error represents hard daily/project quota exhaustion
    rather than a temporary burst rate-limit.
    """
    if status_code != 429:
        return False
    lower = response_text.lower()
    quota_indicators = [
        "quota exceeded for metric",
        "generaterequestsperday",
        "exceeded your current quota",
        "check your plan and billing details",
        "free_tier_requests",
        "quotafailure",
    ]
    return any(indicator in lower for indicator in quota_indicators)


class AdaptiveRateLimiter:
    """Adaptive rate limiter with exponential backoff on temporary rate limits."""
    def __init__(self, requests_per_minute: int = 60, max_concurrent: int = 4):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
        self.backoff_multiplier = 1.0

    async def acquire(self):
        await self.semaphore.acquire()
        async with self._lock:
            now = time.time()
            effective_interval = self.min_interval * self.backoff_multiplier
            elapsed = now - self.last_request_time
            if elapsed < effective_interval:
                await asyncio.sleep(effective_interval - elapsed)
            self.last_request_time = time.time()

    def release(self):
        self.semaphore.release()

    def report_rate_limit(self):
        self.backoff_multiplier = min(4.0, self.backoff_multiplier * 1.5)

    def report_success(self):
        self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.95)


class QuotaManager:
    """Tracks active models and detects when all project quotas are depleted."""
    def __init__(self, initial_models: List[str]):
        self.active_models = list(initial_models)
        self.exhausted_models: Set[str] = set()
        self.all_exhausted = False
        self._lock = asyncio.Lock()
        self.stats = {
            "api_errors": 0,
            "rate_limits_429": 0,
            "quota_exhaustions": 0,
            "retries": 0,
        }

    async def mark_model_exhausted(self, model: str):
        async with self._lock:
            if model in self.active_models:
                self.active_models.remove(model)
                self.exhausted_models.add(model)
                self.stats["quota_exhaustions"] += 1
                print(f"\n[QUOTA EXHAUSTED] Model '{model}' daily/project quota reached. Removed from active pool. Remaining active models: {len(self.active_models)}", flush=True)
            if not self.active_models:
                self.all_exhausted = True
                print("\n[CRITICAL] All available model quotas in pool are exhausted. Gracefully halting pipeline to preserve cache...", flush=True)

    async def get_next_model(self, attempt_idx: int) -> Optional[str]:
        async with self._lock:
            if not self.active_models:
                return None
            return self.active_models[attempt_idx % len(self.active_models)]


async def generate_variations_for_question(
    session: aiohttp.ClientSession,
    question: str,
    domain: str,
    difficulty: str,
    reference: str,
    api_key: str,
    quota_manager: QuotaManager,
    rate_limiter: AdaptiveRateLimiter,
    max_retries: int = 5,
    similarity_threshold: float = 0.85,
) -> Tuple[Optional[List[str]], int, str]:
    """
    Generate 5 distinct answer variations using Gemini REST API.
    Handles temporary rate limits vs hard quota exhaustion cleanly.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        domain=domain,
        difficulty=difficulty,
        reference=reference,
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.7,
        }
    }

    retries = 0
    last_error = ""

    for attempt in range(max_retries):
        if quota_manager.all_exhausted:
            return None, retries, "All model quotas exhausted"

        model = await quota_manager.get_next_model(attempt)
        if not model:
            return None, retries, "No active models remaining"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        await rate_limiter.acquire()
        try:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                status = resp.status
                resp_text = await resp.text()

                if status == 429:
                    quota_manager.stats["rate_limits_429"] += 1
                    retries += 1
                    quota_manager.stats["retries"] += 1

                    if is_hard_quota_exhaustion(status, resp_text):
                        await quota_manager.mark_model_exhausted(model)
                        continue
                    else:
                        rate_limiter.report_rate_limit()
                        wait = min(15.0, (2 ** attempt) * 1.5 + random.uniform(0.5, 1.5))
                        last_error = f"Temporary 429 on {model}, sleeping {wait:.1f}s"
                        await asyncio.sleep(wait)
                        continue

                if status == 503 or status == 500:
                    quota_manager.stats["api_errors"] += 1
                    retries += 1
                    quota_manager.stats["retries"] += 1
                    wait = min(10.0, (1.5 ** attempt) + random.uniform(0.2, 0.8))
                    last_error = f"HTTP {status} on {model}"
                    await asyncio.sleep(wait)
                    continue

                if status != 200:
                    quota_manager.stats["api_errors"] += 1
                    retries += 1
                    quota_manager.stats["retries"] += 1
                    last_error = f"HTTP {status} ({model}): {resp_text[:100]}"
                    await asyncio.sleep(1.0)
                    continue

                # Parse successful response
                data = json.loads(resp_text)
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
                    parsed = json.loads(text)
                    answers = parsed.get("answers", [])
                    if not isinstance(answers, list) or len(answers) != NUM_VARIATIONS:
                        raise ValueError(f"Invalid answer count: {len(answers) if isinstance(answers, list) else 'not a list'}")

                    cleaned_answers = [str(a).strip() for a in answers]
                    is_valid, reason = check_diversity(cleaned_answers, threshold=similarity_threshold)
                    if not is_valid:
                        retries += 1
                        quota_manager.stats["retries"] += 1
                        last_error = f"Diversity check failed: {reason}"
                        payload["generationConfig"]["temperature"] = min(0.95, 0.7 + attempt * 0.05)
                        await asyncio.sleep(0.5)
                        continue

                    rate_limiter.report_success()
                    return cleaned_answers, retries, "OK"

                except Exception as parse_err:
                    retries += 1
                    quota_manager.stats["retries"] += 1
                    last_error = f"Parse/Validation error on {model}: {parse_err}"
                    await asyncio.sleep(0.5)
                    continue

        except asyncio.TimeoutError:
            retries += 1
            quota_manager.stats["retries"] += 1
            quota_manager.stats["api_errors"] += 1
            last_error = f"Timeout on {model}"
            await asyncio.sleep(1.0)
        except Exception as e:
            retries += 1
            quota_manager.stats["retries"] += 1
            quota_manager.stats["api_errors"] += 1
            last_error = f"Exception on {model}: {e}"
            await asyncio.sleep(1.0)
        finally:
            rate_limiter.release()

    return None, retries, last_error


async def run_augmentation_pipeline(args):
    root_dir = Path(__file__).resolve().parent.parent
    input_csv = (root_dir / args.input_csv).resolve()
    output_aug_csv = (root_dir / args.output_aug_csv).resolve()
    output_unique_csv = (root_dir / args.output_unique_csv).resolve()
    cache_jsonl = (root_dir / args.cache_jsonl).resolve()
    report_txt = (root_dir / args.report_txt).resolve()

    print("=" * 60, flush=True)
    print("5x DATASET AUGMENTATION PIPELINE", flush=True)
    print("=" * 60, flush=True)
    print(f"Input Dataset:       {input_csv}", flush=True)
    print(f"Augmented Output:    {output_aug_csv}", flush=True)
    print(f"Unique Output:       {output_unique_csv}", flush=True)
    print(f"Cache File:          {cache_jsonl}", flush=True)
    print(f"Report File:         {report_txt}", flush=True)
    print(f"Concurrency:         {args.concurrency}", flush=True)
    print("=" * 60, flush=True)

    # 1. INITIAL INTEGRITY VERIFICATION
    print("\n[Step 1] Verifying input file and SHA256 integrity...", flush=True)
    if not input_csv.exists():
        sys.exit(f"CRITICAL ERROR: Input dataset not found at {input_csv}")

    initial_sha256 = calculate_sha256(input_csv)
    print(f"Initial SHA256: {initial_sha256}", flush=True)
    if initial_sha256 != EXPECTED_INPUT_SHA256:
        print(f"WARNING: Expected SHA256 '{EXPECTED_INPUT_SHA256}', but found '{initial_sha256}'.", flush=True)

    # 2. LOAD DATASET & ANALYZE
    print("\n[Step 2] Loading and analyzing dataset structure...", flush=True)
    with open(input_csv, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        if header != REQUIRED_COLUMNS:
            sys.exit(f"CRITICAL ERROR: Unexpected CSV columns: {header}. Expected: {REQUIRED_COLUMNS}")
        original_rows = list(reader)

    total_rows = len(original_rows)
    print(f"Total rows loaded: {total_rows}", flush=True)

    # Group by unique questions
    unique_q_map: Dict[str, Dict] = {}
    q_occurrences = defaultdict(list)
    conflicting_questions = []

    for row_idx, row in enumerate(original_rows):
        q = row["Question"].strip()
        ref = row["Reference_Answer"].strip()
        q_occurrences[q].append(row_idx)

        if q not in unique_q_map:
            unique_q_map[q] = {
                "question": q,
                "domain": row["Domain"].strip(),
                "difficulty": row["Difficulty_Level"].strip(),
                "reference": ref,
                "all_refs": {ref},
                "original_indices": [row_idx],
            }
        else:
            unique_q_map[q]["all_refs"].add(ref)
            unique_q_map[q]["original_indices"].append(row_idx)
            if len(unique_q_map[q]["all_refs"]) > 1 and q not in conflicting_questions:
                conflicting_questions.append(q)

    num_unique_questions = len(unique_q_map)
    duplicate_q_count = sum(1 for q, idxs in q_occurrences.items() if len(idxs) > 1)
    print(f"Unique questions identified: {num_unique_questions}", flush=True)
    print(f"Questions appearing multiple times: {duplicate_q_count}", flush=True)
    print(f"Questions with conflicting reference answers: {len(conflicting_questions)}", flush=True)

    expected_unique_pairs = num_unique_questions * NUM_VARIATIONS
    expected_reconstructed_rows = total_rows * NUM_VARIATIONS
    print(f"Expected Unique QA Pairs:     {expected_unique_pairs}", flush=True)
    print(f"Expected Reconstructed Rows:  {expected_reconstructed_rows}", flush=True)

    # 3. LOAD CACHE / CHECKPOINT
    print("\n[Step 3] Loading cache / checkpoint...", flush=True)
    cache_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cached_answers: Dict[str, List[str]] = {}

    if cache_jsonl.exists():
        with open(cache_jsonl, mode="r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    q = record["question"].strip()
                    answers = record["answers"]
                    if len(answers) == NUM_VARIATIONS:
                        cached_answers[q] = answers
                except Exception as e:
                    print(f"  Warning: failed to parse cache line {line_no}: {e}", flush=True)

    print(f"Loaded {len(cached_answers)} valid question entries from cache.", flush=True)

    unique_q_list = list(unique_q_map.values())
    if args.limit and args.limit > 0:
        unique_q_list = unique_q_list[:args.limit]
        print(f"NOTE: Limit flag active. Processing up to {len(unique_q_list)} unique questions.", flush=True)

    pending_questions = [q_obj for q_obj in unique_q_list if q_obj["question"] not in cached_answers]
    print(f"Pending questions to generate: {len(pending_questions)}", flush=True)

    # 4. LLM GENERATION PROCESS
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        sys.exit("CRITICAL ERROR: GEMINI_API_KEY environment variable is not set.")

    quota_manager = QuotaManager(DEFAULT_MODEL_POOL)
    rate_limiter = AdaptiveRateLimiter(
        requests_per_minute=args.rpm,
        max_concurrent=args.concurrency
    )

    completed_count = len(cached_answers)
    target_count = len(unique_q_list)
    failed_questions = []

    if pending_questions and not quota_manager.all_exhausted:
        print(f"\n[Step 4] Launching adaptive generation with {args.concurrency} workers across {len(DEFAULT_MODEL_POOL)} model pool...", flush=True)
        cache_lock = asyncio.Lock()
        t_start = time.time()

        async with aiohttp.ClientSession() as session:
            async def worker(q_obj):
                nonlocal completed_count
                if quota_manager.all_exhausted:
                    return

                q = q_obj["question"]
                dom = q_obj["domain"]
                diff = q_obj["difficulty"]
                ref = q_obj["reference"]

                answers, retries, status_msg = await generate_variations_for_question(
                    session=session,
                    question=q,
                    domain=dom,
                    difficulty=diff,
                    reference=ref,
                    api_key=api_key,
                    quota_manager=quota_manager,
                    rate_limiter=rate_limiter,
                    max_retries=args.max_retries,
                    similarity_threshold=args.similarity_threshold,
                )

                if answers is not None and len(answers) == NUM_VARIATIONS:
                    async with cache_lock:
                        cached_answers[q] = answers
                        completed_count += 1
                        with open(cache_jsonl, mode="a", encoding="utf-8") as cf:
                            cf.write(json.dumps({"question": q, "answers": answers}, ensure_ascii=False) + "\n")
                            cf.flush()

                    if completed_count % 25 == 0 or completed_count == target_count:
                        elapsed = time.time() - t_start
                        rate = completed_count / max(1.0, elapsed)
                        print(f"  Progress: [{completed_count}/{target_count}] ({(completed_count/target_count)*100:.1f}%) | Speed: {rate:.2f} q/s | 429s: {quota_manager.stats['rate_limits_429']} | Quota Exhaustions: {quota_manager.stats['quota_exhaustions']}", flush=True)
                else:
                    failed_questions.append(q)
                    if not quota_manager.all_exhausted:
                        print(f"  [FAILED/SKIPPED] Question: '{q[:50]}...' -> {status_msg}", flush=True)

            batch_size = args.concurrency * 2
            for i in range(0, len(pending_questions), batch_size):
                if quota_manager.all_exhausted:
                    print("\n[NOTICE] Stopping further requests: all available model quotas exhausted. Progress preserved.", flush=True)
                    break
                batch = pending_questions[i:i + batch_size]
                tasks = [worker(q_obj) for q_obj in batch]
                await asyncio.gather(*tasks)

    # 5. RECONSTRUCT OUTPUT DATASETS (if full or requested)
    print("\n[Step 5] Reconstructing output datasets from cached progress...", flush=True)
    output_aug_csv.parent.mkdir(parents=True, exist_ok=True)
    output_unique_csv.parent.mkdir(parents=True, exist_ok=True)

    actual_unique_rows = 0
    questions_missing_answers = 0
    questions_invalid_variations = 0

    with open(output_unique_csv, mode="w", encoding="utf-8", newline="") as f_unique:
        writer_unique = csv.DictWriter(f_unique, fieldnames=REQUIRED_COLUMNS)
        writer_unique.writeheader()

        for q_obj in unique_q_map.values():
            q = q_obj["question"]
            if q not in cached_answers:
                questions_missing_answers += 1
                continue
            variations = cached_answers[q]
            if len(variations) != NUM_VARIATIONS:
                questions_invalid_variations += 1
                continue

            for var_ans in variations:
                writer_unique.writerow({
                    "ID": q_obj["original_indices"][0] + 1,
                    "Domain": q_obj["domain"],
                    "Difficulty_Level": q_obj["difficulty"],
                    "Question": q,
                    "Reference_Answer": var_ans,
                })
                actual_unique_rows += 1

    print(f"Wrote {actual_unique_rows} rows to {output_unique_csv}", flush=True)

    actual_reconstructed_rows = 0
    blank_field_count = 0

    with open(output_aug_csv, mode="w", encoding="utf-8", newline="") as f_aug:
        writer_aug = csv.DictWriter(f_aug, fieldnames=REQUIRED_COLUMNS)
        writer_aug.writeheader()

        for row in original_rows:
            q = row["Question"].strip()
            if q not in cached_answers or len(cached_answers[q]) != NUM_VARIATIONS:
                continue
            variations = cached_answers[q]
            for var_ans in variations:
                row_dict = {
                    "ID": row["ID"],
                    "Domain": row["Domain"],
                    "Difficulty_Level": row["Difficulty_Level"],
                    "Question": row["Question"],
                    "Reference_Answer": var_ans,
                }
                for k, v in row_dict.items():
                    if not str(v).strip():
                        blank_field_count += 1
                writer_aug.writerow(row_dict)
                actual_reconstructed_rows += 1

    print(f"Wrote {actual_reconstructed_rows} rows to {output_aug_csv}", flush=True)

    # 6. POST-RUN AUDIT & SHA256 VERIFICATION
    print("\n[Step 6] Running post-generation integrity audits...", flush=True)
    final_input_sha256 = calculate_sha256(input_csv)
    original_file_modified = (final_input_sha256 != initial_sha256) or (final_input_sha256 != EXPECTED_INPUT_SHA256)

    duplicate_violation_count = 0
    for q, vars in cached_answers.items():
        is_valid, _ = check_diversity(vars, threshold=args.similarity_threshold)
        if not is_valid:
            duplicate_violation_count += 1

    is_overall_pass = (
        not original_file_modified
        and questions_missing_answers == 0
        and questions_invalid_variations == 0
        and blank_field_count == 0
        and duplicate_violation_count == 0
        and (args.limit or (actual_unique_rows == expected_unique_pairs and actual_reconstructed_rows == expected_reconstructed_rows))
    )

    overall_status = "PASS" if is_overall_pass else ("IN_PROGRESS (Resumable)" if quota_manager.all_exhausted or questions_missing_answers > 0 else "FAIL")

    report_content = f"""=== 5x DATASET AUGMENTATION REPORT ===

Input file:
{args.input_csv}

Original SHA256:
{initial_sha256}

Final input SHA256:
{final_input_sha256}

Original rows:
{total_rows}

Unique questions:
{num_unique_questions}

Questions with conflicting reference answers:
{len(conflicting_questions)}

Required variations per unique question:
{NUM_VARIATIONS}

Generated unique QA pairs:
{actual_unique_rows}

Expected unique QA pairs:
{expected_unique_pairs}

Actual unique QA pairs:
{actual_unique_rows}

Expected reconstructed rows:
{expected_reconstructed_rows}

Actual reconstructed rows:
{actual_reconstructed_rows}

Questions with fewer than 5 answers:
{questions_missing_answers}

Questions with more than 5 answers:
{questions_invalid_variations}

Blank required fields:
{blank_field_count}

Duplicate/near-duplicate answer violations:
{duplicate_violation_count}

API Errors:
{quota_manager.stats['api_errors']}

Rate Limit (429) events:
{quota_manager.stats['rate_limits_429']}

Quota Exhaustion events:
{quota_manager.stats['quota_exhaustions']}

Total Retries:
{quota_manager.stats['retries']}

Original file modified:
{"YES - CRITICAL ERROR" if original_file_modified else "NO"}

Overall validation:
{overall_status}
"""

    report_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(report_txt, mode="w", encoding="utf-8") as rf:
        rf.write(report_content)

    print("\n" + report_content, flush=True)
    print(f"Validation report saved to: {report_txt}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate 5x Augmented LLM Interview Dataset")
    parser.add_argument("--input-csv", default="data/LLM_Interview_Dataset-v2.csv", help="Path to input CSV")
    parser.add_argument("--output-aug-csv", default="data/LLM_Interview_Dataset-Augmented-5x.csv", help="Path to full reconstructed augmented CSV")
    parser.add_argument("--output-unique-csv", default="data/LLM_Interview_Dataset-Unique-5x.csv", help="Path to unique training CSV")
    parser.add_argument("--cache-jsonl", default="scratch/augmented_5x_cache.jsonl", help="Path to cache / checkpoint JSONL")
    parser.add_argument("--report-txt", default="reports/augmentation_5x_validation.txt", help="Path to validation report file")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument("--rpm", type=int, default=60, help="Requests per minute target")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per request on failure/429")
    parser.add_argument("--similarity-threshold", type=float, default=0.82, help="Max pairwise Jaccard similarity threshold")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for testing/dry runs)")

    args = parser.parse_args()
    asyncio.run(run_augmentation_pipeline(args))


if __name__ == "__main__":
    main()
