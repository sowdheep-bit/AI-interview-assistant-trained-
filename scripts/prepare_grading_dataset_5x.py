"""
prepare_grading_dataset_5x.py
=============================
Prepares stratified, concept-level disjoint Train / Validation / Test splits
from data/LLM_Interview_Dataset-Unique-5x.csv for the interview grading task.

Guarantees:
- Concept-level disjoint: No normalized question concept leaks between train, val, and test.
- The 30 GOLD_STANDARD questions from evaluate_model.py are strictly excluded from train.
- Generates balanced Perfect (score 8-10), Partial (score 4-7), and Wrong (score 0-3) grading examples.
- Generates train (80%), validation (10%), and test (10%) JSONL datasets.
"""

import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
INPUT_UNIQUE_CSV = ROOT / "data" / "LLM_Interview_Dataset-Unique-5x.csv"
EVAL_SCRIPT = ROOT / "scripts" / "evaluate_model.py"
OUT_TRAIN_JSONL = ROOT / "data" / "grading_train_5x.jsonl"
OUT_VAL_JSONL = ROOT / "data" / "grading_val_5x.jsonl"
OUT_TEST_JSONL = ROOT / "data" / "grading_test_5x.jsonl"
OUT_SPLITS_META = ROOT / "data" / "splits_metadata_5x.json"

RANDOM_SEED = 42

PREFIX_RE = re.compile(
    r"^(Candidate prompt|Could you explain this|Define and explain"
    r"|Interview Question|Please answer the following)\s*:\s*",
    re.IGNORECASE,
)


def norm_q(q: str) -> str:
    return PREFIX_RE.sub("", str(q).strip()).strip().lower()


def load_gold_standard_questions() -> set:
    src = EVAL_SCRIPT.read_text(encoding="utf-8")
    m = re.search(r"GOLD_STANDARD\s*=\s*\[(.*?)\n\]\n", src, re.DOTALL)
    if not m:
        raise RuntimeError("Could not find GOLD_STANDARD list in evaluate_model.py")
    qs = re.findall(r'\(\s*"((?:[^"\\]|\\.)*)"\s*,', m.group(1))
    return {norm_q(q) for q in qs}


def create_partial_answers(reference_variations):
    """Generates realistic partial candidate answers (vague, missing details, score 4-7)."""
    partials = []
    for var in reference_variations:
        # 1. Take first clause / introductory fragment
        sentences = re.split(r"[.!?]\s+", var.strip())
        if sentences:
            s1 = sentences[0].strip()
            words = s1.split()
            if len(words) > 10:
                short_s = " ".join(words[: len(words) // 2])
                partials.append((short_s + "...", 4))
            else:
                partials.append((s1 + " but I don't remember all the specific details.", 5))
        
        # 2. Add a vague high-level paraphrase
        words = var.split()
        if len(words) >= 6:
            vague = f"It relates to {words[1]} and {words[2]}, providing foundational support for the architecture."
            partials.append((vague, 5))
    return partials


def main():
    print("=== PREPARING 5x GRADING DATASETS ===")
    gold_qs = load_gold_standard_questions()
    print(f"Loaded {len(gold_qs)} gold-standard benchmark questions to exclude from training.")

    # 1. Group rows by NORMALIZED question concept to guarantee zero concept leakage
    rows = list(csv.DictReader(open(INPUT_UNIQUE_CSV, encoding="utf-8")))
    concept_groups = defaultdict(list)
    for r in rows:
        concept_key = norm_q(r["Question"])
        concept_groups[concept_key].append(r)

    print(f"Total Unique Normalized Concepts in Dataset: {len(concept_groups)}")

    # 2. Stratified question split by (Domain, Difficulty)
    strata = defaultdict(list)
    for concept_key, grp in concept_groups.items():
        dom = grp[0]["Domain"].strip()
        diff = grp[0]["Difficulty_Level"].strip()
        strata[(dom, diff)].append((concept_key, grp))

    rng = random.Random(RANDOM_SEED)

    train_concepts = []
    val_concepts = []
    test_concepts = []

    # Domain distractor pools for generating wrong candidate answers
    domain_distractors = defaultdict(list)
    for concept_key, grp in concept_groups.items():
        dom = grp[0]["Domain"].strip()
        domain_distractors[dom].extend([r["Reference_Answer"].strip() for r in grp])

    for cell, items in strata.items():
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, int(round(n * 0.10)))
        n_test = max(1, int(round(n * 0.10)))
        n_train = n - n_val - n_test

        train_pool = items[:n_train]
        val_pool = items[n_train : n_train + n_val]
        test_pool = items[n_train + n_val :]

        train_concepts.extend(train_pool)
        val_concepts.extend(val_pool)
        test_concepts.extend(test_pool)

    # Strictly exclude gold standard questions from train
    train_set = {k for k, _ in train_concepts}
    val_set = {k for k, _ in val_concepts}
    test_set = {k for k, _ in test_concepts}

    gold_train_overlap = train_set.intersection(gold_qs)
    if gold_train_overlap:
        print(f"Removing {len(gold_train_overlap)} gold standard questions from train set...")
        train_concepts = [item for item in train_concepts if item[0] not in gold_qs]
        train_set = {k for k, _ in train_concepts}

    # Verify zero question leakage
    assert len(train_set.intersection(val_set)) == 0, "Leakage between Train and Val!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage between Train and Test!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage between Val and Test!"
    assert len(train_set.intersection(gold_qs)) == 0, "Gold standard leaked into Train!"

    print(f"\nDisjoint Split Summary:")
    print(f"  Train Concepts:      {len(train_concepts)} ({len(train_concepts)/len(concept_groups)*100:.1f}%)")
    print(f"  Validation Concepts: {len(val_concepts)} ({len(val_concepts)/len(concept_groups)*100:.1f}%)")
    print(f"  Test Concepts:       {len(test_concepts)} ({len(test_concepts)/len(concept_groups)*100:.1f}%)")
    print("Leakage verification: PASSED (0 overlap between splits, 0 gold standard in train).")

    # 3. Generate grading examples for each split
    def build_grading_records(split_concepts, is_train=True):
        records = []
        for concept_key, grp in split_concepts:
            q_raw = grp[0]["Question"].strip()
            dom = grp[0]["Domain"].strip()
            diff = grp[0]["Difficulty_Level"].strip()
            variations = list(dict.fromkeys([r["Reference_Answer"].strip() for r in grp]))

            if len(variations) < 2:
                continue

            # --- A. PERFECT TIER (Score 8-10) ---
            # Cross-variation pairs: Ideal = V_i, Candidate = V_j (j != i)
            # Both are valid, complete, and factually accurate formulations
            num_vars = len(variations)
            perfect_pairs = []
            for i in range(num_vars):
                j = (i + 1) % num_vars
                perfect_pairs.append((i, j, 9))
            
            if not is_train:
                perfect_pairs = perfect_pairs[:2]
            else:
                perfect_pairs = perfect_pairs[:4]

            for ref_idx, cand_idx, score in perfect_pairs:
                records.append({
                    "question": q_raw,
                    "reference": variations[ref_idx],
                    "candidate": variations[cand_idx],
                    "score": score,
                    "tier": "P",
                    "domain": dom,
                    "difficulty": diff,
                })

            # --- B. PARTIAL TIER (Score 4-7) ---
            partials = create_partial_answers(variations)
            num_partials = 3 if is_train else 2
            rng.shuffle(partials)
            for cand_text, score in partials[:num_partials]:
                ref_choice = rng.choice(variations)
                records.append({
                    "question": q_raw,
                    "reference": ref_choice,
                    "candidate": cand_text,
                    "score": score,
                    "tier": "M",
                    "domain": dom,
                    "difficulty": diff,
                })

            # --- C. WRONG TIER (Score 0-3) ---
            # Same-domain hard negative distractor
            dom_pool = [d for d in domain_distractors[dom] if d not in variations]
            num_wrong = 3 if is_train else 2
            wrong_candidates = []
            if len(dom_pool) >= num_wrong:
                wrong_candidates.extend([(rng.choice(dom_pool), rng.choice([1, 2])) for _ in range(num_wrong - 1)])
            wrong_candidates.append(("It is an unrelated feature that is not used in this architecture.", 1))

            for cand_text, score in wrong_candidates[:num_wrong]:
                ref_choice = rng.choice(variations)
                records.append({
                    "question": q_raw,
                    "reference": ref_choice,
                    "candidate": cand_text,
                    "score": score,
                    "tier": "W",
                    "domain": dom,
                    "difficulty": diff,
                })

        return records

    train_records = build_grading_records(train_concepts, is_train=True)
    val_records = build_grading_records(val_concepts, is_train=False)
    test_records = build_grading_records(test_concepts, is_train=False)

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    with open(OUT_TRAIN_JSONL, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUT_VAL_JSONL, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUT_TEST_JSONL, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metadata = {
        "random_seed": RANDOM_SEED,
        "total_unique_concepts": len(concept_groups),
        "train_concepts": len(train_concepts),
        "val_concepts": len(val_concepts),
        "test_concepts": len(test_concepts),
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "test_examples": len(test_records),
        "gold_standard_excluded": len(gold_qs),
    }

    with open(OUT_SPLITS_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n--- GRADING DATASET SPLITS CREATED ---")
    print(f"Train Dataset:      {len(train_records)} examples -> {OUT_TRAIN_JSONL}")
    print(f"Validation Dataset: {len(val_records)} examples -> {OUT_VAL_JSONL}")
    print(f"Test Dataset:       {len(test_records)} examples -> {OUT_TEST_JSONL}")
    print(f"Metadata saved:     {OUT_SPLITS_META}")


if __name__ == "__main__":
    main()
