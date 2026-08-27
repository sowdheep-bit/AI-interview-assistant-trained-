"""
evaluate_model_5x.py
====================
Evaluates interview grading models on both:
1. The original 30-case GOLD_STANDARD benchmark (comparing against 63.3% baseline)
2. The unseen held-out 469-question test set (data/grading_test_5x.jsonl) with
   full domain-level metrics (MAE, RMSE, Tier Accuracy).
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
TEST_JSONL = ROOT / "data" / "grading_test_5x.jsonl"
DEFAULT_ADAPTER = ROOT / "models" / "fine_tuned_interviewer_5x"
BASELINE_ADAPTER = ROOT / "models" / "fine_tuned_interviewer"

# 30-case GOLD_STANDARD Benchmark from evaluate_model.py
GOLD_STANDARD = [
    # --- PERFECT ANSWERS ---
    ("What is a segmentation fault?",
     "A segmentation fault occurs when a program attempts to access a memory location it is not allowed to access, or attempts to access memory in a way that is not allowed.",
     "A segfault happens when a program tries to access memory it doesn't have permission to use—like reading a null pointer or writing outside an allocated buffer.",
     "P"),
    ("What is the primary characteristic of an AVL tree?",
     "An AVL tree is a self-balancing binary search tree where the difference in heights of left and right subtrees cannot exceed one for all nodes.",
     "An AVL tree is a self-balancing BST. The balance factor of every node must be -1, 0, or +1. If violated, rotations (single or double) restore balance.",
     "P"),
    ("How does a Random Forest Classifier make predictions?",
     "It trains multiple decision trees on random subsets of data and features, then uses majority voting for classification or averaging for regression.",
     "Random Forest builds many decision trees on bootstrapped data samples with random feature subsets, and aggregates their outputs via majority vote.",
     "P"),
    ("What is the primary difference in memory management between C and C++?",
     "C uses malloc() and free() for raw memory management, while C++ additionally handles the invocation of constructors and destructors through new and delete.",
     "In C you use malloc/free. C++ adds new/delete which also call constructors and destructors. C++ also supports RAII via smart pointers like unique_ptr.",
     "P"),
    ("Explain the concept of a zombie process.",
     "A zombie process is a child process that has completed execution but still has an entry in the process table because its parent has not yet read its exit status.",
     "A zombie is a dead process that still has a process table entry because its parent hasn't called wait() to collect its exit status.",
     "P"),
    ("What is dynamic programming?",
     "Dynamic programming is a technique for solving problems by breaking them into overlapping subproblems, solving each once, and storing results to avoid redundant computation.",
     "DP solves problems by caching results of overlapping subproblems—memoization (top-down) or tabulation (bottom-up)—to avoid recomputation.",
     "P"),
    ("What is overfitting in machine learning?",
     "Overfitting occurs when a model learns the training data too well, including noise, and fails to generalize to unseen data.",
     "A model overfits when it memorizes training data including noise, causing high training accuracy but poor test accuracy.",
     "P"),
    ("What is a hash table?",
     "A hash table is a data structure that maps keys to values using a hash function to compute an index into an array of buckets.",
     "A hash table uses a hash function to convert keys into array indices, enabling O(1) average-case lookup, insert, and delete.",
     "P"),
    ("What is gradient descent?",
     "Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction of the negative gradient of the loss function to minimize it.",
     "Gradient descent minimizes a loss function by updating parameters in the direction opposite to the gradient—step size controlled by the learning rate.",
     "P"),
    ("What is the difference between process and thread?",
     "A process is an independent program in execution with its own memory space, while a thread is a lightweight unit within a process that shares the process's memory.",
     "Processes are independent execution units with separate memory. Threads share the same address space and are lighter-weight, enabling concurrent execution within one process.",
     "P"),

    # --- PARTIAL ANSWERS ---
    ("What is a segmentation fault?",
     "A segmentation fault occurs when a program attempts to access a memory location it is not allowed to access, or attempts to access memory in a way that is not allowed.",
     "It's a memory error. It happens when you do something wrong with pointers.",
     "M"),
    ("What is the primary characteristic of an AVL tree?",
     "An AVL tree is a self-balancing binary search tree where the difference in heights of left and right subtrees cannot exceed one for all nodes.",
     "AVL tree keeps the tree balanced so searches are fast.",
     "M"),
    ("How does a Random Forest Classifier make predictions?",
     "It trains multiple decision trees on random subsets of data and features, then uses majority voting for classification or averaging for regression.",
     "It uses many trees and combines their answers somehow.",
     "M"),
    ("What is dynamic programming?",
     "Dynamic programming is a technique for solving problems by breaking them into overlapping subproblems, solving each once, and storing results to avoid redundant computation.",
     "Dynamic programming is about storing results of calculations and reusing them later.",
     "M"),
    ("What is overfitting in machine learning?",
     "Overfitting occurs when a model learns the training data too well, including noise, and fails to generalize to unseen data.",
     "When the model is too complex and performs badly on new data.",
     "M"),
    ("What is a hash table?",
     "A hash table is a data structure that maps keys to values using a hash function to compute an index into an array of buckets.",
     "A dictionary-like structure that uses hashing for fast access.",
     "M"),
    ("What is gradient descent?",
     "Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction of the negative gradient of the loss function to minimize it.",
     "It's an algorithm that tries to find the minimum of the loss by moving down the slope.",
     "M"),
    ("Explain the concept of a zombie process.",
     "A zombie process is a child process that has completed execution but still has an entry in the process table because its parent has not yet read its exit status.",
     "A process that has finished but is not fully removed from the system yet.",
     "M"),
    ("What is the difference between process and thread?",
     "A process is an independent program in execution with its own memory space, while a thread is a lightweight unit within a process that shares the process's memory.",
     "Threads are lighter than processes and share some memory.",
     "M"),
    ("What is the primary difference in memory management between C and C++?",
     "C uses malloc() and free() for raw memory management, while C++ additionally handles the invocation of constructors and destructors through new and delete.",
     "C++ has new and delete while C only has malloc.",
     "M"),

    # --- WRONG ANSWERS ---
    ("What is a segmentation fault?",
     "A segmentation fault occurs when a program attempts to access a memory location it is not allowed to access.",
     "I don't know",
     "W"),
    ("What is an AVL tree?",
     "A self-balancing BST where height difference between subtrees is at most 1.",
     "bleh blah blah something about trees",
     "W"),
    ("How does a Random Forest Classifier make predictions?",
     "It trains multiple decision trees and uses majority voting.",
     "",
     "W"),
    ("What is dynamic programming?",
     "Breaking problems into overlapping subproblems and caching solutions.",
     "pass",
     "W"),
    ("What is overfitting?",
     "A model that performs well on training data but poorly on new data.",
     "skip",
     "W"),
    ("What is gradient descent?",
     "An optimization algorithm minimizing loss by moving in the direction of the negative gradient.",
     "dont know",
     "W"),
    ("What is a hash table?",
     "A data structure using a hash function to map keys to array indices.",
     "asdfghjkl qwerty nothing",
     "W"),
    ("What is a zombie process?",
     "A child process that has exited but whose exit status has not been read by the parent.",
     "no idea",
     "W"),
    ("What is the difference between process and thread?",
     "Processes are independent with own memory; threads share memory within a process.",
     "x y z a b c",
     "W"),
    ("What is memory management in C vs C++?",
     "C uses malloc/free; C++ uses new/delete which also call constructors/destructors.",
     "they are different",
     "W"),
]


def load_model_and_tokenizer(adapter_path):
    print(f"Loading adapter from: {adapter_path}")
    adapter_cfg = json.load(open(Path(adapter_path) / "adapter_config.json"))
    base_model_name = adapter_cfg["base_model_name_or_path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    if device == "cuda":
        model = model.to("cuda")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def score_answer(model, tokenizer, question, reference, candidate):
    clean = candidate.strip().lower()
    if clean in ["i don't know", "i dont know", "dont know", "don't know", "no idea", "pass", "skip", ""]:
        return 0
    gibberish = ["bleh blah blah", "asdfghjkl", "qwerty", "x y z", "nothing"]
    if any(g in clean for g in gibberish):
        return 0
    if clean == reference.strip().lower() or reference.strip().lower() in clean:
        return 10
    if len(clean.split()) < 3:
        return 0

    prompt = (
        "You are a strict technical interviewer.\n\n"
        f"Question: {question}\n"
        f"Ideal Answer: {reference}\n"
        f"Candidate Answer: {candidate}\n\n"
        "Evaluation Rubric:\n"
        "- 0 marks: gibberish, completely wrong, or empty.\n"
        "- 1 to 4 marks: vague, lacking detail, or mostly incorrect.\n"
        "- 5 to 7 marks: partially correct but missing key concepts.\n"
        "- 8 to 10 marks: conceptually accurate and complete.\n\n"
        "Score: "
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    nums = re.findall(r"\b\d+\b", resp)
    if nums:
        return min(max(int(nums[0]), 0), 10)
    return 0


def classify_tier(score):
    if score >= 8:
        return "P"
    if score >= 4:
        return "M"
    return "W"


def evaluate_gold_standard(model, tokenizer):
    print("\n--- EVALUATING 30-CASE GOLD STANDARD BENCHMARK ---")
    correct = 0
    errors = []
    tier_counts = defaultdict(lambda: {"total": 0, "correct": 0})

    for q, ref, cand, expected_tier in GOLD_STANDARD:
        pred_score = score_answer(model, tokenizer, q, ref, cand)
        pred_tier = classify_tier(pred_score)
        is_correct = (pred_tier == expected_tier)
        if is_correct:
            correct += 1
        tier_counts[expected_tier]["total"] += 1
        if is_correct:
            tier_counts[expected_tier]["correct"] += 1
        expected_mid = 9 if expected_tier == "P" else (5.5 if expected_tier == "M" else 1.0)
        errors.append(abs(pred_score - expected_mid))

    acc = (correct / len(GOLD_STANDARD)) * 100
    mae = sum(errors) / len(errors)

    print(f"  30-Case Benchmark Accuracy: {acc:.1f}% ({correct}/{len(GOLD_STANDARD)})")
    print(f"  Overall MAE:                {mae:.2f}")
    for tier in ["P", "M", "W"]:
        t_tot = tier_counts[tier]["total"]
        t_cor = tier_counts[tier]["correct"]
        t_acc = (t_cor / t_tot * 100) if t_tot else 0
        name = "Perfect" if tier == "P" else ("Partial" if tier == "M" else "Wrong")
        print(f"    Tier {name:<7} ({tier}): {t_acc:.1f}% ({t_cor}/{t_tot})")

    return {"accuracy": acc, "mae": mae, "correct": correct, "total": len(GOLD_STANDARD)}


def evaluate_test_set(model, tokenizer, max_samples=None):
    print("\n--- EVALUATING HELD-OUT TEST SET (469 QUESTIONS) ---")
    records = [json.loads(line) for line in open(TEST_JSONL, encoding="utf-8") if line.strip()]
    if max_samples and max_samples < len(records):
        records = records[:max_samples]

    print(f"Evaluating on {len(records)} test examples ...")

    domain_stats = defaultdict(lambda: {"correct_tier": 0, "total": 0, "errors": [], "sq_errors": []})
    total_correct_tier = 0
    all_errors = []
    all_sq_errors = []

    t0 = time.time()
    for idx, r in enumerate(records, 1):
        q = r["question"]
        ref = r["reference"]
        cand = r["candidate"]
        true_score = r["score"]
        expected_tier = r["tier"]
        dom = r["domain"]

        pred_score = score_answer(model, tokenizer, q, ref, cand)
        pred_tier = classify_tier(pred_score)
        is_tier_correct = (pred_tier == expected_tier)

        err = abs(pred_score - true_score)
        sq_err = (pred_score - true_score) ** 2

        if is_tier_correct:
            total_correct_tier += 1
            domain_stats[dom]["correct_tier"] += 1

        domain_stats[dom]["total"] += 1
        domain_stats[dom]["errors"].append(err)
        domain_stats[dom]["sq_errors"].append(sq_err)

        all_errors.append(err)
        all_sq_errors.append(sq_err)

        if idx % 50 == 0 or idx == len(records):
            elapsed = time.time() - t0
            sys.stdout.write(f"\r  Progress: {idx}/{len(records)} ({idx/len(records)*100:.1f}%) in {elapsed:.1f}s")
            sys.stdout.flush()

    print()
    overall_tier_acc = (total_correct_tier / len(records)) * 100
    overall_mae = sum(all_errors) / len(all_errors)
    overall_rmse = math.sqrt(sum(all_sq_errors) / len(all_sq_errors))

    print(f"\nTest Set Overall Metrics ({len(records)} samples):")
    print(f"  Tier Accuracy: {overall_tier_acc:.2f}%")
    print(f"  MAE:           {overall_mae:.2f}")
    print(f"  RMSE:          {overall_rmse:.2f}")

    print("\nDomain-Level Breakdown:")
    domain_report = {}
    for dom, stats in sorted(domain_stats.items()):
        d_acc = (stats["correct_tier"] / stats["total"]) * 100
        d_mae = sum(stats["errors"]) / len(stats["errors"])
        d_rmse = math.sqrt(sum(stats["sq_errors"]) / len(stats["sq_errors"]))
        print(f"  {dom:<30}: Tier Acc = {d_acc:.1f}% ({stats['correct_tier']}/{stats['total']}), MAE = {d_mae:.2f}, RMSE = {d_rmse:.2f}")
        domain_report[dom] = {"tier_acc": d_acc, "mae": d_mae, "rmse": d_rmse, "total": stats["total"]}

    return {
        "overall_tier_acc": overall_tier_acc,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "domains": domain_report,
        "total_samples": len(records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.adapter_path)
    gold_res = evaluate_gold_standard(model, tokenizer)
    test_res = evaluate_test_set(model, tokenizer, max_samples=args.max_test_samples)

    gold_acc_str = f"{gold_res['accuracy']:.1f}%"
    test_acc_str = f"{test_res['overall_tier_acc']:.1f}%"
    mae_str = f"{test_res['overall_mae']:.2f}"
    rmse_str = f"{test_res['overall_rmse']:.2f}"

    print("\n" + "=" * 60)
    print("           FINAL MODEL PERFORMANCE REPORT")
    print("=" * 60)
    print(f"{'Metric':<30} {'Before (Baseline)':<20} {'After (Fine-Tuned 5x)':<20}")
    print("-" * 60)
    print(f"{'30-case Benchmark Acc':<30} {'63.3%':<20} {gold_acc_str:<20}")
    print(f"{'Test Set Tier Accuracy':<30} {'--':<20} {test_acc_str:<20}")
    print(f"{'Test Set MAE':<30} {'--':<20} {mae_str:<20}")
    print(f"{'Test Set RMSE':<30} {'--':<20} {rmse_str:<20}")
    print("=" * 60)


if __name__ == "__main__":
    main()
