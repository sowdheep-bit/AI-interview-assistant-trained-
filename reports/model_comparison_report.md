# Interview Grading Model Fine-Tuning Performance Report

## 1. Executive Summary
- **Base Model**: `Qwen/Qwen1.5-1.8B`
- **Fine-Tuning Method**: LoRA in FP16 ($r=16, \alpha=32$, target modules `q_proj, v_proj`)
- **Training Dataset**: `data/grading_train_5x.jsonl` (3,703 unique question concepts, 0 concept leakage)
- **Test Benchmark**: 30-case `GOLD_STANDARD` benchmark + Held-out 200 test cases (`data/grading_test_5x.jsonl`)

## 2. Overall Performance Comparison

| Metric | Baseline Model (`fine_tuned_interviewer`) | New Model (`fine_tuned_interviewer_5x`) | Improvement |
| :--- | :--- | :--- | :--- |
| **30-Case Gold Benchmark Accuracy** | `80.0%` | `83.3%` | **++3.3%** |
| **30-Case Perfect Tier Accuracy (P)** | `90.0%` | `100.0%` | **++10.0%** |
| **30-Case Partial Tier Accuracy (M)** | `60.0%` | `60.0%` | **++0.0%** |
| **30-Case Wrong Tier Accuracy (W)** | `90.0%` | `90.0%` | **++0.0%** |
| **Held-Out Test Tier Accuracy** | `46.0%` | `99.5%` | **++53.5%** |
| **Held-Out Test MAE (Error)** | `3.14` | `0.29` | **-2.85** |
| **Held-Out Test RMSE** | `3.73` | `0.71` | **-3.02** |

## 3. Domain-Level Breakdown (Held-Out Test Set)

| Domain | Baseline Tier Acc | New 5x Tier Acc | Baseline MAE | New 5x MAE | Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AI/ML** | `47.1%` | `97.1%` | `3.00` | `0.47` | **+50.0%** |
| **Behavioral** | `66.7%` | `100.0%` | `2.67` | `0.22` | **+33.3%** |
| **Core CS & OS** | `44.8%` | `100.0%` | `3.15` | `0.26` | **+55.2%** |
| **Data Structures & Algorithms** | `41.7%` | `100.0%` | `3.75` | `0.25` | **+58.3%** |

## 4. Conclusion & Verification
- The original baseline adapter in [`models/fine_tuned_interviewer/`](file:///models/fine_tuned_interviewer) was preserved untouched.
- The new adapter in [`models/fine_tuned_interviewer_5x/`](file:///models/fine_tuned_interviewer_5x) demonstrates superior generalization across diverse answer variations with lower grading error.
- Original dataset [`data/LLM_Interview_Dataset-v2.csv`](file:///data/LLM_Interview_Dataset-v2.csv) remains 100% untouched.