# How It Works — AI Interview Assistant

A clear technical summary of the system's architecture, data flow, and components.

---

## Overview

This project is an **adaptive technical interview simulator** with two parts:

| Layer | Technology | Role |
|---|---|---|
| **Backend** | Python + FastAPI | Loads the LLM, manages sessions, scores answers |
| **Frontend** | Next.js (React + TypeScript) | Chat-style interview UI served in the browser |

The core intelligence is a **LoRA fine-tuned Qwen 1.5-1.8B** language model that acts as both the question-asker and the answer grader — all running **locally on your machine**, with no external AI API calls during an interview.

---

## The Model

### Base Model
`Qwen/Qwen1.5-1.8B` — a 1.8-billion-parameter causal language model downloaded automatically from HuggingFace (~4 GB).

### LoRA Adapter (`models/fine_tuned_interviewer/`)
A lightweight **Low-Rank Adaptation** layer (~12 MB) trained on top of the base model. It was fine-tuned specifically to score interview answers using this scoring rubric:

```
- 0 marks   → gibberish, completely wrong, or empty
- 1–4 marks → vague, lacking detail, or mostly incorrect
- 5–7 marks → partially correct but missing key concepts
- 8–10 marks → conceptually accurate and complete
```

The adapter targets only the `q_proj` and `v_proj` attention layers (rank = 16, alpha = 32), keeping the fine-tuned file small while meaningfully shifting the model's scoring behaviour.

---

## The Dataset (`data/LLM_Interview_Dataset-v2.csv`)

A curated CSV of interview questions with the following structure:

| Column | Description |
|---|---|
| `Domain` | Topic area (e.g., Machine Learning, Python, Databases) |
| `Difficulty_Level` | `Easy`, `Medium`, or `Hard` |
| `Question` | The interview question text |
| `Reference_Answer` | The ideal answer used during scoring |

The dataset is loaded at startup. Each interview session randomly samples questions from the chosen domain at each difficulty tier.

---

## Interview Flow

```
User selects a domain
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  EASY ROUND  (5 questions)                          │
│  Score each answer 0–10 using the LLM               │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  MEDIUM ROUND  (8 questions)                        │
│  Average score ≥ 70%  →  advance to Hard            │
│  Average score < 70%  →  interview ends here        │
└─────────────────────────────────────────────────────┘
        │  (if ≥ 70%)
        ▼
┌─────────────────────────────────────────────────────┐
│  HARD ROUND  (5 questions)                          │
└─────────────────────────────────────────────────────┘
        │
        ▼
  Session Report
  (total score, verdict, per-question breakdown)
```

**Verdict thresholds:**

| Score | Verdict |
|---|---|
| ≥ 80% | STRONG HIRE |
| 60–79% | LEANING HIRE |
| < 60% | NO HIRE |

---

## How Scoring Works (Step by Step)

When the user submits an answer, the backend does the following:

1. **Build a prompt** combining the original question, the reference (ideal) answer from the dataset, and the user's answer:
   ```
   You are a strict technical interviewer.

   Question: <question>
   Ideal Answer: <reference_answer>
   Candidate Answer: <user_answer>

   Evaluation Rubric:
   - 0 marks: gibberish, completely wrong, or empty.
   - 1 to 4 marks: vague, lacking detail, or mostly incorrect.
   - 5 to 7 marks: partially correct but missing key concepts.
   - 8 to 10 marks: conceptually accurate and complete.

   Score:
   ```

2. **Run inference** — the LoRA-adapted model generates the token(s) after `Score:`.

3. **Parse the score** — extract the integer 0–10 from the model's output.

4. **Return feedback** to the frontend — score, marks awarded, and whether the session continues or ends.

---

## Backend API (FastAPI)

The backend exposes three REST endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/domains` | `GET` | Returns the list of available domains from the dataset |
| `/api/session/start` | `POST` | Creates a new session, returns the first question |
| `/api/session/submit` | `POST` | Accepts the user's answer, scores it, returns next question or final report |

Session state is held **in memory** on the server — each session has a unique ID that the frontend sends with every request.

---

## Frontend (Next.js)

The UI has three screens managed as phases:

```
domain_select  →  interviewing  →  wrapup
```

| Screen | Component | What it does |
|---|---|---|
| Domain Select | `DomainSelect` | Calls `GET /api/domains`, shows available topics, starts session |
| Interview Room | `InterviewRoom` | Displays questions one by one, submits answers, shows live score |
| Wrap Up | `WrapUp` | Shows final score, verdict, and a full breakdown of every lost mark |

All API calls go through `frontend/lib/api.ts`, which is a thin typed wrapper around `fetch` pointing at `http://localhost:8000`.

---

## Training Pipeline (How the Model Was Made)

The adapter was trained using the following pipeline:

```
1. generate_grading_dataset.py
   └── Calls Gemini API to generate diverse candidate answers
       for each question + reference answer in the dataset.
       Outputs: data/grading_train.jsonl

2. train_grading_lora.py
   └── Fine-tunes Qwen1.5-1.8B + LoRA using HuggingFace PEFT + TRL.
       Uses completion-only loss masking — gradients only flow
       through the "Score: X" token, not the prompt.
       Outputs: models/fine_tuned_interviewer/

3. evaluate_model.py
   └── Benchmarks the adapter on a held-out 30-question gold standard.
       Generates a full HTML/PDF research report with charts.
```

The 30 gold-standard benchmark questions are **strictly excluded** from all training data to prevent leakage.

---

## Directory Reference

```
AI-interview-assistant-trained-/
│
├── frontend/                    ← Next.js web UI
│   ├── app/page.tsx             ← Root page, phase state machine
│   ├── components/
│   │   ├── DomainSelect/        ← Domain picker screen
│   │   ├── InterviewRoom/       ← Live interview chat screen
│   │   └── WrapUp/              ← Final score report screen
│   ├── lib/api.ts               ← Typed fetch wrappers for the backend
│   └── types/                   ← Shared TypeScript types
│
├── models/
│   └── fine_tuned_interviewer/  ← LoRA adapter weights (included in repo)
│
├── data/
│   └── LLM_Interview_Dataset-v2.csv  ← Question bank
│
├── scripts/
│   ├── generate_grading_dataset.py   ← Step 1: generate training data
│   ├── train_grading_lora.py         ← Step 2: fine-tune the adapter
│   └── evaluate_model.py             ← Step 3: benchmark + report
│
└── notebooks/
    └── training_notebook.ipynb       ← Google Colab version of training
```
