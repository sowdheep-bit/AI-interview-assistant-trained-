"""
train_grading_lora.py
======================
Fine-tunes Qwen1.5-1.8B + LoRA on the actual grading task, using the
EXACT prompt template that evaluate_model.py's score_answer() uses at
inference time. Uses completion-only loss masking (via trl's
DataCollatorForCompletionOnlyLM) so gradient only flows through the score
token(s) after "Score: ", not through the re-generated question/rubric text.

Also re-excludes the 30 GOLD_STANDARD questions as a defense-in-depth check
(generate_grading_dataset.py already excludes them, but this guards against
stale data files).

Usage:
    python scripts/train_grading_lora.py --epochs 3 --rank 16
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_JSONL = ROOT / "data" / "grading_train.jsonl"
EVAL_SCRIPT = ROOT / "scripts" / "evaluate_model.py"
BASE_MODEL = "Qwen/Qwen1.5-1.8B"

# ── MUST stay byte-identical to the template in evaluate_model.py's
# score_answer(), or we reintroduce the train/inference mismatch this
# whole retrain is meant to fix. ────────────────────────────────────────
PROMPT_TEMPLATE = (
    "You are a strict technical interviewer.\n\n"
    "Question: {question}\n"
    "Ideal Answer: {reference}\n"
    "Candidate Answer: {candidate}\n\n"
    "Evaluation Rubric:\n"
    "- 0 marks: gibberish, completely wrong, or empty.\n"
    "- 1 to 4 marks: vague, lacking detail, or mostly incorrect.\n"
    "- 5 to 7 marks: partially correct but missing key concepts.\n"
    "- 8 to 10 marks: conceptually accurate and complete.\n\n"
    "Score: "
)
RESPONSE_TEMPLATE = "Score: "  # collator masks everything before this


PREFIX_RE = re.compile(
    r"^(Candidate prompt|Could you explain this|Define and explain"
    r"|Interview Question|Please answer the following)\s*:\s*",
    re.IGNORECASE,
)


def norm_q(q: str) -> str:
    return PREFIX_RE.sub("", str(q).strip()).strip().lower()


def load_gold_standard_questions() -> set:
    src = EVAL_SCRIPT.read_text()
    m = re.search(r"GOLD_STANDARD\s*=\s*\[(.*?)\n\]\n", src, re.DOTALL)
    if not m:
        raise RuntimeError("Could not find GOLD_STANDARD list in evaluate_model.py")
    qs = re.findall(r'\(\s*"((?:[^"\\]|\\.)*)"\s*,', m.group(1))
    return {norm_q(q) for q in qs}


def build_dataset(tokenizer, validation_fraction=0.15):
    from datasets import Dataset

    gold_qs = load_gold_standard_questions()
    records = [json.loads(l) for l in open(DATA_JSONL, encoding="utf-8")]

    n_before = len(records)
    records = [r for r in records if norm_q(r["question"]) not in gold_qs]
    n_leaked = n_before - len(records)
    if n_leaked:
        print(f"WARNING: dropped {n_leaked} training examples that matched "
              f"GOLD_STANDARD questions (leak guard triggered).")
    if not records:
        sys.exit(f"No training records found in {DATA_JSONL}. Run "
                  f"generate_grading_dataset.py first.")

    def to_text(r):
        prompt = PROMPT_TEMPLATE.format(
            question=r["question"], reference=r["reference"], candidate=r["candidate"]
        )
        # target is just the integer score; EOS lets the model learn to stop
        full = prompt + str(int(r["score"])) + tokenizer.eos_token
        return {"text": full}

    # Split *questions*, rather than answer rows.  Each question has perfect,
    # partial and wrong variants; a row-wise split leaks the same reference
    # answer into validation and makes validation accuracy look better than it
    # is for unseen interview questions.
    questions = sorted({norm_q(r["question"]) for r in records})
    rng = __import__("random").Random(42)
    rng.shuffle(questions)
    n_val_questions = max(1, round(len(questions) * validation_fraction))
    val_questions = set(questions[:n_val_questions])
    train_records = [r for r in records if norm_q(r["question"]) not in val_questions]
    val_records = [r for r in records if norm_q(r["question"]) in val_questions]

    train_ds = Dataset.from_list(train_records).map(to_text)
    val_ds = Dataset.from_list(val_records).map(to_text)
    print(
        f"Train: {len(train_ds)} examples / {len(questions) - n_val_questions} questions; "
        f"validation: {len(val_ds)} examples / {n_val_questions} unseen questions"
    )
    return train_ds, val_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    ap.add_argument("--out-dir", default=str(ROOT / "models" / "fine_tuned_interviewer_v2"))
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj",
                     help="comma-separated attention projections to adapt")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not 0 < args.validation_fraction < 0.5:
        sys.exit("--validation-fraction must be between 0 and 0.5")
    train_ds, val_ds = build_dataset(tokenizer, args.validation_fraction)

    # Sanity check: response template must be tokenizable/findable, or the
    # collator will silently mask the whole example (loss = garbage).
    probe = train_ds[0]["text"]
    if RESPONSE_TEMPLATE not in probe:
        sys.exit("RESPONSE_TEMPLATE not found in a built example — template mismatch.")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map="auto" if torch.cuda.is_available() else "cpu"
    )

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=args.target_modules.split(","),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer
    )

    sft_config = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"\nSaved adapter to {args.out_dir}")
    print("Next: point evaluate_model.py's MODEL_PATH at this dir (or copy "
          "it over models/fine_tuned_interviewer) and re-run the gold-standard eval.")


if __name__ == "__main__":
    main()
