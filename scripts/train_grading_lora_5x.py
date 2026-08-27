"""
train_grading_lora_5x.py
========================
Fine-tunes Qwen1.5-1.8B + LoRA on the 5x interview grading dataset using
completion-only loss masking and GPU-optimized FP16 precision.

Outputs to: models/fine_tuned_interviewer_5x/
Preserves:  models/fine_tuned_interviewer/ (known-good baseline)
"""

import argparse
import json
import os
import re
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN_JSONL = ROOT / "data" / "grading_train_5x.jsonl"
VAL_JSONL = ROOT / "data" / "grading_val_5x.jsonl"
BASE_MODEL = "Qwen/Qwen1.5-1.8B"
DEFAULT_OUT_DIR = str(ROOT / "models" / "fine_tuned_interviewer_5x")

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
RESPONSE_TEMPLATE = "Score: "


class CompletionOnlyDataCollator:
    """Masks out prompt and rubric tokens (-100) so loss only computes on the score."""
    def __init__(self, tokenizer, response_template="Score: ", max_length=512):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.max_length = max_length

    def __call__(self, examples):
        texts = [e["text"] for e in examples]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        for i, input_ids in enumerate(batch["input_ids"]):
            input_list = input_ids.tolist()
            found = False
            for idx in range(len(input_list) - len(self.response_template_ids) + 1):
                if input_list[idx : idx + len(self.response_template_ids)] == self.response_template_ids:
                    labels[i, : idx + len(self.response_template_ids)] = -100
                    found = True
                    break
            if not found:
                labels[i, :] = -100
            labels[i, batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        return batch


def load_dataset(tokenizer, max_train_samples=None, max_val_samples=None):
    from datasets import Dataset

    print(f"Loading training data from {TRAIN_JSONL} ...")
    train_records = [json.loads(line) for line in open(TRAIN_JSONL, encoding="utf-8") if line.strip()]
    val_records = [json.loads(line) for line in open(VAL_JSONL, encoding="utf-8") if line.strip()]

    if max_train_samples and max_train_samples < len(train_records):
        train_records = train_records[:max_train_samples]
    if max_val_samples and max_val_samples < len(val_records):
        val_records = val_records[:max_val_samples]

    def format_example(r):
        prompt = PROMPT_TEMPLATE.format(
            question=r["question"], reference=r["reference"], candidate=r["candidate"]
        )
        full_text = prompt + str(int(r["score"])) + tokenizer.eos_token
        return {"text": full_text}

    train_ds = Dataset.from_list(train_records).map(format_example)
    val_ds = Dataset.from_list(val_records).map(format_example)

    print(f"Formatted {len(train_ds)} train examples and {len(val_ds)} validation examples.")
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--sanity-check", action="store_true", help="Run 5 steps to verify GPU memory & loss")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = load_dataset(
        tokenizer,
        max_train_samples=40 if args.sanity_check else args.max_train_samples,
        max_val_samples=20 if args.sanity_check else args.max_val_samples,
    )

    print(f"Loading base model {BASE_MODEL} in FP16 directly to {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    if device == "cuda":
        model = model.to("cuda")
        model.gradient_checkpointing_enable()

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

    collator = CompletionOnlyDataCollator(
        tokenizer=tokenizer,
        response_template=RESPONSE_TEMPLATE,
        max_length=512,
    )

    max_steps = 10 if args.sanity_check else -1
    epochs = 1 if args.sanity_check else args.epochs

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=(device == "cuda"),
        remove_unused_columns=False,
        logging_steps=1 if args.sanity_check else 50,
        eval_strategy="no" if args.sanity_check else "steps",
        eval_steps=200,
        save_strategy="no" if args.sanity_check else "steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=False if args.sanity_check else True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
    }
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)

    print("\nStarting Training ...")
    train_result = trainer.train()

    if not args.sanity_check:
        print(f"\nSaving fine-tuned adapter to {args.out_dir} ...")
        trainer.save_model(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)
        print(f"Adapter saved successfully to: {args.out_dir}")
    else:
        print("\nSANITY CHECK PASSED: Forward & backward pass completed cleanly without OOM.")


if __name__ == "__main__":
    main()
