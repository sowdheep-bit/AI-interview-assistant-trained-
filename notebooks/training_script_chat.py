"""
Training Script for Qwen1.5-1.8B-Chat with LoRA Fine-Tuning
=============================================================
This replaces the main training cell in training_notebook.ipynb.
Run this in Google Colab (with GPU) or locally with a CUDA GPU.

Usage (Colab):
  1. Upload LLM_Interview_Dataset-v2.csv
  2. Run: !pip install transformers peft trl torch pandas datasets accelerate
  3. Run this script: !python training_script_chat.py

Usage (Local):
  1. Ensure LLM_Interview_Dataset-v2.csv is in the same directory
  2. pip install transformers peft trl torch pandas datasets accelerate
  3. python training_script_chat.py
"""

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# 1. Load Base Model and Tokenizer (moved up so we can use apply_chat_template)
model_id = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 2. Load and format dataset using ChatML template
df = pd.read_csv("LLM_Interview_Dataset-v2.csv")


def format_chat(row):
    """Format each row as a ChatML conversation for the Chat model."""
    messages = [
        {
            "role": "system",
            "content": "You are an expert technical interviewer. Given a domain, difficulty level, and question, provide the ideal reference answer.",
        },
        {
            "role": "user",
            "content": f"Domain: {row['Domain']}\nDifficulty: {row['Difficulty_Level']}\nQuestion: {row['Question']}",
        },
        {"role": "assistant", "content": row["Reference_Answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


df["text"] = df.apply(format_chat, axis=1)
dataset = Dataset.from_pandas(df[["text"]])

# 3. Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", dtype=torch.float16
)

# 4. Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 5. Set Arguments via SFTConfig (max_length increased for chat template tokens)
sft_config = SFTConfig(
    output_dir="./results",
    dataset_text_field="text",
    max_length=512,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
)

# 6. Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()

# 7. Save fine-tuned weights
trainer.model.save_pretrained("./fine_tuned_interviewer")
tokenizer.save_pretrained("./fine_tuned_interviewer")
print("Training complete! Download the './fine_tuned_interviewer' folder.")
