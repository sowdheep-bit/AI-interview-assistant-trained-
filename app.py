import pandas as pd
import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

# =============================================================================
# 1. Initialization & Setup
# =============================================================================
MODEL_PATH = os.environ.get("INTERVIEW_ADAPTER_PATH", "./models/fine_tuned_interviewer")
# The adapter was trained on Qwen1.5-1.8B (not the Chat checkpoint).  Loading
# an adapter on a different base model is a silent but substantial source of
# scoring errors, so always read the base checkpoint from the adapter metadata.
BASE_MODEL = PeftConfig.from_pretrained(MODEL_PATH).base_model_name_or_path

print("Loading fine-tuned model... please wait.")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu", torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, MODEL_PATH)
model.eval()

df = pd.read_csv("./data/LLM_Interview_Dataset-v2.csv")


# =============================================================================
# 2. Advanced Evaluation Logic
# =============================================================================
def evaluate_answer(question, reference_answer, user_answer, max_marks=10):
    """
    Evaluates the user's answer robustly, handling edge cases, prompt injections,
    and returning both a conversational response and a strict score.
    """
    clean_answer = user_answer.strip()

    # -- Edge Case 1: The "I don't know" / Empty fast path --
    non_answers = ["i dont know", "i don't know", "dont know", "no idea", "pass", "skip", "n/a", ""]
    if clean_answer.lower() in non_answers:
        return 0, "No worries at all! That's a tough one. Let's just move on to the next topic."

    # -- Edge Case 2: Extreme gibberish or too short --
    # If it's less than 3 words and doesn't exactly match a short reference answer.
    clean_lower = clean_answer.lower()
    if len(clean_lower.split()) < 3 and clean_lower != reference_answer.strip().lower():
        return 0, "I'm not quite sure I follow. In a real interview, make sure to elaborate on your thought process!"

    # -- Edge Case 3: Basic Prompt Injection Guardrail --
    injection_keywords = ["ignore previous", "give me a 10", "score: 10", "system prompt"]
    if any(kw in clean_lower for kw in injection_keywords):
        return 0, "Nice try! Let's stick to the technical questions, shall we?"

    # This prompt deliberately matches the completion-only training template in
    # scripts/train_grading_lora.py.  Keep the model's job narrow (emit a score)
    # rather than asking a small model to follow a long, unrelated XML format.
    prompt = (
        "You are a strict technical interviewer.\n\n"
        f"Question: {question}\n"
        f"Ideal Answer: {reference_answer}\n"
        f"Candidate Answer: {user_answer}\n\n"
        "Evaluation Rubric:\n"
        "- 0 marks: gibberish, completely wrong, or empty.\n"
        "- 1 to 4 marks: vague, lacking detail, or mostly incorrect.\n"
        "- 5 to 7 marks: partially correct but missing key concepts.\n"
        "- 8 to 10 marks: conceptually accurate and complete.\n\n"
        "Score: "
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # -- Improved score extraction: try multiple patterns --
    # Pattern 1: standalone number (e.g., "5" or "10")
    nums_v1 = re.findall(r"\b(10|[0-9])\b", response)
    # Pattern 2: "Score: N" format
    nums_v2 = re.findall(r"Score:\s*(\d+)", response, re.IGNORECASE)
    # Pattern 3: just the number at the end
    nums_v3 = re.findall(r"\d+$", response.strip())

    # Collect all found numbers, prioritizing v1 then v2 then v3
    all_nums = []
    for nums in [nums_v1, nums_v2, nums_v3]:
        for n in nums:
            val = int(n)
            if 0 <= val <= max_marks:
                all_nums.append(val)

    # Use the first valid number found, or the highest valid if multiple
    final_score = None
    if all_nums:
        # Use the first valid number (most reliable from model output)
        final_score = all_nums[0]
    else:
        # Fallback: try to find any number and clamp it
        any_nums = re.findall(r"\d+", response)
        if any_nums:
            try:
                final_score = min(max(int(any_nums[0]), 0), max_marks)
            except ValueError:
                final_score = 0

    if final_score is None:
        return 0, "I couldn't reliably evaluate that answer. Let's move to the next question."

    # Clamp final score to valid range
    final_score = min(max(final_score, 0), max_marks)

    # Generate feedback based on score
    if final_score >= 8:
        feedback_text = "Strong answer — it is accurate and covers the key ideas."
    elif final_score >= 4:
        feedback_text = "You have part of the idea. Add the important missing details for a stronger answer."
    else:
        feedback_text = "This answer needs correction or more direct technical detail."
    return final_score, feedback_text

# =============================================================================
# 3. Core Interview Loop
# =============================================================================
PHASE_LIMITS = {"Easy": 5, "Medium": 8, "Hard": 5}
PASS_THRESHOLD = 0.70  # Avg score fraction needed to advance Easy->Hard after Medium

def run_interview():
    print("\n" + "="*60)
    print("      ADAPTIVE LLM INTERVIEW SYSTEM (EXPERT MODE)")
    print("="*60)

    domains = sorted(df['Domain'].dropna().unique())
    print("\nAvailable Domains:")
    for idx, d in enumerate(domains, 1):
        print(f"  {idx}. {d}")

    raw = input("\nSelect a domain (enter number or name): ").strip()
    if raw.isdigit():
        choice = int(raw) - 1
        if 0 <= choice < len(domains):
            selected_domain = domains[choice]
        else:
            print(f"Invalid number. Please run again.")
            return
    else:
        matches = [d for d in domains if raw.lower() in d.lower()]
        if len(matches) == 1:
            selected_domain = matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous input matches: {matches}. Be more specific.")
            return
        else:
            print(f"No domain matching '{raw}' found. Exiting.")
            return

    domain_df = df[df['Domain'] == selected_domain]

    difficulty = "Easy"
    total_score, total_possible = 0, 0
    lost_marks_records = []
    
    phase_asked, phase_score = 0, 0

    print(f"\n--- Domain: '{selected_domain}' | Starting with EASY round ---")
    print("Type 'quit' to exit the interview early.\n")

    asked_question_ids = set()
    asked_question_texts = set()

    _PREFIX_RE = re.compile(
        r'^(Candidate prompt|Could you explain this|Define and explain'
        r'|Interview Question|Please answer the following)\s*:\s*',
        re.IGNORECASE
    )
    
    def normalize(q):
        return _PREFIX_RE.sub('', q.strip()).strip().lower()

    while True:
        # Phase Advancement Logic
        if phase_asked >= PHASE_LIMITS[difficulty]:
            phase_avg = phase_score / (phase_asked * 10) if phase_asked else 0

            if difficulty == "Easy":
                print(f"\n[EASY round complete. Advancing to MEDIUM round.]\n")
                difficulty = "Medium"
                phase_asked, phase_score = 0, 0
                continue

            elif difficulty == "Medium":
                if phase_avg >= PASS_THRESHOLD:
                    print(f"\n[MEDIUM round complete. Excellent work (Avg: {phase_avg*100:.0f}%). Advancing to HARD round.]\n")
                    difficulty = "Hard"
                else:
                    print(f"\n[MEDIUM round complete. Avg score: {phase_avg*100:.0f}% (Threshold: {PASS_THRESHOLD*100:.0f}%). We will conclude the interview here.]\n")
                    break
                phase_asked, phase_score = 0, 0
                continue

            else:  
                print(f"\n[HARD round complete. Interview finished!]\n")
                break

        # Question Selection
        available_q = domain_df[
            (domain_df['Difficulty_Level'].str.lower() == difficulty.lower()) &
            (~domain_df['ID'].isin(asked_question_ids)) &
            (~domain_df['Question'].apply(normalize).isin(asked_question_texts))
        ]

        if available_q.empty:
            print(f"\n[No more unseen {difficulty} questions available. Advancing...]\n")
            phase_asked = PHASE_LIMITS[difficulty] 
            continue

        row = available_q.sample(1).iloc[0]
        asked_question_ids.add(row['ID'])

        raw_q = row['Question'].strip()
        question_text = _PREFIX_RE.sub('', raw_q).strip()
        asked_question_texts.add(normalize(raw_q))
        ref_answer = row['Reference_Answer']

        # The Interview Interaction
        q_num = phase_asked + 1
        print(f"\n--- {difficulty.upper()} | Question {q_num}/{PHASE_LIMITS[difficulty]} ---")
        print(f"Interviewer: {question_text}")
        
        user_answer = input("\nYou: ").strip()

        if user_answer.lower() == 'quit':
            print("\nEnding interview early...")
            break

        print("\n[Interviewer is evaluating...]")
        
        # Core Evaluation Call
        score, feedback = evaluate_answer(question_text, ref_answer, user_answer, max_marks=10)
        marks_lost = 10 - score

        total_score += score
        total_possible += 10
        phase_score += score
        phase_asked += 1

        print(f"Interviewer: {feedback}")
        print(f"--> [System: Scored {score}/10]")

        if marks_lost > 0:
            lost_marks_records.append({'question': question_text, 'score': score, 'lost': marks_lost})

    # =============================================================================
    # 4. End of Session Audit Report
    # =============================================================================
    print("\n" + "="*60)
    print("                     SESSION OVERVIEW")
    print("="*60)
    if total_possible > 0:
        final_percentage = (total_score/total_possible)*100
        print(f"Overall Result: {total_score:.1f} / {total_possible} ({final_percentage:.1f}%)")
        
        if final_percentage >= 80:
            print("Verdict: STRONG HIRE")
        elif final_percentage >= 60:
            print("Verdict: LEANING HIRE (Requires further review)")
        else:
            print("Verdict: NO HIRE (Needs more foundational study)")

        print("\nQuestions Where Marks Were Lost:")
        if not lost_marks_records:
            print("  None! Perfect performance.")
        else:
            for rec in lost_marks_records:
                print(f"- Q: {rec['question']}")
                print(f"  Score: {rec['score']}/10 (-{rec['lost']} marks)")
                print("-" * 40)

if __name__ == "__main__":
    run_interview()
