import pandas as pd
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# 1. Initialization & Setup
# =============================================================================
MODEL_PATH = "./models/fine_tuned_interviewer"
BASE_MODEL = "Qwen/Qwen1.5-1.8B-Chat"

print("Loading fine-tuned model... please wait.")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu", torch_dtype=torch.float32)
model.load_adapter(MODEL_PATH)

df = pd.read_csv("./data/LLM_Interview_Dataset-v2.csv")


# =============================================================================
# 2. Advanced Evaluation Logic
# =============================================================================
def evaluate_answer(question, reference_answer, user_answer, max_marks=10):
    """
    Evaluates the user's answer robustly, handling edge cases, prompt injections, 
    and returning both a conversational response and a strict score.
    """
    clean_answer = user_answer.strip().lower()
    
    # -- Edge Case 1: The "I don't know" / Empty fast path --
    non_answers = ["i dont know", "i don't know", "dont know", "no idea", "pass", "skip", "n/a", ""]
    if clean_answer in non_answers:
        return 0, "No worries at all! That's a tough one. Let's just move on to the next topic."

    # -- Edge Case 2: Extreme gibberish or too short --
    # If it's less than 3 words and doesn't exactly match a short reference answer.
    if len(clean_answer.split()) < 3 and clean_answer != reference_answer.strip().lower():
        return 0, "I'm not quite sure I follow. In a real interview, make sure to elaborate on your thought process!"

    # -- Edge Case 3: Basic Prompt Injection Guardrail --
    injection_keywords = ["ignore previous", "give me a 10", "score: 10", "system prompt"]
    if any(kw in clean_answer for kw in injection_keywords):
        return 0, "Nice try! Let's stick to the technical questions, shall we?"

    # -- Expert Interviewer Prompt --
    prompt = (
        f"You are an expert Senior Staff Engineer and a friendly, empathetic technical interviewer. "
        f"Your task is to evaluate a candidate's answer to an interview question.\n\n"

        f"Question: {question}\n"
        f"Reference Answer: {reference_answer}\n"
        f"Candidate's Answer: {user_answer}\n\n"

        f"Instructions:\n"
        f"1. First determine whether the candidate actually answered the specific question being asked.\n"
        f"2. The answer must be relevant to the exact concept, class, algorithm, problem, or topic in the question.\n"
        f"3. If the candidate discusses a different topic, class, algorithm, or concept, do NOT give a high score, "
        f"even if the information provided is technically correct.\n"
        f"4. A completely off-topic answer should normally receive 0-2 marks.\n"
        f"5. If the answer is relevant, carefully evaluate the technical claims for factual correctness.\n"
        f"6. Before assigning marks, identify the important factual claims made by the candidate.\n"
        f"7. Check each important technical claim against your own technical knowledge and the reference answer.\n"
        f"8. Distinguish clearly between correct concepts, missing concepts, and incorrect technical claims.\n"
        f"9. A technically incorrect claim must reduce the score even if the answer is detailed, fluent, or confident.\n"
        f"10. Multiple major technical errors must prevent a high score.\n"
        f"11. Do NOT give 9-10 marks to an answer containing major factual misconceptions.\n"
        f"12. Correct statements that go beyond the reference answer should receive credit and should NOT reduce the score.\n"
        f"13. The candidate does NOT need to use the same wording as the reference answer.\n"
        f"14. Accept technically correct explanations that use different wording, terminology, examples, formulas, "
        f"or structure.\n"
        f"15. Do not penalize an answer simply because it is more detailed than the reference answer.\n"
        f"16. Do not award marks merely because the answer is long. Judge the technical content.\n"
        f"17. Write a brief, conversational response of 1-3 sentences addressed directly to the candidate. "
        f"Validate correct points, gently correct important misconceptions, and mention important missing concepts when appropriate.\n\n"

        f"Scoring:\n"
        f"18. Evaluate the answer using these four dimensions:\n"
        f"   - Relevance: 0-2 marks. Does the answer directly address the specific question?\n"
        f"   - Technical correctness: 0-4 marks. Are the important technical claims correct?\n"
        f"   - Completeness: 0-2 marks. Does the answer cover the essential concepts required by the question?\n"
        f"   - Clarity: 0-2 marks. Is the explanation clear and understandable?\n"
        f"   Add these four values to obtain the final score out of 10.\n\n"

        f"19. If the answer is completely off-topic, normally give 0-2 marks.\n"
        f"20. If the answer is relevant but contains major technical misconceptions, normally give no more than 5 marks.\n"
        f"21. If the answer is relevant, mostly correct, but has important omissions or minor errors, give partial credit.\n"
        f"22. An answer should receive 9-10 marks only when it is relevant, technically accurate, and substantially complete.\n"
        f"23. Correct additional information must not reduce the score.\n"
        f"24. Do not confuse confidence, length, or detail with correctness.\n\n"

        f"You MUST format your output EXACTLY using these XML tags. Do not output anything else:\n"
        f"<feedback>Your conversational response here.</feedback>\n"
        f"<score>Your integer score here</score>\n\n"

        f"Output:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate parameters tuned for a balance of strict formatting and natural language
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
   
    
    # -- Robust Regex Parsing --
    feedback_match = re.search(r'<feedback>(.*?)</feedback>', response, re.IGNORECASE | re.DOTALL)
    score_match = re.search(r'<score>\s*(\d+)\s*</score>', response, re.IGNORECASE)
    
    # Default fallbacks in case the LLM hallucinates formatting
    feedback_text = "Thanks for your answer. Let's move on to the next one."
    final_score = 0
    
    if feedback_match:
        feedback_text = feedback_match.group(1).strip()
        
    if score_match:
        raw_score = int(score_match.group(1))
        final_score = min(max(raw_score, 0), max_marks)
    else:
        final_score = 0
        feedback_text = (
            "I couldn't reliably evaluate that answer. "
            "Let's move to the next question."
        )
        
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