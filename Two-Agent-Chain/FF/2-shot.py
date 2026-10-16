#2A CHAIN 2SHOT
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# ==========================================

BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition"

CSV_PATH = os.path.join(BASE_DIR, "data", "redefinitionsCSV.csv")

# Outputs
MAIN_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "results_2achain_qwen72b_constants_2s.csv")

NR_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "nr_baseline")
os.makedirs(NR_OUTPUT_DIR, exist_ok=True)
NR_OUTPUT_PATH = os.path.join(NR_OUTPUT_DIR, "nr_results_2achain_qwen72b_constants_2s.csv")

MODEL_ID = os.path.join(BASE_DIR, "models", "Qwen2.5-72B-Instruct")

print(f"Loading Qwen 72B model from local directory: {MODEL_ID}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    max_length=None,
    truncation=True,
    return_full_text=False, 
    pad_token_id=tokenizer.eos_token_id
)

def query_local_model(system_prompt: str, user_prompt: str, do_sample=False, temperature=None) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
   
    if do_sample:
        outputs = pipe(messages, do_sample=True, temperature=temperature)
    else:
        outputs = pipe(messages, do_sample=False, temperature=0.0)

    generated = outputs[0]["generated_text"]

   
    if isinstance(generated, list):
       
        return generated[-1]["content"].strip()
    else:
        
        return str(generated).strip()

# ==========================================

def get_few_shot_examples(df, current_constant, diff_level, q_diff, k=2, is_nr=False):
    """
    Fetches 'k' examples from the dataframe that match the difficulty criteria
    but use a different constant to avoid giving away the answer.
    """
    pool = df[
        (df['constant'] != current_constant) &
        (df['difficulty_level'] == diff_level) &
        (df['question_difficulty'] == q_diff)
    ]

    sampled = pool.sample(n=min(k, len(pool)), random_state=42)

    examples_str = ""
    for _, row in sampled.iterrows():
        if not is_nr:
            examples_str += f"Redefine {row['constant']} as {row['redefinition']}.\n"
            examples_str += f"Question: {row['question_text']}\n"
            examples_str += f"Answer: {row['correct_redefined_answer']}\n\n"
        else:
            examples_str += f"Question: {row['question_text']}\n"
            examples_str += f"Answer: {row['anchored_answer']}\n\n"

    return examples_str.strip()

class PeerReviewMAS:
    def __init__(self, name):
        self.name = name
        
        # Personas
        self.solver_system_prompt = ""
        self.reviewer_system_prompt = ""

    def run_collaboration(self, constant, redefinition, question, fs_examples=""):
        
        solver_prompt = (
            f"Redefine {constant} as {redefinition}.\n"
            f"Answer the following question: {question}\n"
            f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n"
            f"End the response with the phrase 'The final answer is: ' followed only by the correct result, with no additional text or commentary."
        )
        
        solver_raw = query_local_model(self.solver_system_prompt, solver_prompt, do_sample=False)
        
        solver_clean = solver_raw
        if "The final answer is:" in solver_raw:
            solver_clean = solver_raw.split("The final answer is:")[-1].strip()
        elif "Answer:" in solver_raw:
            solver_clean = solver_raw.split("Answer:")[-1].strip()

        print(f"\n   [Agent 1 - Solver's Initial Answer]: {solver_clean}")

    
        reviewer_prompt = (
            f"Redefine {constant} as {redefinition}.\n"
            f"Answer the following question: {question}\n"
            f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n"
            f"Another solver proposed this answer: {solver_clean}\n"
            f"End the response with the phrase 'The final answer is: ' followed only by the correct result, with no additional text or commentary."
        )
        
        reviewer_raw = query_local_model(self.reviewer_system_prompt, reviewer_prompt, do_sample=False)
        
        reviewer_clean = reviewer_raw
        if "The final answer is:" in reviewer_raw:
            reviewer_clean = reviewer_raw.split("The final answer is:")[-1].strip()
        elif "Answer:" in reviewer_raw:
            reviewer_clean = reviewer_raw.split("Answer:")[-1].strip()

        print(f"   [Agent 2 - Reviewer's Consensus Answer]: {reviewer_clean}")

        return solver_raw, solver_clean, reviewer_raw, reviewer_clean

   
    def run_nr_collaboration(self, question,fs_examples=""):
      
        solver_prompt = (
            f"Answer the following question: {question}\n"
            f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n"
            f"End the response with the phrase 'The final answer is: ' followed only by the correct result, with no additional text or commentary."
        )
        
     
        solver_raw = query_local_model(self.solver_system_prompt, solver_prompt, do_sample=False)
        
        solver_clean = solver_raw
        if "The final answer is:" in solver_raw:
            solver_clean = solver_raw.split("The final answer is:")[-1].strip()
        elif "Answer:" in solver_raw:
            solver_clean = solver_raw.split("Answer:")[-1].strip()

        print(f"\n   [NR Agent 1 - Initial]: {solver_clean}")

       
        reviewer_prompt = (
            f"Answer the following question: {question}\n"
            f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n"
            f"Another solver proposed this answer: {solver_clean}\n"
            f"End the response with the phrase 'The final answer is: ' followed only by the correct result, with no additional text or commentary."
        )
        
      
        reviewer_raw = query_local_model(self.reviewer_system_prompt, reviewer_prompt, do_sample=False)
        
        reviewer_clean = reviewer_raw
        if "The final answer is:" in reviewer_raw:
            reviewer_clean = reviewer_raw.split("The final answer is:")[-1].strip()
        elif "Answer:" in reviewer_raw:
            reviewer_clean = reviewer_raw.split("Answer:")[-1].strip()

        print(f"   [NR Agent 2 - Final Consensus]: {reviewer_clean}")

        return solver_raw, solver_clean, reviewer_raw, reviewer_clean


# ==========================================

mas_system = PeerReviewMAS("Qwen-PeerReview")

try:
    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df)} questions from dataset.")
    results = []

   
    for index, row in df.iterrows():
        print(f"\n=== Question {index+1}: {row['constant']} -> {row['redefinition']} ===")

        fs_examples = get_few_shot_examples(
            df, 
            current_constant=row['constant'], 
            diff_level=row['difficulty_level'], 
            q_diff=row['question_difficulty'], 
            k=2, 
            is_nr=False
        )
        
        s_raw, s_clean, r_raw, r_clean = mas_system.run_collaboration(
            row['constant'], row['redefinition'], row['question_text'], fs_examples=fs_examples
        )
        
        results.append({
            "constant": row['constant'],
            "redefinition": row['redefinition'],
            "question": row['question_text'],
            "difficulty_level": row['difficulty_level'],
            "question_difficulty": row['question_difficulty'],
            "raw_model_output": r_raw,
            "extracted_answer": r_clean, 
            "agent1_solver_raw": s_raw,
            "agent1_solver_clean": s_clean
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(MAIN_OUTPUT_PATH, index=False)
    print(f"\nRedefinition Experiment finished. Results saved to '{MAIN_OUTPUT_PATH}'.")

except FileNotFoundError:
    print(f"Error: {CSV_PATH} not found. Please check your data folder.")
    exit()


# ==========================================

unique_questions_df = df.drop_duplicates(subset=['question_text']).copy()



print(f"Original dataset size: {len(df)}")
print(f"Unique questions to run: {len(unique_questions_df)}")

nr_results = []

print("\nRunning No-Redefinition (NR) MAS Baseline on unique questions...")


for index, row in unique_questions_df.iterrows():
    q_text = row['question_text']
    
    print(f"\nEvaluating NR Question: {q_text[:50]}...")
    fs_examples = get_few_shot_examples(
        df, 
        current_constant=row['constant'], 
        diff_level=row['difficulty_level'], 
        q_diff=row['question_difficulty'], 
        k=2, 
        is_nr=True
    )
    
   
    s_raw, s_clean, r_raw, r_clean = mas_system.run_nr_collaboration(row['question_text'],fs_examples=fs_examples)
    
    print(f"Final MAS NR Answer: {r_clean}")
    
    nr_results.append({
        "question": q_text, 
        "nr_output": r_clean, 
        "nr_raw_output": r_raw,
        "expected_nr_answer": row['anchored_answer'],
        "agent1_initial_clean": s_clean
    })

nr_df = pd.DataFrame(nr_results)
nr_df.to_csv(NR_OUTPUT_PATH, index=False)
print(f"NR Baseline Done! Saved unique results to '{NR_OUTPUT_PATH}'")
