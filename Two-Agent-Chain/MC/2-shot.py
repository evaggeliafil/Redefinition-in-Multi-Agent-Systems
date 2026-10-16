#2A CHAIN MC 2s

import os 
import torch 
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig 

# ========================================== 

BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition"

CSV_PATH = os.path.join(BASE_DIR, "data", "constantsrRED_MC.csv") 


MAIN_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "results_qwen72b_2achain_constants_mc_2s.csv") 

NR_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "nr_baseline") 
os.makedirs(NR_OUTPUT_DIR, exist_ok=True) 
NR_OUTPUT_PATH = os.path.join(NR_OUTPUT_DIR, "nr_results_qwen72b_2achain_constants_mc_2s.csv") 
NR_CSV_PATH = os.path.join(BASE_DIR, "data", "NR_constantRED_MC.csv")

MODEL_ID = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition/models/Qwen2.5-72B-Instruct" 

print(f"Loading Mixtral model from local directory: {MODEL_ID}...") 

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


def get_few_shot_examples(df, current_constant, diff_level, q_diff, k=2, is_nr=False):
    """
    Fetches 'k' examples from the dataframe and formats them as MC questions.
    """
    if not is_nr:
        pool = df[
            (df['constant'] != current_constant) &
            (df['difficulty_level'] == diff_level) &
            (df['question_difficulty'] == q_diff)
        ]

    else:
        pool = df[
            (df['constant'] != current_constant) &
            (df['question_difficulty'] == q_diff)
        ]

    sampled = pool.sample(n=min(k, len(pool)), random_state=42)

    examples_str = ""
    for _, row in sampled.iterrows():
        if not is_nr:
            examples_str += f"Redefine {row['constant']} as {row['redefinition']}.\n"
            examples_str += f"Choose A, B, C or D to answer the question:\n"
            examples_str += f"Question: {row['question_text']}\n"
            examples_str += f"A: {row['option_A']}\nB: {row['option_B']}\nC: {row['option_C']}\nD: {row['option_D']}\n"
            examples_str += f"Answer: {row['expected_mc_answer']}\n\n"
        else:
            examples_str += f"Choose A, B, C or D to answer the question:\n"
            examples_str += f"Question: {row['question_text']}\n"
            examples_str += f"A: {row['option_A']}\nB: {row['option_B']}\nC: {row['option_C']}\nD: {row['option_D']}\n"
            examples_str += f"Answer: {row['expected_nr_answer']}\n\n"

    return examples_str.strip()

class PeerReviewMAS:
    def __init__(self, name):
        self.name = name
        
        # Personas
        self.solver_system_prompt = ""
        self.reviewer_system_prompt = ""

    def run_collaboration(self, constant, redefinition, question, opt_a, opt_b, opt_c, opt_d, fs_examples=""):
        
        redef_str = f"Redefine {constant} as {redefinition}.\n"
        examples_str = ""
        if fs_examples:
            examples_str = f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n\n"

        question_str = (
            f"Choose A, B, C or D to answer the question:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n"
        )

        constraint_str = (
            f"Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            f"End the response with the phrase 'The final answer is: ' followed by the correct letter, with no additional text or commentary."
        )
        full_prompt = redef_str + question_str + examples_str + constraint_str
        solver_raw = query_local_model(self.solver_system_prompt, full_prompt)
        
        solver_clean = solver_raw
        if "The final answer is:" in solver_raw:
            solver_clean = solver_raw.split("The final answer is:")[-1].strip().rstrip('.')
        elif "Answer:" in solver_raw:
            solver_clean = solver_raw.split("Answer:")[-1].strip().rstrip('.')

        print(f"\n   [Agent 1 - Solver's Initial Answer]: {solver_clean}")

        
        reviewer_question = (
            f"Choose A, B, C or D to answer the question:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n"
            
         )

        constraint_str = (
            f"Another solver proposed this answer: {solver_clean}\n"
            f"Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            f"End the response with the phrase 'The final answer is: ' followed by the correct letter, with no additional text or commentary."
        )
        reviewer_prompt = redef_str + reviewer_question + examples_str + constraint_str
        
        reviewer_raw = query_local_model(self.reviewer_system_prompt, reviewer_prompt)
        
        reviewer_clean = reviewer_raw
        if "The final answer is:" in reviewer_raw:
            reviewer_clean = reviewer_raw.split("The final answer is:")[-1].strip().rstrip('.')
        elif "Answer:" in reviewer_raw:
            reviewer_clean = reviewer_raw.split("Answer:")[-1].strip().rstrip('.')

        print(f"   [Agent 2 - Reviewer's Consensus Answer]: {reviewer_clean}")

        return solver_raw, solver_clean, reviewer_raw, reviewer_clean

    def run_nr_collaboration(self, question, opt_a, opt_b, opt_c, opt_d, fs_examples=""):
       
        examples_str = ""
        if fs_examples:
            examples_str = f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n\n"

        solver_question = (
            f"Choose A, B, C or D to answer the question:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n"
        )
        constraint_str = (
            f"Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            f"End the response with the phrase 'The final answer is: ' followed by the correct letter, with no additional text or commentary."
        )
        solver_prompt = solver_question + examples_str + constraint_str
        
        solver_raw = query_local_model(self.solver_system_prompt, solver_prompt)
        
        solver_clean = solver_raw
        if "The final answer is:" in solver_raw:
            solver_clean = solver_raw.split("The final answer is:")[-1].strip().rstrip('.')
        elif "Answer:" in solver_raw:
            solver_clean = solver_raw.split("Answer:")[-1].strip().rstrip('.')

        print(f"\n   [NR Agent 1 - Initial]: {solver_clean}")

       
        reviewer_question = (
            f"Choose A, B, C or D to answer the question:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n"
            
            )
        constraint_str = (
            f"Another solver proposed this answer: {solver_clean}\n"
            f"Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            f"End the response with the phrase 'The final answer is: ' followed by the correct letter, with no additional text or commentary."
        )
        
        reviewer_prompt = reviewer_question + examples_str + constraint_str
        
        
        reviewer_raw = query_local_model(self.reviewer_system_prompt, reviewer_prompt)
        
        reviewer_clean = reviewer_raw
        if "The final answer is:" in reviewer_raw:
            reviewer_clean = reviewer_raw.split("The final answer is:")[-1].strip().rstrip('.')
        elif "Answer:" in reviewer_raw:
            reviewer_clean = reviewer_raw.split("Answer:")[-1].strip().rstrip('.')

        print(f"   [NR Agent 2 - Final Consensus]: {reviewer_clean}")

        return solver_raw, solver_clean, reviewer_raw, reviewer_clean


mas_system = PeerReviewMAS("Qwen-PeerReview-MC")

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
        print(fs_examples)
        s_raw, s_clean, r_raw, r_clean = mas_system.run_collaboration(
            row['constant'], row['redefinition'], row['question_text'],
            row['option_A'], row['option_B'], row['option_C'], row['option_D'],fs_examples=fs_examples
        )
        
        results.append({
            "constant": row['constant'],
            "redefinition": row['redefinition'],
            "question": row['question_text'],
            "difficulty_level": row['difficulty_level'],
            "question_difficulty": row['question_difficulty'],
            "option_A": row['option_A'],
            "option_B": row['option_B'],
            "option_C": row['option_C'],
            "option_D": row['option_D'],
            "expected_mc_answer": row['expected_mc_answer'],
            "anchored_mc_answer": row['anchored_mc_answer'],
            "agent1_solver_raw": s_raw,
            "agent1_solver_clean": s_clean,
            "raw_model_output": r_raw,
            "extracted_answer": r_clean
        })
    results_df = pd.DataFrame(results) 
    results_df.to_csv(MAIN_OUTPUT_PATH, index=False) 
    print(f"\nExperiment finished. Redefined results saved to {MAIN_OUTPUT_PATH}") 

    

except FileNotFoundError:
    print(f"Error: CSV not found at {CSV_PATH}. Please check the path.")



try:
    nr_df_input = pd.read_csv(NR_CSV_PATH)
    print(f"\nOriginal NR dataset size: {len(nr_df_input)}")
    print("\nRunning No-Redefinition (NR) MAS Baseline...")

    nr_results = []

    for index, row in nr_df_input.iterrows():
        q_text = row['question_text']
        fs_examples = get_few_shot_examples(
            nr_df_input,
            current_constant=row['constant'],
            diff_level="", 
            q_diff=row['question_difficulty'],
            k=2,
            is_nr=True
        )
        print(f"\nEvaluating NR Question: {q_text[:50]}...")
        
        s_raw, s_clean, r_raw, r_clean = mas_system.run_nr_collaboration(
            q_text, 
            row['option_A'], row['option_B'], row['option_C'], row['option_D'], fs_examples=fs_examples
        )
        
        nr_results.append({
            "constant": row['constant'],
            "question": q_text, 
            "option_A": row['option_A'],
            "option_B": row['option_B'],
            "option_C": row['option_C'],
            "option_D": row['option_D'],
            "expected_nr_answer": row['expected_nr_answer'],
            "agent1_initial_clean": s_clean,
            "nr_output": r_clean, 
            "nr_raw_output": r_raw
        })

    # ==========================================
    
    nr_df_output = pd.DataFrame(nr_results) 
    nr_df_output.to_csv(NR_OUTPUT_PATH, index=False) 
    print(f"NR Baseline Done! Saved results to {NR_OUTPUT_PATH}")

except FileNotFoundError:
    print(f"Error: NR CSV not found at {NR_CSV_PATH}. Please check the path.")



