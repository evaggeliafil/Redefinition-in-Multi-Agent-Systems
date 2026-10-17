import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition"

CSV_PATH = os.path.join(BASE_DIR, "data", "redefinitionsCSV.csv")


MAIN_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "results_qwen72b_council_4agent_constants_2s.csv")

NR_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "nr_baseline") 
os.makedirs(NR_OUTPUT_DIR, exist_ok=True) 
NR_OUTPUT_PATH = os.path.join(NR_OUTPUT_DIR, "nr_results_qwen72b_council_4agent_constants_2s.csv") 


MODEL_ID = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition/models/Qwen2.5-72B-Instruct" 

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
        outputs = pipe(messages, do_sample=False)
        
    generated = outputs[0]["generated_text"]
    
    if isinstance(generated, list):
        return generated[-1]["content"].strip()
    else:
        return str(generated).strip()

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

class EnsembleCouncilMAS:
    def __init__(self, name):
        self.name = name
        

        self.sys_prompt_faithful = "Ignore your prior knowledge. You must strictly answer based on the user's prompt and its rules."
        self.sys_memory_faithful = "Ignore the user's redefinition rules. Answer according to standard, conventional knowledge, scientific accuracy, and real-world facts."
        self.sys_neutral = "You are a helpful, independent assistant solving a problem."
        self.sys_arbitrator = (
            "You are an Arbitrator evaluating a conflict between user-provided rules and standard knowledge. "
            "Review the original prompt, and the solutions from three independent solvers. "
            "Decide which source of knowledge should prevail based on the hierarchy of instructions "
            "and output the final, correct answer."
        )

      
        self.nr_sys_A = "For this task, carefully read the problem and provide the correct answer."
        self.nr_sys_B = "For this task, carefully read the problem and provide the correct answer." 
        self.nr_sys_C = "You are a helpful, independent assistant solving a problem."
        self.nr_sys_arbitrator = (
            "You are an impartial judge. Review the original prompt and the solutions from three independent solvers. "
            "Determine the correct answer and output the final result."
        )

    def extract_answer(self, text):
        if "The final answer is:" in text:
            return text.split("The final answer is:")[-1].strip()
        elif "Answer:" in text:
            return text.split("Answer:")[-1].strip()
        return text.strip()

    def run_collaboration(self, constant, redefinition, question, fs_examples=""):
        task_prompt = (
            f"Redefine {constant} as {redefinition}.\n"
            f"Answer the following question: {question}\n"
            f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n"
            f"End the response with the phrase "
            f"'The final answer is: ' followed only by the correct result, with no additional text or commentary."
        )

   
        print("\n --- STAGE 1: INDEPENDENT COUNCIL SOLVING ---")
        agent_A_initial = query_local_model(self.sys_prompt_faithful, task_prompt, do_sample=False)
        agent_B_initial = query_local_model(self.sys_memory_faithful, task_prompt, do_sample=False)
        agent_C_initial = query_local_model(self.sys_neutral, task_prompt, do_sample=False)
        

        print(" --- STAGE 2: ARBITRATOR VERDICT ---")
        judge_template = (
            f"ORIGINAL PROMPT:\n{task_prompt}\n\n"
            f"--- PROMPT-FAITHFUL SOLVER (A) ---\n{agent_A_initial}\n\n"
            f"--- MEMORY-FAITHFUL SOLVER (B) ---\n{agent_B_initial}\n\n"
            f"--- INDEPENDENT SOLVER (C) ---\n{agent_C_initial}\n\n"
            f"Carefully evaluate the conflict and the solvers' answers.\n"
            f"End your response with the phrase "
            f"'The final answer is: ' followed only by the winning correct result."
        )

        judge_final = query_local_model(self.sys_arbitrator, judge_template, do_sample=False)
        
        return {
            "agent_A_raw": agent_A_initial,
            "agent_A_clean": self.extract_answer(agent_A_initial),
            "agent_B_raw": agent_B_initial,
            "agent_B_clean": self.extract_answer(agent_B_initial),
            "agent_C_raw": agent_C_initial,
            "agent_C_clean": self.extract_answer(agent_C_initial),
            "judge_final_raw": judge_final,
            "judge_final_clean": self.extract_answer(judge_final)
        }

    def run_nr_collaboration(self, question, fs_examples=""):
        task_prompt = (
            f"Answer the following question: {question}\n"
            f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n"
            f"End the response with the phrase "
            f"'The final answer is: ' followed only by the correct result, with no additional text or commentary."
        )

        print("\n --- NR STAGE 1: INDEPENDENT COUNCIL SOLVING ---")
        agent_A_initial = query_local_model(self.nr_sys_A, task_prompt, do_sample=False)
        agent_B_initial = query_local_model(self.nr_sys_B, task_prompt, do_sample=False)
        agent_C_initial = query_local_model(self.nr_sys_C, task_prompt, do_sample=False)
        
        print(" --- NR STAGE 2: ARBITRATOR VERDICT ---")
        judge_template = (
            f"ORIGINAL PROMPT:\n{task_prompt}\n\n"
            f"--- SOLVER A ---\n{agent_A_initial}\n\n"
            f"--- SOLVER B ---\n{agent_B_initial}\n\n"
            f"--- SOLVER C ---\n{agent_C_initial}\n\n"
            f"Determine the correct answer.\n"
            f"End your response with the phrase "
            f"'The final answer is: ' followed only by the winning correct result."
        )

        judge_final = query_local_model(self.nr_sys_arbitrator, judge_template, do_sample=False)
        
        return {
            "nr_agent_A_raw": agent_A_initial,
            "nr_agent_A_clean": self.extract_answer(agent_A_initial),
            "nr_agent_B_raw": agent_B_initial,
            "nr_agent_B_clean": self.extract_answer(agent_B_initial),
            "nr_agent_C_raw": agent_C_initial,
            "nr_agent_C_clean": self.extract_answer(agent_C_initial),
            "nr_judge_final_raw": judge_final,
            "nr_judge_final_clean": self.extract_answer(judge_final)
        }

mas_system = EnsembleCouncilMAS("Llama-Council-4Agent")

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
        stage_results = mas_system.run_collaboration(
            row['constant'], row['redefinition'], row['question_text'], fs_examples=fs_examples
        )
        
        record = {
            "constant": row['constant'],
            "redefinition": row['redefinition'],
            "question": row['question_text'],
            "difficulty_level": row.get('difficulty_level', ''),
            "question_difficulty": row.get('question_difficulty', ''),
             
     
            "agent_A_raw": stage_results["agent_A_raw"],
            "agent_A_clean": stage_results["agent_A_clean"],
            "agent_B_raw": stage_results["agent_B_raw"],
            "agent_B_clean": stage_results["agent_B_clean"],
            "agent_C_raw": stage_results["agent_C_raw"],
            "agent_C_clean": stage_results["agent_C_clean"],
            
            "judge_final_raw": stage_results["judge_final_raw"],
            "judge_final_clean": stage_results["judge_final_clean"],
            "correct_redefined_answer": row['correct_redefined_answer'], 
            "anchored_answer": row['anchored_answer']
        }
        results.append(record)

    results_df = pd.DataFrame(results)
    results_df.to_csv(MAIN_OUTPUT_PATH, index=False) 
    print(f"\nExperiment finished. Redefined results saved to {MAIN_OUTPUT_PATH}") 

except FileNotFoundError:
    print(f"Error: CSV not found at {CSV_PATH}. Please check the path.")

df = pd.read_csv(CSV_PATH)
        
try:
    if 'df' in locals():
        unique_questions_df = df.drop_duplicates(subset=['question_text']).copy()
        print(f"\nUnique questions to run for NR: {len(unique_questions_df)}")
        nr_results = [] 
        

        print("\nRunning No-Redefinition (NR) Council Baseline on unique questions...")
        for index, row in unique_questions_df.iterrows():
            print(f"\nEvaluating NR Question: {row['question_text'][:50]}...")
            fs_examples = get_few_shot_examples(
                df, 
                current_constant=row['constant'], 
                diff_level=row['difficulty_level'], 
                q_diff=row['question_difficulty'], 
                k=2, 
                is_nr=True
            )
            stage_results = mas_system.run_nr_collaboration(row['question_text'], fs_examples=fs_examples)
            
            nr_record = {
                    "question": row['question_text'],
                    "expected_answer": row.get('anchored_answer', ''),
                    
                    "nr_agent_A_raw": stage_results["nr_agent_A_raw"],
                    "nr_agent_A_clean": stage_results["nr_agent_A_clean"],
                    "nr_agent_B_raw": stage_results["nr_agent_B_raw"],
                    "nr_agent_B_clean": stage_results["nr_agent_B_clean"],
                    "nr_agent_C_raw": stage_results["nr_agent_C_raw"],
                    "nr_agent_C_clean": stage_results["nr_agent_C_clean"],
                    
                    "nr_judge_final_raw": stage_results["nr_judge_final_raw"],
                    "nr_judge_final_clean": stage_results["nr_judge_final_clean"]
                }
            nr_results.append(nr_record)
   
        nr_df = pd.DataFrame(nr_results)
        nr_df.to_csv(NR_OUTPUT_PATH, index=False)
        print(f"NR Baseline Done! Saved results to {NR_OUTPUT_PATH}")

except Exception as e:
   
    print(f"An error occurred during the NR baseline: {e}")
