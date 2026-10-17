import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition"

CSV_PATH = os.path.join(BASE_DIR, "data", "redefinitionsCSV.csv")


MAIN_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "results_qwen72b_competitive_3agent_constants_zs.csv")

NR_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "nr_baseline") 
os.makedirs(NR_OUTPUT_DIR, exist_ok=True) 
NR_OUTPUT_PATH = os.path.join(NR_OUTPUT_DIR, "nr_results_qwen72b_competitive_3agent_constants_zs.csv") 


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

class CompetitiveDebateMAS:
    def __init__(self, name):
        self.name = name
        
        self.agent_A_sys = "For this task, prioritize strict reading comprehension and logical consistency with the text rules provided by the user. Follow the user's instructions."
        self.agent_B_sys = "For this task, prioritize established knowledge, scientific accuracy and standard mathematical principles."
        self.judge_sys = "You are an impartial judge evaluator. Review the user's original problem, followed by the arguments from Agent A and Agent B. Determine which agent correctly answered based on the user's original prompt."

        # --- NO-REDEFINITION (NR) PROMPTS ---
        self.nr_agent_A_sys = "For this task, carefully read the problem and provide the correct answer."
        
        self.nr_agent_B_sys = "For this task, your goal is to deliberately introduce a subtle but plausible error in your reasoning. Confidently arrive at an incorrect final answer and defend it."
        
        self.nr_judge_sys = "You are an impartial judge evaluator. Review the user's original problem, followed by the arguments from Agent A and Agent B. Determine which agent correctly answered based on the user's original prompt."

    def extract_answer(self, text):
        if "The final answer is:" in text:
            return text.split("The final answer is:")[-1].strip()
        elif "Answer:" in text:
            return text.split("Answer:")[-1].strip()
        return text.strip()

    def run_collaboration(self, constant, redefinition, question):
      
      
        task_prompt = (
            f"Redefine {constant} as {redefinition}.\n"
            f"Answer the following question: {question}\n"
            f"End the response with the phrase "
            f"'The final answer is: ' followed only by the correct result, with no additional text or commentary."
        )

        print("\n --- STAGE 1: OPENING ARGUMENTS ---")
        agent_A_initial = query_local_model(self.agent_A_sys, task_prompt, do_sample=False)
        agent_B_initial = query_local_model(self.agent_B_sys, task_prompt, do_sample=False)
        
        print(" --- STAGE 2: REBUTTALS ---")
        rebuttal_template = (
            f"The original prompt was:\n{task_prompt}\n\n"
            "An opposing agent provided this solution:\n{opposing_solution}\n\n"
            "Critique this solution. Point out any logical flaws based on your priorities. "
            "Do not solve the problem again, just provide a brief critique of the other agent's reasoning."
        )

        agent_A_rebuttal = query_local_model(self.agent_A_sys, rebuttal_template.format(opposing_solution=agent_B_initial), do_sample=False)
        agent_B_rebuttal = query_local_model(self.agent_B_sys, rebuttal_template.format(opposing_solution=agent_A_initial), do_sample=False)
        
        print(" --- STAGE 3: THE VERDICT (JUDGE) ---")
        judge_template = (
            f"ORIGINAL PROMPT:\n{task_prompt}\n\n"
            f"--- AGENT A'S CASE ---\nSolution:\n{agent_A_initial}\nCritique of B:\n{agent_A_rebuttal}\n\n"
            f"--- AGENT B'S CASE ---\nSolution:\n{agent_B_initial}\nCritique of A:\n{agent_B_rebuttal}\n\n"
            f"Carefully read the original prompt and both agents' solutions and critiques.\n"
            f"Then declare the winner's answer. End your response with the phrase "
            f"'The final answer is: ' followed only by the winning correct result."
        )

        judge_final = query_local_model(self.judge_sys, judge_template, do_sample=False)
        
        return {
            "agent_A_initial_raw": agent_A_initial,
            "agent_A_initial_clean": self.extract_answer(agent_A_initial),
            "agent_B_initial_raw": agent_B_initial,
            "agent_B_initial_clean": self.extract_answer(agent_B_initial),
            "agent_A_rebuttal": agent_A_rebuttal,
            "agent_B_rebuttal": agent_B_rebuttal,
            "judge_final_raw": judge_final,
            "judge_final_clean": self.extract_answer(judge_final)
        }

    def run_nr_collaboration(self, question):
       
        task_prompt = (
            f"Answer the following question: {question}\n"
            f"End the response with the phrase "
            f"'The final answer is: ' followed only by the correct result, with no additional text or commentary."
        )

        print("\n --- NR STAGE 1: OPENING ARGUMENTS ---")
       
        agent_A_initial = query_local_model(self.nr_agent_A_sys, task_prompt, do_sample=False)
        agent_B_initial = query_local_model(self.nr_agent_B_sys, task_prompt, do_sample=False)
        
        print(" --- NR STAGE 2: REBUTTALS ---")
        rebuttal_template = (
            f"The original prompt was:\n{task_prompt}\n\n"
            "An opposing agent provided this solution:\n{opposing_solution}\n\n"
            "Critique this solution. Point out any logical flaws based on your priorities. "
            "Do not solve the problem again, just provide a brief critique of the other agent's reasoning."
        )

        
        agent_A_rebuttal = query_local_model(self.nr_agent_A_sys, rebuttal_template.format(opposing_solution=agent_B_initial), do_sample=False)
        agent_B_rebuttal = query_local_model(self.nr_agent_B_sys, rebuttal_template.format(opposing_solution=agent_A_initial), do_sample=False)
        
        print(" --- NR STAGE 3: THE VERDICT (JUDGE) ---")
        judge_template = (
            f"ORIGINAL PROMPT:\n{task_prompt}\n\n"
            f"--- AGENT A'S CASE ---\nSolution:\n{agent_A_initial}\nCritique of B:\n{agent_A_rebuttal}\n\n"
            f"--- AGENT B'S CASE ---\nSolution:\n{agent_B_initial}\nCritique of A:\n{agent_B_rebuttal}\n\n"
            f"Carefully read the original prompt and both agents' solutions and critiques.\n"
            f"Then declare the winner's answer. End your response with the phrase "
            f"'The final answer is: ' followed only by the winning correct result."
        )

      
        judge_final = query_local_model(self.nr_judge_sys, judge_template, do_sample=False)
        
        return {
            "nr_agent_A_raw": agent_A_initial,
            "nr_agent_A_clean": self.extract_answer(agent_A_initial),
            "nr_agent_B_raw": agent_B_initial,
            "nr_agent_B_clean": self.extract_answer(agent_B_initial),
            "nr_agent_A_rebuttal": agent_A_rebuttal,
            "nr_agent_B_rebuttal": agent_B_rebuttal,
            "nr_judge_final_raw": judge_final,
            "nr_judge_final_clean": self.extract_answer(judge_final)
        }

mas_system = CompetitiveDebateMAS("Mistral-Competitive-3Agent")

try:
    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df)} questions from dataset.")
    results = []

    for index, row in df.iterrows():
        print(f"\n=== Question {index+1}: {row['constant']} -> {row['redefinition']} ===")
        
        stage_results = mas_system.run_collaboration(
            row['constant'], row['redefinition'], row['question_text']
        )
        
        record = {
            "constant": row['constant'],
            "redefinition": row['redefinition'],
            "question": row['question_text'],
            "difficulty_level": row.get('difficulty_level', ''),
            "question_difficulty": row.get('question_difficulty', ''),
             
            
           
            "agent_A_initial_raw": stage_results["agent_A_initial_raw"],
            "agent_A_initial_clean": stage_results["agent_A_initial_clean"],
            "agent_B_initial_raw": stage_results["agent_B_initial_raw"],
            "agent_B_initial_clean": stage_results["agent_B_initial_clean"],
            
            "agent_A_rebuttal": stage_results["agent_A_rebuttal"],
            "agent_B_rebuttal": stage_results["agent_B_rebuttal"],
            
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
    unique_questions_df = df[['question_text', 'anchored_answer']].drop_duplicates()
    print(f"\nUnique questions to run for NR: {len(unique_questions_df)}")
    nr_results = []
    
    print("\nRunning No-Redefinition (NR) MAS Baseline on unique questions...")
    for index, row in unique_questions_df.iterrows():
        print(f"\nEvaluating NR Question: {row['question_text'][:50]}...")
        
        stage_results = mas_system.run_nr_collaboration(row['question_text'])
        
        nr_record = {
                "question": row['question_text'],
                "expected_answer": row.get('anchored_answer', ''),
                
                
                "nr_agent_A_initial_raw": stage_results["nr_agent_A_raw"],
                "nr_agent_A_initial_clean": stage_results["nr_agent_A_clean"],
                "nr_agent_B_initial_raw": stage_results["nr_agent_B_raw"],
                "nr_agent_B_initial_clean": stage_results["nr_agent_B_clean"],
                
                "nr_agent_A_rebuttal": stage_results["nr_agent_A_rebuttal"],
                "nr_agent_B_rebuttal": stage_results["nr_agent_B_rebuttal"],
                
                "nr_judge_final_raw": stage_results["nr_judge_final_raw"],
                "nr_judge_final_clean": stage_results["nr_judge_final_clean"]
            }
        nr_results.append(nr_record)

    nr_df_output = pd.DataFrame(nr_results) 
    nr_df_output.to_csv(NR_OUTPUT_PATH, index=False) 
    print(f"NR Baseline Done! Saved results to {NR_OUTPUT_PATH}")

except NameError:
    print("Skipping NR baseline because the main dataframe 'df' was not loaded.")

