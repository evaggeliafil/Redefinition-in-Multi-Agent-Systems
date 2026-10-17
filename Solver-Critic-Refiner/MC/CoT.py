import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig


BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition"


CSV_PATH = os.path.join(BASE_DIR, "data", "constantsrRED_MC.csv") 

MAIN_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "results_qwen72b_3scr_constants_mc_cot.csv") 

NR_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "nr_baseline") 
os.makedirs(NR_OUTPUT_DIR, exist_ok=True) 
NR_OUTPUT_PATH = os.path.join(NR_OUTPUT_DIR, "nr_results_qwen72b_3scr_constants_mc_cot.csv") 
NR_CSV_PATH = os.path.join(BASE_DIR, "data", "NR_constantRED_MC.csv")

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
        outputs = pipe(messages, do_sample=False, temperature=0.0)
        
    generated = outputs[0]["generated_text"]
    
    
    if isinstance(generated, list):
      
        return generated[-1]["content"].strip()
    else:
   
        return str(generated).strip()


class ThreeAgentChainMAS_MC:
    def __init__(self, name):
        self.name = name
        self.solver_system_prompt = ""
        self.critic_system_prompt = ""
        self.refiner_system_prompt = ""

    def extract_answer(self, text):
        if "The final answer is:" in text:
            return text.split("The final answer is:")[-1].strip().rstrip('.')
        elif "Answer:" in text:
            return text.split("Answer:")[-1].strip().rstrip('.')
        return text.strip().rstrip('.')

    def run_collaboration(self, constant, redefinition, question,
                          opt_a, opt_b, opt_c, opt_d):

        # --- AGENT 1: SOLVER ---
        solver_prompt = (
            f"Redefine {constant} as {redefinition}.\n"
            f"Choose A, B, C or D to answer the question:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n"
            f"Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            f"Let's think step by step. "
            f"End the response with the phrase 'The final answer is: ' "
            f"followed by the correct letter, with no additional text or commentary."
        )
        agent1_raw = query_local_model(self.solver_system_prompt, solver_prompt, do_sample=False)
        agent1_clean = self.extract_answer(agent1_raw)
        print(f"\n [Agent 1 - Solver]: {agent1_clean}")

        # --- AGENT 2: CRITIC ---
        critic_prompt = (
            f"Redefine {constant} as {redefinition}.\n"
            f"The following question was asked:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n\n"
            f"Another agent chose this answer: {agent1_raw}\n\n"
            f"Your task: carefully check whether the agent used the redefined value "
            f"of {constant} (which is {redefinition}) when selecting its answer, "
            f"or whether it incorrectly relied on the real-world value of {constant}. "
            f"Do not solve the problem yourself. "
            f"End the response with the phrase 'The final answer is: ' "
            f"followed only by one of these three words: "
            f"'correct' if the redefined value was used and the letter choice is right, "
            f"'anchored' if the agent relied on the real-world value instead, "
            f"'wrong' if the redefined value was used but the letter choice is still incorrect."
        )
        agent2_raw = query_local_model(self.critic_system_prompt, critic_prompt, do_sample=False)
        agent2_clean = self.extract_answer(agent2_raw)
        print(f" [Agent 2 - Critic]: {agent2_clean}")

        # --- AGENT 3: REFINER ---
        refiner_prompt = (
            f"Redefine {constant} as {redefinition}.\n"
            f"Choose A, B, C or D to answer the question:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n\n"
            f"A previous agent chose this answer: {agent1_raw}\n\n"
            f"A critic reviewed it and said: {agent2_raw}\n\n"
            f"Taking the critique into account, provide the final correct answer. "
            f"Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            f"Let's think step by step. "
            f"End the response with the phrase 'The final answer is: ' "
            f"followed by the correct letter, with no additional text or commentary."
        )
        agent3_raw = query_local_model(self.refiner_system_prompt, refiner_prompt, do_sample=False)
        agent3_clean = self.extract_answer(agent3_raw)
        print(f" [Agent 3 - Refiner]: {agent3_clean}")

        return {
            "agent1_raw": agent1_raw,
            "agent1_clean": agent1_clean,
            "agent2_raw": agent2_raw,
            "agent2_clean": agent2_clean,
            "agent3_raw": agent3_raw,
            "agent3_clean": agent3_clean,
        }

    def run_nr_collaboration(self, question, opt_a, opt_b, opt_c, opt_d):

        # --- NR AGENT 1: SOLVER ---
        solver_prompt = (
            f"Choose A, B, C or D to answer the question:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n"
            f"Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            f"Let's think step by step. "
            f"End the response with the phrase 'The final answer is: ' "
            f"followed by the correct letter, with no additional text or commentary."
        )
        agent1_raw = query_local_model(self.solver_system_prompt, solver_prompt, do_sample=False)
        agent1_clean = self.extract_answer(agent1_raw)
        print(f"\n [NR Agent 1 - Solver]: {agent1_clean}")

        # --- NR AGENT 2: CRITIC ---
        critic_prompt = (
            f"The following question was asked:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n\n"
            f"Another agent chose this answer: {agent1_raw}\n\n"
            f"Your task: check whether the agent's reasoning and letter choice are correct. "
            f"Point out any factual or logical errors if present. "
            f"Do not solve the problem yourself. "
            f"End the response with the phrase 'The final answer is: ' "
            f"followed only by one of these three words: "
            f"'correct' if the answer is right, "
            f"'incorrect' if the answer is wrong. "
           
        )
        agent2_raw = query_local_model(self.critic_system_prompt, critic_prompt, do_sample=False)
        agent2_clean = self.extract_answer(agent2_raw)
        print(f" [NR Agent 2 - Critic]: {agent2_clean}")

        # --- NR AGENT 3: REFINER ---
        refiner_prompt = (
            f"Choose A, B, C or D to answer the question:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n\n"
            f"A previous agent chose this answer: {agent1_raw}\n\n"
            f"A critic reviewed it and said: {agent2_raw}\n\n"
            f"Taking the critique into account, provide the final correct answer. "
            f"Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            f"Let's think step by step. "
            f"End the response with the phrase 'The final answer is: ' "
            f"followed by the correct letter, with no additional text or commentary."
        )
        agent3_raw = query_local_model(self.refiner_system_prompt, refiner_prompt, do_sample=False)
        agent3_clean = self.extract_answer(agent3_raw)
        print(f" [NR Agent 3 - Refiner]: {agent3_clean}")

        return {
            "agent1_raw": agent1_raw,
            "agent1_clean": agent1_clean,
            "agent2_raw": agent2_raw,
            "agent2_clean": agent2_clean,
            "agent3_raw": agent3_raw,
            "agent3_clean": agent3_clean,
        }


mas_system = ThreeAgentChainMAS_MC("Llama-3AgentChain-MC")

try:
    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df)} questions from dataset.")
    results = []

    for index, row in df.iterrows():
        print(f"\n=== Question {index+1}: {row['constant']} -> {row['redefinition']} ===")

        stage_results = mas_system.run_collaboration(
            row['constant'], row['redefinition'], row['question_text'],
            row['option_A'], row['option_B'], row['option_C'], row['option_D']
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
            "agent1_solver_raw": stage_results["agent1_raw"],
            "agent1_solver_clean": stage_results["agent1_clean"],
            "agent2_critic_raw": stage_results["agent2_raw"],
            "agent2_critic_clean": stage_results["agent2_clean"],
            "agent3_refiner_raw": stage_results["agent3_raw"],
            "agent3_refiner_clean": stage_results["agent3_clean"],
        })

    results_df = pd.DataFrame(results) 
    results_df.to_csv(MAIN_OUTPUT_PATH, index=False) 
    print(f"\nExperiment finished. Redefined results saved to {MAIN_OUTPUT_PATH}")

except FileNotFoundError:
    print(f"Error: CSV not found at {CSV_PATH}. Please check the path.")



try:
    nr_df_input = pd.read_csv(NR_CSV_PATH)
    print(f"\nNR dataset size: {len(nr_df_input)}")
    nr_results = []

    for index, row in nr_df_input.iterrows():
        print(f"\nNR Question: {row['question_text'][:50]}...")

        stage_results = mas_system.run_nr_collaboration(
            row['question_text'],
            row['option_A'], row['option_B'], row['option_C'], row['option_D']
        )

        nr_results.append({
            "constant": row['constant'],
            "question": row['question_text'],
            "question_difficulty": row['question_difficulty'],
            "option_A": row['option_A'],
            "option_B": row['option_B'],
            "option_C": row['option_C'],
            "option_D": row['option_D'],
            "expected_nr_answer": row['expected_nr_answer'],
            "agent1_solver_raw": stage_results["agent1_raw"],
            "agent1_solver_clean": stage_results["agent1_clean"],
            "agent2_critic_raw": stage_results["agent2_raw"],
            "agent2_critic_clean": stage_results["agent2_clean"],
            "agent3_refiner_raw": stage_results["agent3_raw"],
            "agent3_critic_clean": stage_results["agent3_clean"],
        })

    nr_df_output = pd.DataFrame(nr_results) 
    nr_df_output.to_csv(NR_OUTPUT_PATH, index=False) 
    print(f"NR Baseline Done! Saved results to {NR_OUTPUT_PATH}")

except FileNotFoundError:
    print(f"Error: CSV not found at {NR_CSV_PATH}. Please check the path.")
