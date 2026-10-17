import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition"

CSV_PATH = os.path.join(BASE_DIR, "data", "redefinitionsCSV.csv")


MAIN_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "results_qwen72b_3scr_2s.csv")
NR_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "nr_baseline")
os.makedirs(NR_OUTPUT_DIR, exist_ok=True)
NR_OUTPUT_PATH = os.path.join(NR_OUTPUT_DIR, "nr_results_qwen72b_3scr_2s.csv")

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


class ThreeAgentChainMAS:
    def __init__(self, name):
        self.name = name
      
        self.solver_system_prompt = ""
        self.critic_system_prompt = ""
        self.refiner_system_prompt = ""

    def extract_answer(self, text):
        if "The final answer is:" in text:
            return text.split("The final answer is:")[-1].strip()
        elif "Answer:" in text:
            return text.split("Answer:")[-1].strip()
        return text.strip()

    def run_collaboration(self, constant, redefinition, question, fs_examples=""):

        # --- AGENT 1: SOLVER ---
        solver_prompt = (
            f"Redefine {constant} as {redefinition}.\n"
            f"Answer the following question: {question}\n"
            f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n"
            f"End the response with the phrase 'The final answer is: ' "
            f"followed only by the correct result, with no additional text or commentary."
        )
        agent1_raw = query_local_model(self.solver_system_prompt, solver_prompt, do_sample=False)
        agent1_clean = self.extract_answer(agent1_raw)
        print(f"\n [Agent 1 - Solver]: {agent1_clean}")

        # --- AGENT 2: CRITIC ---
        critic_prompt = (
            f"Redefine {constant} as {redefinition}.\n"
            f"The following question was asked: {question}\n"
            f"Another agent proposed this answer: {agent1_raw}\n\n"
            f"Your task: carefully check whether the agent used the redefined value "
            f"of {constant} (which is {redefinition}) in its reasoning "
            f"Also check whether the mathematical reasoning is correct. "
            f"Do not solve the problem yourself. "
            f"End the response with the phrase 'The final answer is: ' "
            f"followed only by one of these three words: "
            f"'correct' if the redefined value was used and the math is right, "
            f"'anchored' if the real-world value was used instead, "
            f"'wrong' if the redefined value was used but the math contains an error."
        )
        agent2_raw = query_local_model(self.critic_system_prompt, critic_prompt, do_sample=False)
        agent2_clean = self.extract_answer(agent2_raw)
        print(f" [Agent 2 - Critic]: {agent2_clean}")

        # --- AGENT 3: REFINER ---
        refiner_prompt = (
            f"Redefine {constant} as {redefinition}.\n"
            f"Answer the following question: {question}\n\n"
            f"A previous agent gave this answer: {agent1_raw}\n\n"
            f"A critic reviewed it and said: {agent2_raw}\n\n"
            f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n"
            f"Taking the critique into account, provide the final correct answer. "
            f"End the response with the phrase 'The final answer is: ' "
            f"followed only by the correct result, with no additional text or commentary."
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

    def run_nr_collaboration(self, question, fs_examples=""):

        # --- NR AGENT 1: SOLVER ---
        solver_prompt = (
            f"Answer the following question: {question}\n"
            f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n"
            f"End the response with the phrase 'The final answer is: ' "
            f"followed only by the correct result, with no additional text or commentary."
        )
        agent1_raw = query_local_model(self.solver_system_prompt, solver_prompt, do_sample=False)
        agent1_clean = self.extract_answer(agent1_raw)
        print(f"\n [NR Agent 1 - Solver]: {agent1_clean}")

        # --- NR AGENT 2: CRITIC ---
        critic_prompt = (
            f"The following question was asked: {question}\n"
            f"Another agent proposed this answer: {agent1_raw}\n\n"
            f"Your task: check whether the agent's reasoning and answer are correct. "
            f"Point out any factual or logical errors if present. "
            f"Do not solve the problem yourself. "
            f"End the response with the phrase 'The final answer is: ' "
            f"followed only by either 'correct' if no errors were found, "
            f"or 'incorrect' if an error was found."
        )
        agent2_raw = query_local_model(self.critic_system_prompt, critic_prompt, do_sample=False)
        agent2_clean = self.extract_answer(agent2_raw)
        print(f" [NR Agent 2 - Critic]: {agent2_clean}")

        # --- NR AGENT 3: REFINER ---
        refiner_prompt = (
            f"Answer the following question: {question}\n\n"
            f"A previous agent gave this answer: {agent1_raw}\n\n"
            f"A critic reviewed it and said: {agent2_raw}\n\n"
            f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n"
            f"Taking the critique into account, provide the final correct answer. "
            f"End the response with the phrase 'The final answer is: ' "
            f"followed only by the correct result, with no additional text or commentary."
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

mas_system = ThreeAgentChainMAS("QwenMath-3Agent-Chain")

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

        stage_results = mas_system.run_collaboration(
            row['constant'], row['redefinition'], row['question_text'], fs_examples=fs_examples
        )

        results.append({
            "constant": row['constant'],
            "redefinition": row['redefinition'],
            "question": row['question_text'],
            "difficulty_level": row.get('difficulty_level', ''),
            "question_difficulty": row.get('question_difficulty', ''),
            "agent1_solver_raw": stage_results["agent1_raw"],
            "agent1_solver_clean": stage_results["agent1_clean"],
            "agent2_critic_raw": stage_results["agent2_raw"],
            "agent2_critic_clean": stage_results["agent2_clean"],
            "agent3_refiner_raw": stage_results["agent3_raw"],
            "agent3_refiner_clean": stage_results["agent3_clean"],
            "correct_redefined_answer": row['correct_redefined_answer'], 
            "anchored_answer": row['anchored_answer'],
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(MAIN_OUTPUT_PATH, index=False)
    print(f"\nExperiment finished. Redefined results saved to {MAIN_OUTPUT_PATH}")

except FileNotFoundError:
    print(f"Error: CSV not found at {CSV_PATH}. Please check the path.")


try:
    df = pd.read_csv(CSV_PATH)
    unique_questions_df = df.drop_duplicates(subset=['question_text']).copy()
    print(f"\nUnique questions for NR baseline: {len(unique_questions_df)}")
    nr_results = []

    for index, row in unique_questions_df.iterrows():
        print(f"\nNR Question: {row['question_text'][:50]}...")

        fs_examples = get_few_shot_examples(
            df, 
            current_constant=row['constant'], 
            diff_level=row['difficulty_level'], 
            q_diff=row['question_difficulty'], 
            k=2, 
            is_nr=True
        )

        stage_results = mas_system.run_nr_collaboration(
            question=row['question_text'],
            fs_examples=fs_examples
        )

        nr_results.append({
            "question": row['question_text'],
            "expected_answer": row.get('anchored_answer', ''),
            "agent1_solver_raw": stage_results["agent1_raw"],
            "agent1_solver_clean": stage_results["agent1_clean"],
            "agent2_critic_raw": stage_results["agent2_raw"],
            "agent2_critic_clean": stage_results["agent2_clean"],
            "agent3_refiner_raw": stage_results["agent3_raw"],
            "agent3_refiner_clean": stage_results["agent3_clean"],
        })

    nr_df = pd.DataFrame(nr_results)
    nr_df.to_csv(NR_OUTPUT_PATH, index=False)
    print(f"NR Baseline Done! Saved unique results to {NR_OUTPUT_PATH}")

except NameError:
    print("Skipping NR baseline because df was not loaded.")
