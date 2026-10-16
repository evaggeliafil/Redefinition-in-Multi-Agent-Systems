import os 
import torch 
import random
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig 



BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition"
CSV_PATH = os.path.join(BASE_DIR, "data", "constantsrRED_MC.csv") 

MAIN_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "results_qwen72b_peerreview_constants_mc_3s.csv") 

NR_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "nr_baseline") 
os.makedirs(NR_OUTPUT_DIR, exist_ok=True) 
NR_OUTPUT_PATH = os.path.join(NR_OUTPUT_DIR, "nr_results_qwen72b_peerreview_constants_mc_3s.csv") 
NR_CSV_PATH = os.path.join(BASE_DIR, "data", "NR_constantRED_MC.csv")

MODEL_ID = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition/models/Qwen72B" 


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
    do_sample=False, 
    temperature=0.0, 
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


def get_few_shot_examples(df, current_constant, diff_level, q_diff, k=3, is_nr=False):
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
        
       
        self.roles = [
            "You are an advanced AI capable of performing complex calculations, logical reasoning, and providing efficient solutions to mathematical problems.",
            "You are an experienced math teacher with a deep understanding of mathematical concepts, methods, and pedagogy. You can offer insights into problem-solving strategies and educational approaches to mathematics.",
            "You are a scientist who specializes in advanced mathematical theories and research. You can contribute deep theoretical knowledge and expertise in cutting-edge mathematical areas, such as topology, abstract algebra, or number theory.",
            "You are a professional engineer with practical experience in applying mathematical concepts to solve real-world problems. You can offer a practical viewpoint on how mathematics is used in designing, analyzing, and improving systems and structures.",
            "You are an expert in computer science, well-versed in algorithms, computational theory, and programming. You can assist in solving problems that involve algorithmic thinking, computational complexity, or numerical methods."
        ]

    def extract_answer(self, text):
        
        if "The final answer is:" in text:
            return text.split("The final answer is:")[-1].strip().rstrip('.')
        elif "Answer:" in text:
            return text.split("Answer:")[-1].strip().rstrip('.')
        return text.strip().rstrip('.')

    def run_collaboration(self, constant, redefinition, question, opt_a, opt_b, opt_c, opt_d, fs_examples=""):
        
        agent_roles = random.sample(self.roles, 2)
        role_1, role_2 = agent_roles[0], agent_roles[1]

        examples_str = ""
        if fs_examples:
            examples_str = f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n\n"

        
       
        task = (
            f"Redefine {constant} as {redefinition}.\n"
            f"Choose A, B, C or D to answer the question:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n"
        )
        constraint = (
            f"Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            f"End the response with the phrase 'The final answer is: ' followed by the correct letter, with no additional text or commentary."
        )
        task_prompt = task + examples_str + constraint
        

        print("\n --- STAGE 1: CREATE ---")
        agent1_initial = query_local_model(role_1, task_prompt, do_sample=False, temperature=0.0)
        agent2_initial = query_local_model(role_2, task_prompt, do_sample=False, temperature=0.0)
        
        agent1_initial_clean = self.extract_answer(agent1_initial)
        agent2_initial_clean = self.extract_answer(agent2_initial)

        print(f" [Agent 1 Initial]: {agent1_initial_clean}")
        print(f" [Agent 2 Initial]: {agent2_initial_clean}")

        print(" --- STAGE 2: REVIEW ---")
        review_template = (
            f"{task_prompt}\n\n"
            "Here is a solution from another agent:\n{peer_solution}\n\n"
            "Please examine this agent's reasoning process step by step and offer feedback on its reasoning. "
            "You can rate your confidence in your feedback on a scale from 1-10, where 10 indicates the highest level of confidence."
        )
        

        review_1_to_2 = query_local_model(role_1, review_template.format(peer_solution=agent2_initial), do_sample=False)
        review_2_to_1 = query_local_model(role_2, review_template.format(peer_solution=agent1_initial), do_sample=False)
        
        print(" --- STAGE 3: REVISE ---")
        revise_template = (
            f"{task_prompt}\n\n"
            "Your initial solution was:\n{own_solution}\n\n"
            "Here is the feedback for your solution from another agent:\n{peer_review}\n\n"
            "Using the other agent's feedback as additional information, can you provide your revised answer to the problem? "
            "Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            "End the response with the phrase 'The final answer is: ' followed by the correct letter, with no additional text or commentary."
        )

        agent1_final = query_local_model(role_1, revise_template.format(own_solution=agent1_initial, peer_review=review_2_to_1), do_sample=False)
        agent2_final = query_local_model(role_2, revise_template.format(own_solution=agent2_initial, peer_review=review_1_to_2), do_sample=False)

        agent1_final_clean = self.extract_answer(agent1_final)
        agent2_final_clean = self.extract_answer(agent2_final)

        print(f" [Agent 1 Final Consensus]: {agent1_final_clean}")
        print(f" [Agent 2 Final Consensus]: {agent2_final_clean}")
        
        is_tie = False
        if agent1_final_clean == agent2_final_clean:
            final_consensus = agent1_final_clean 
        else:
            is_tie = True
            final_consensus = agent1_final_clean 

        return {
            "roles_used": [role_1, role_2],
            "agent1_initial_raw": agent1_initial,
            "agent1_initial_clean": agent1_initial_clean,
            "agent2_initial_raw": agent2_initial,
            "agent2_initial_clean": agent2_initial_clean,
            "review_1_to_2": review_1_to_2,
            "review_2_to_1": review_2_to_1,
            "agent1_final_raw": agent1_final,
            "agent1_final_clean": agent1_final_clean,
            "agent2_final_raw": agent2_final,
            "agent2_final_clean": agent2_final_clean,
            "final_consensus": final_consensus,
            "is_tie": is_tie
        }

    def run_nr_collaboration(self, question, opt_a, opt_b, opt_c, opt_d, fs_examples=""):
        agent_roles = random.sample(self.roles, 2)
        role_1, role_2 = agent_roles[0], agent_roles[1]

        examples_str = ""
        if fs_examples:
            examples_str = f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n\n"


        task = (
            f"Choose A, B, C or D to answer the question:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n"
        )

        constraint = (
            f"Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            f"End the response with the phrase 'The final answer is: ' followed by the correct letter, with no additional text or commentary."
        )

        task_prompt = task + examples_str + constraint

        print("\n --- NR STAGE 1: CREATE ---")
        agent1_initial = query_local_model(role_1, task_prompt, do_sample=False, temperature=0.0)
        agent2_initial = query_local_model(role_2, task_prompt, do_sample=False, temperature=0.0)
        
        agent1_initial_clean = self.extract_answer(agent1_initial)
        agent2_initial_clean = self.extract_answer(agent2_initial)

        print(f" [NR Agent 1 Initial]: {agent1_initial_clean}")
        print(f" [NR Agent 2 Initial]: {agent2_initial_clean}")

        print(" --- NR STAGE 2: REVIEW ---")
        review_template = (
            f"{task_prompt}\n\n"
            "Here is a solution from another agent:\n{peer_solution}\n\n"
            "Please examine this agent's reasoning process step by step and offer feedback on its reasoning. "
            "You can rate your confidence in your feedback on a scale from 1-10, where 10 indicates the highest level of confidence."
        )

        review_1_to_2 = query_local_model(role_1, review_template.format(peer_solution=agent2_initial), do_sample=False)
        review_2_to_1 = query_local_model(role_2, review_template.format(peer_solution=agent1_initial), do_sample=False)
        
        print(" --- NR STAGE 3: REVISE ---")
        revise_template = (
            f"{task_prompt}\n\n"
            "Your initial solution was:\n{own_solution}\n\n"
            "Here is the feedback for your solution from another agent:\n{peer_review}\n\n"
            "Using the other agent's feedback as additional information, can you provide your revised answer to the problem? "
            "Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            "End the response with the phrase 'The final answer is: ' followed by the correct letter, with no additional text or commentary."
        )

        agent1_final = query_local_model(role_1, revise_template.format(own_solution=agent1_initial, peer_review=review_2_to_1), do_sample=False)
        agent2_final = query_local_model(role_2, revise_template.format(own_solution=agent2_initial, peer_review=review_1_to_2), do_sample=False)

        agent1_final_clean = self.extract_answer(agent1_final)
        agent2_final_clean = self.extract_answer(agent2_final)

        print(f" [NR Agent 1 Final Consensus]: {agent1_final_clean}")
        print(f" [NR Agent 2 Final Consensus]: {agent2_final_clean}")
        
        is_tie = False
        if agent1_final_clean == agent2_final_clean:
            final_consensus = agent1_final_clean 
        else:
            is_tie = True
            final_consensus = agent1_final_clean

        return {
            "roles_used": [role_1, role_2],
            "agent1_initial_raw": agent1_initial,
            "agent1_initial_clean": agent1_initial_clean,
            "agent2_initial_raw": agent2_initial,
            "agent2_initial_clean": agent2_initial_clean,
            "review_1_to_2": review_1_to_2,
            "review_2_to_1": review_2_to_1,
            "agent1_final_raw": agent1_final,
            "agent1_final_clean": agent1_final_clean,
            "agent2_final_raw": agent2_final,
            "agent2_final_clean": agent2_final_clean,
            "final_consensus": final_consensus,
            "is_tie": is_tie
        }



mas_system = PeerReviewMAS("Qwen-MAS-ZeroShot")

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
            k=3,
            is_nr=False
        )
        print(fs_examples)
        
      
        stage_results = mas_system.run_collaboration(
            row['constant'], row['redefinition'], row['question_text'],
            row['option_A'], row['option_B'], row['option_C'], row['option_D'], fs_examples=fs_examples
        )
   
        record = {
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
            
            "roles_used": stage_results["roles_used"], 
            
            "agent1_initial_clean": stage_results["agent1_initial_clean"],
            "agent2_initial_clean": stage_results["agent2_initial_clean"],
            "agent1_initial_raw": stage_results["agent1_initial_raw"],
            "agent2_initial_raw": stage_results["agent2_initial_raw"],
            
            "review_1_to_2": stage_results["review_1_to_2"],
            "review_2_to_1": stage_results["review_2_to_1"],
            
            "agent1_final_clean": stage_results["agent1_final_clean"],
            "agent2_final_clean": stage_results["agent2_final_clean"],
            "agent1_final_raw": stage_results["agent1_final_raw"],
            "agent2_final_raw": stage_results["agent2_final_raw"],
            
            "extracted_answer": stage_results["final_consensus"], 
            "is_tie": stage_results["is_tie"]
        }
        
        results.append(record)

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
        print(f"\nEvaluating NR Question: {row['question_text'][:50]}...")
        fs_examples = get_few_shot_examples(
            nr_df_input,
            current_constant=row['constant'],
            diff_level="", 
            q_diff=row['question_difficulty'],
            k=3,
            is_nr=True
        )
        
        stage_results = mas_system.run_nr_collaboration(
            row['question_text'],
            row['option_A'], row['option_B'], row['option_C'], row['option_D'], fs_examples=fs_examples
        )
        
        nr_record = {
            "constant": row['constant'],
            "question": row['question_text'],
            "option_A": row['option_A'],
            "option_B": row['option_B'],
            "option_C": row['option_C'],
            "option_D": row['option_D'],
            "expected_nr_answer": row['expected_nr_answer'],
            
            "roles_used": stage_results["roles_used"], 
            "agent1_initial_clean": stage_results["agent1_initial_clean"],
            "agent2_initial_clean": stage_results["agent2_initial_clean"],
            "agent1_initial_raw": stage_results["agent1_initial_raw"],
            "agent2_initial_raw": stage_results["agent2_initial_raw"],
            
            "review_1_to_2": stage_results["review_1_to_2"],
            "review_2_to_1": stage_results["review_2_to_1"],
            
            "agent1_final_clean": stage_results["agent1_final_clean"],
            "agent2_final_clean": stage_results["agent2_final_clean"],
            "agent1_final_raw": stage_results["agent1_final_raw"],
            "agent2_final_raw": stage_results["agent2_final_raw"],
            
            "nr_output": stage_results["final_consensus"], 
            "is_tie": stage_results["is_tie"]
        }
        nr_results.append(nr_record)

    nr_df_output = pd.DataFrame(nr_results) 
    nr_df_output.to_csv(NR_OUTPUT_PATH, index=False) 
    print(f"NR Baseline Done! Saved results to {NR_OUTPUT_PATH}")
except FileNotFoundError:
    print(f"Error: CSV not found at {NR_CSV_PATH}. Please check the path.")

