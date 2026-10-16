# constant mc zs 
import os 
import torch 
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig 

# ========================================== 
BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition"


CSV_PATH = os.path.join(BASE_DIR, "data", "constantsrRED_MC.csv") 

MAIN_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "results_qwen72b_constants_mc_cot.csv") 

NR_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "nr_baseline") 
os.makedirs(NR_OUTPUT_DIR, exist_ok=True) 
NR_OUTPUT_PATH = os.path.join(NR_OUTPUT_DIR, "nr_results_qwen72b_constants_mc_cot.csv") 
NR_CSV_PATH = os.path.join(BASE_DIR, "data", "NR_constantRED_MC.csv")

MODEL_ID = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition/models/Qwen2.5-72B-Instruct" 

print(f"Loading Qwen model from local directory: {MODEL_ID}...") 


# ========================================== 
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

# ========================================== 
class SingleAgent: 
    def __init__(self, name): 
        self.name = name 
        self.system_prompt = "" 


    def respond(self, constant, redefinition, question, opt_a, opt_b, opt_c, opt_d): 
        full_prompt = ( 
            f"Redefine {constant} as {redefinition}.\n" 
            f"Choose A, B, C or D to answer the question:\n"
            f"Question: {question}\n"
            f"A: {opt_a}\n"
            f"B: {opt_b}\n"
            f"C: {opt_c}\n"
            f"D: {opt_d}\n"
            f"Let's think step by step.\n"
            f"Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
            f"End the response with the phrase 'The final answer is: ' followed by the correct letter, with no additional text or commentary." 
        ) 
        raw_response = query_local_model(self.system_prompt, full_prompt) 

        clean_response = raw_response 
        if "The final answer is:" in raw_response: 
          
            clean_response = raw_response.split("The final answer is:")[-1].strip().rstrip('.') 

        return raw_response, clean_response 

baseline_model = SingleAgent("QMath") 


def get_nr_response(question, opt_a, opt_b, opt_c, opt_d): 
    prompt = ( 
        f"Choose A, B, C or D to answer the question:\n"
        f"Question: {question}\n"
        f"A: {opt_a}\n"
        f"B: {opt_b}\n"
        f"C: {opt_c}\n"
        f"D: {opt_d}\n"
        f"Let's think step by step.\n"
        f"Provide only the letter corresponding to the correct answer: \"A\", \"B\", \"C\", or \"D\".\n"
        f"End the response with the phrase 'The final answer is: ' followed by the correct letter, with no additional text or commentary." 
    ) 
    messages = [{"role": "user", "content": prompt}] 
    outputs = pipe(messages) 
    

    generated_text = outputs[0]["generated_text"]
    if isinstance(generated_text, list):
        raw_response = generated_text[-1]["content"]
    else:
        raw_response = generated_text

        
    clean_response = raw_response 
    if "The final answer is:" in raw_response: 
        clean_response = raw_response.split("The final answer is:")[-1].strip().rstrip('.') 
        
    return clean_response

# ========================================== 
print(f"Loading dataset from {CSV_PATH}...") 
df = pd.read_csv(CSV_PATH) 
print(f"Loaded {len(df)} questions.") 

results = [] 

print("\n--- Starting Redefined Constants MC Experiment ---") 
for index, row in df.iterrows(): 
    print(f"\n=== Question {index+1}: {row['constant']} -> {row['redefinition']} ===") 
    print(f"Q: {row['question_text']}") 

   
    raw_ans, clean_ans = baseline_model.respond(
        row['constant'], row['redefinition'], row['question_text'],
        row['option_A'], row['option_B'], row['option_C'], row['option_D']
    ) 
    print(f"Model Answer: {clean_ans} (Expected: {row['expected_mc_answer']}, Anchored: {row['anchored_mc_answer']})") 

  
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
        "raw_model_output": raw_ans, 
        "extracted_answer": clean_ans 
    }) 


results_df = pd.DataFrame(results) 
results_df.to_csv(MAIN_OUTPUT_PATH, index=False) 
print(f"\nExperiment finished. Redefined results saved to {MAIN_OUTPUT_PATH}") 

# ========================================== 

print(f"\n--- Starting No-Redefinition (NR) MC Baseline ---") 
print(f"Loading NR dataset from {NR_CSV_PATH}...") 

nr_df_input = pd.read_csv(NR_CSV_PATH)
print(f"Questions to run for baseline: {len(nr_df_input)}") 

nr_results = [] 
for index, row in nr_df_input.iterrows(): 
    q_text = row['question_text'] 
    
    clean_ans = get_nr_response(
        q_text, 
        row['option_A'], row['option_B'], row['option_C'], row['option_D']
    ) 

    nr_results.append({ 
        "constant": row['constant'],
        "question": q_text, 
        "option_A": row['option_A'],
        "option_B": row['option_B'],
        "option_C": row['option_C'],
        "option_D": row['option_D'],
        "nr_extracted_answer": clean_ans, 
        "expected_nr_answer": row['expected_nr_answer'] 
    }) 


nr_df_output = pd.DataFrame(nr_results) 
nr_df_output.to_csv(NR_OUTPUT_PATH, index=False) 
print(f"NR Baseline Done! Saved results to {NR_OUTPUT_PATH}")
