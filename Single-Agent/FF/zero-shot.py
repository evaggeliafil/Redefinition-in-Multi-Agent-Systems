#SINGLE-AGENT constants

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig


BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition"
CSV_PATH = os.path.join(BASE_DIR, "data", "redefinitionsCSV.csv")

# Outputs
MAIN_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "results_qwen72b_constants_zs.csv")

NR_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "nr_baseline")
os.makedirs(NR_OUTPUT_DIR, exist_ok=True)
NR_OUTPUT_PATH = os.path.join(NR_OUTPUT_DIR, "nr_results_qwen72b_constants_zs.csv")

MODEL_ID = os.path.join(BASE_DIR, "models", "Qwen2.5-72B-Instruct")

print(f"Loading Qwen 72B model from local directory: {MODEL_ID}...")

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

def query_local_model(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    outputs = pipe(messages)
    return outputs[0]["generated_text"][-1]["content"]

# ==========================================

class SingleAgent:
    def __init__(self, name):
        self.name = name
        self.system_prompt = ""

    def respond(self, constant, redefinition, question):
        full_prompt = (
            f"Redefine {constant} as {redefinition}.\n"
            f"Answer the following question: {question}\n"
            f"End the response with the phrase 'The final answer is: ' followed only by the correct result, with no additional text or commentary."
        )
        raw_response = query_local_model(self.system_prompt, full_prompt)

        clean_response = raw_response
        if "The final answer is:" in raw_response:
            clean_response = raw_response.split("The final answer is:")[-1].strip()

        return raw_response, clean_response

baseline_model = SingleAgent("Qwen")

def get_nr_response(question):
    prompt = (
        f"Answer the following question: {question}\n"
        f"End the response with the phrase 'The final answer is: ' followed only by the correct result, with no additional text or commentary."
    )
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(messages)
    return outputs[0]["generated_text"][-1]["content"]

# ==========================================
print(f"Loading dataset from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} questions.")

results = []

print("\n--- Starting Redefined Constants Experiment ---")
for index, row in df.iterrows():
    print(f"\n=== Question {index+1}: {row['constant']} -> {row['redefinition']} ===")
    print(f"Q: {row['question_text']}")


    raw_ans, clean_ans = baseline_model.respond(row['constant'], row['redefinition'], row['question_text'])
    print(f"Model Answer: {clean_ans}")

    #redefined results
    results.append({
        "constant": row['constant'],
        "redefinition": row['redefinition'],
        "question": row['question_text'],
        "difficulty_level": row['difficulty_level'],
        "question_difficulty": row['question_difficulty'],
        "raw_model_output": raw_ans,
        "extracted_answer": clean_ans
    })


results_df = pd.DataFrame(results)
results_df.to_csv(MAIN_OUTPUT_PATH, index=False)
print(f"\nExperiment finished. Redefined results saved to {MAIN_OUTPUT_PATH}")

# ==========================================
print("\n--- Starting No-Redefinition (NR) Baseline ---")
unique_questions_df = df[['question_text', 'anchored_answer']].drop_duplicates()
print(f"Unique questions to run for baseline: {len(unique_questions_df)}")

nr_results = []
for index, row in unique_questions_df.iterrows():
    q_text = row['question_text']
    ans = get_nr_response(q_text)
    #nr results
    nr_results.append({
        "question": q_text,
        "nr_output": ans,
        "expected_nr_answer": row['anchored_answer']
    })


nr_df = pd.DataFrame(nr_results)
nr_df.to_csv(NR_OUTPUT_PATH, index=False)
print(f"NR Baseline Done! Saved unique results to {NR_OUTPUT_PATH}")
                                                                                                                                                                
