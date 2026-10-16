import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# ==========================================
BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition"


CSV_PATH = os.path.join(BASE_DIR, "data", "redefinitionsCSV.csv")

# Outputs
MAIN_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "results_qwen72b_3shot_constants_fs.csv")

NR_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "nr_baseline")
os.makedirs(NR_OUTPUT_DIR, exist_ok=True)
NR_OUTPUT_PATH = os.path.join(NR_OUTPUT_DIR, "nr_results_qwen72b_3shot_constants_fs.csv")


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
def get_few_shot_examples(df, current_constant, diff_level, q_diff, k=3, is_nr=False):
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

class SingleAgent:
    def __init__(self, name):
        self.name = name
        self.system_prompt = ""

    def respond(self, constant, redefinition, question, fs_examples=""):
        redef_str = f"Redefine {constant} as {redefinition}.\n"
        question_str = f"Answer the following question:\n{question}\n"

        examples_str = ""
        if fs_examples:
            examples_str = f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n"

        constraint_str = "End the response with the phrase 'The final answer is: ' followed only by the correct result, with no additional text or commentary."

        full_prompt = redef_str + question_str + examples_str + constraint_str
        raw_response = query_local_model(self.system_prompt, full_prompt)

        clean_response = raw_response
        if "The final answer is:" in raw_response:
            clean_response = raw_response.split("The final answer is:")[-1].strip()

        return raw_response, clean_response

baseline_model = SingleAgent("Baseline_Qwen")

def get_nr_response(question, fs_examples=""):
    question_str = f"Answer the following question:\n{question}\n"

    examples_str = ""
    if fs_examples:
        examples_str = f"Here are some examples of similar questions with their correct answers:\n{fs_examples}\n"

    constraint_str = "End the response with the phrase 'The final answer is: ' followed only by the correct result, with no additional text or commentary."

    prompt = question_str + examples_str + constraint_str
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(messages)
    return outputs[0]["generated_text"][-1]["content"]


# ==========================================
print(f"Loading dataset from {CSV_PATH}...")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} questions for Redefinition Task.")
    results = []

    for index, row in df.iterrows():
        print(f"\n=== Redefinition Q{index+1}: {row['constant']} -> {row['redefinition']} ===")

        fs_examples = get_few_shot_examples(
            df,
            current_constant=row['constant'],
            diff_level=row['difficulty_level'],
            q_diff=row['question_difficulty'],
            k=3,
            is_nr=False
        )

        raw_ans, clean_ans = baseline_model.respond(
            row['constant'],
            row['redefinition'],
            row['question_text'],
            fs_examples=fs_examples
        )

        print(f"Model Answer: {clean_ans}")

        results.append({
            "constant": row['constant'],
            "redefinition": row['redefinition'],
            "difficulty_level": row['difficulty_level'],
            "question_difficulty": row['question_difficulty'],
            "question": row['question_text'],
            "raw_model_output": raw_ans,
            "extracted_answer": clean_ans
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(MAIN_OUTPUT_PATH, index=False)
    print(f"\nRedefinition Experiment finished. Results saved to '{MAIN_OUTPUT_PATH}'.")

except FileNotFoundError:
    print(f"Error: {CSV_PATH} not found. Please check your data folder.")
    exit()

# ==========================================
unique_questions_df = df.drop_duplicates(subset=['question_text']).copy()
print(f"\nRunning NR Baseline on {len(unique_questions_df)} unique questions...")

nr_results = []

for index, row in unique_questions_df.iterrows():
    print(f"\n=== NR Baseline Q{index+1} ===")
    fs_examples = get_few_shot_examples(
        df,
        current_constant=row['constant'],
        diff_level=row['difficulty_level'],
        q_diff=row['question_difficulty'],
        k=3,
        is_nr=True
    )

    raw_ans = get_nr_response(row['question_text'], fs_examples=fs_examples)

    clean_ans = raw_ans
    if "The final answer is:" in raw_ans:
        clean_ans = raw_ans.split("The final answer is:")[-1].strip()

    nr_results.append({
        "question": row['question_text'],
        "nr_output": raw_ans,
        "extracted_nr_output": clean_ans,
        "expected_nr_answer": row['anchored_answer']
    })

nr_df = pd.DataFrame(nr_results)
nr_df.to_csv(NR_OUTPUT_PATH, index=False)
print(f"NR Baseline Done! Saved unique results to '{NR_OUTPUT_PATH}'")
     
