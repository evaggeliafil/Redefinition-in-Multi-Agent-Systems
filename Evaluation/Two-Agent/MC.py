import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition"
MODEL_ID = os.path.join(BASE_DIR, "models", "Meta-Llama-3.1-70B-Instruct")

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
    max_new_tokens=50,
    do_sample=False,
    temperature=0.0,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)

def query_judge(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a strict, objective evaluation assistant."},
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(messages)
    return outputs[0]["generated_text"].strip().lower()

# ==========================================

def eval_nr(llm_ans, real_ans):
    prompt = f"""You are tasked with comparing two answers for a Multiple Choice question: one provided by an LLM (the "LLM answer") and the correct answer (the "real answer"). Your job is to determine if the LLM answer matches the real answer. The correct answer will be a single letter (A, B, C, or D).
The comparison should strictly focus on whether the LLM final answer conveys the same meaning or provide the same information as the correct answer. Minor differences in phrasing, wording, or structure are acceptable as long as the core meaning remains identical.

**Instructions:**
1. Compare the LLM answer to the real answer carefully.
2. If the LLM answer matches the real answer, output: correct
3. If the LLM answer does not match the real answer, output: incorrect

Do not output explanations, justifications, or any additional text only the word "correct" or "incorrect".

**LLM Answer:**
{llm_ans}

**Real Answer:**
{real_ans}

**Output:**"""
    return query_judge(prompt)

def eval_redefinition(llm_ans, ref_1, ref_2):
    prompt = f"""You are tasked with comparing a Multiple Choice answer provided by an LLM (the "LLM answer") to two reference answers: "Reference Answer 1" and "Reference Answer 2". Your job is to determine if the LLM answer matches either of the two reference answers.
The comparison should strictly focus on whether the LLM final answer conveys the same meaning or provides the same information as one of the reference answers. Minor differences in phrasing, wording, or structure are acceptable as long as the core meaning remains identical.

**Instructions:**
1. Compare the LLM answer carefully with "Reference Answer 1" and "Reference Answer 2".
2. If the LLM answer matches "Reference Answer 1", output: first
3. If the LLM answer matches "Reference Answer 2", output: second
4. If the LLM answer matches neither of the two, output: none

Do not output explanations, justifications, or any additional text only the words "first", "second", or "none".

**LLM Answer:**
{llm_ans}

**Reference Answer 1:**
{ref_1}

**Reference Answer 2:**
{ref_2}

**Output:**"""
    return query_judge(prompt)

def eval_refusal(llm_ans):
    prompt = f"""You are tasked with analyzing an LLM answer that does not match either of two reference answers. Your job is to classify the LLM answer into one of the following categories:
1. **Wrong Answer**: The LLM provided an incorrect response to the question, either factually or logically.
2. **Blank Answer**: The LLM provided no substantive response, leaving the answer blank or completely empty.
3. **Refusal to Answer**: The LLM explicitly refused to answer the question, citing reasons such as the question being nonsensical, impossible to answer, or against its guidelines.

**Instructions:**
1. Analyze the LLM answer and determine which of the three categories it belongs to.
2. If the LLM answer is a **Wrong Answer**, output: wrong
3. If the LLM answer is a **Blank Answer**, output: blank
4. If the LLM answer is a **Refusal to Answer**, output: refusal
5. If the classification is unclear, choose the category that best fits the content of the LLM answer.

Do not output explanations, justifications, or any additional text only the words "wrong", "blank", or "refusal".

**LLM Answer:**
{llm_ans}

**Output:**"""
    return query_judge(prompt)

# ==========================================

def grade_experiment(main_csv_path, nr_csv_path, out_csv_path, out_txt_path):
    print(f"\n[{main_csv_path.split('/')[-1]}] -> Starting Evaluation...")

    nr_df = pd.read_csv(os.path.join(BASE_DIR, nr_csv_path))
    exp_df = pd.read_csv(os.path.join(BASE_DIR, main_csv_path))

   
    full_df = pd.merge(exp_df, nr_df, on=["constant", "question"], how="inner")

    paper_categories = []
    wrong_types = []
    
   
    agent1_categories = []
    agent1_wrong_types = []
    
    is_nr_correct_list = []

    for index, row in full_df.iterrows():
       
        model_ans = str(row['extracted_answer'])
        
        
        agent1_ans = str(row.get('agent1_solver_clean', '')) 
        
        nr_ans = str(row['nr_output'])
        ref_redefined = str(row['expected_mc_answer'])
        ref_anchored = str(row['anchored_mc_answer'])
        ref_nr_expected = str(row['expected_nr_answer'])

        
        nr_judge = eval_nr(nr_ans, ref_nr_expected)
        nr_correct = True if "correct" in nr_judge and "incorrect" not in nr_judge else False
        is_nr_correct_list.append(nr_correct)

       
        match_result = eval_redefinition(model_ans, ref_redefined, ref_anchored)
        error_type = "N/A"

        if "first" in match_result: category = "Correct responses"
        elif "second" in match_result: category = "Anchored responses"
        else: category = "Completely wrong responses"

        
        if category == "Completely wrong responses":
            if len(model_ans.strip()) < 1 or model_ans.lower() == "nan":
                error_type = "Blank"
            else:
                ref_judge = eval_refusal(model_ans)
                if "refusal" in ref_judge: error_type = "Refusal"
                elif "blank" in ref_judge: error_type = "Blank"
                else: error_type = "Wrong"

        paper_categories.append(category)
        wrong_types.append(error_type)
        
        
        a1_match_result = eval_redefinition(agent1_ans, ref_redefined, ref_anchored)
        a1_error_type = "N/A"

        if "first" in a1_match_result: a1_category = "Correct responses"
        elif "second" in a1_match_result: a1_category = "Anchored responses"
        else: a1_category = "Completely wrong responses"

       
        if a1_category == "Completely wrong responses":
            if len(agent1_ans.strip()) < 1 or agent1_ans.lower() == "nan": a1_error_type = "Blank"
            else:
                a1_ref_judge = eval_refusal(agent1_ans)
                if "refusal" in a1_ref_judge: a1_error_type = "Refusal"
                elif "blank" in a1_ref_judge: a1_error_type = "Blank"
                else: a1_error_type = "Wrong"
                
        agent1_categories.append(a1_category)
        agent1_wrong_types.append(a1_error_type)

        if (index + 1) % 50 == 0: print(f"Evaluated {index + 1} rows...")
        
   
    full_df['NR_Correct'] = is_nr_correct_list
    full_df['Paper_Category'] = paper_categories
    full_df['Wrong_Type'] = wrong_types
    
    
    full_df['Agent1_Category'] = agent1_categories
    full_df['Agent1_Wrong_Type'] = agent1_wrong_types

    
    os.makedirs(os.path.dirname(os.path.join(BASE_DIR, out_csv_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(BASE_DIR, out_txt_path)), exist_ok=True)

    full_df.to_csv(os.path.join(BASE_DIR, out_csv_path), index=False)

    
    total = len(full_df)
    nr_correct_count = sum(is_nr_correct_list)
    
    exp_counts = full_df['Paper_Category'].value_counts()
    a1_counts = full_df['Agent1_Category'].value_counts() 

    summary_text = f"FINAL EVALUATION FOR: {main_csv_path}\n" + "="*50 + "\n\n"
    summary_text += f"[Category 1] No redefinition (NR) correct: {nr_correct_count}/{total} ({(nr_correct_count/total)*100:.2f}%)\n\n"
    
    summary_text += "--- FINAL OUTPUT (AGENT 2) STATS ---\n"
    summary_text += f"[Category 2] Anchored responses: {exp_counts.get('Anchored responses', 0)}/{total} ({(exp_counts.get('Anchored responses', 0)/total)*100:.2f}%)\n"
    summary_text += f"[Category 3] Correct responses: {exp_counts.get('Correct responses', 0)}/{total} ({(exp_counts.get('Correct responses', 0)/total)*100:.2f}%)\n"
    summary_text += f"[Category 4] Completely wrong: {exp_counts.get('Completely wrong responses', 0)}/{total} ({(exp_counts.get('Completely wrong responses', 0)/total)*100:.2f}%)\n\n"

    wrong_df = full_df[full_df['Paper_Category'] == "Completely wrong responses"]
    if len(wrong_df) > 0:
        summary_text += "- Final Output Wrong Breakdown -\n" + str(wrong_df['Wrong_Type'].value_counts()) + "\n"
        summary_text += "- Final Output Percentages -\n" + str(wrong_df['Wrong_Type'].value_counts(normalize=True) * 100) + "\n\n"

    
    summary_text += "--- AGENT 1 STATS ---\n"
    summary_text += f"[Category 2] Anchored responses: {a1_counts.get('Anchored responses', 0)}/{total} ({(a1_counts.get('Anchored responses', 0)/total)*100:.2f}%)\n"
    summary_text += f"[Category 3] Correct responses: {a1_counts.get('Correct responses', 0)}/{total} ({(a1_counts.get('Correct responses', 0)/total)*100:.2f}%)\n"
    summary_text += f"[Category 4] Completely wrong: {a1_counts.get('Completely wrong responses', 0)}/{total} ({(a1_counts.get('Completely wrong responses', 0)/total)*100:.2f}%)\n\n"
    
    a1_wrong_df = full_df[full_df['Agent1_Category'] == "Completely wrong responses"]
    if len(a1_wrong_df) > 0:
        summary_text += "- Agent 1 Wrong Breakdown -\n" + str(a1_wrong_df['Agent1_Wrong_Type'].value_counts()) + "\n"
        summary_text += "- Agent 1 Percentages -\n" + str(a1_wrong_df['Agent1_Wrong_Type'].value_counts(normalize=True) * 100) + "\n"

    with open(os.path.join(BASE_DIR, out_txt_path), "w") as f: f.write(summary_text)
    print(f"[{main_csv_path.split('/')[-1]}] -> Finished and Saved!\n")


EXPERIMENTS_TO_RUN = [
    
    #Add here the results csvs in this form: (redefinition_results_csv, nr_results_csv, evaluated_output_csv, stats_output_txt)
]

print(f"\nBatch processing {len(EXPERIMENTS_TO_RUN)} experiments...")
for main_csv, nr_csv, out_csv, out_txt in EXPERIMENTS_TO_RUN:
    grade_experiment(main_csv, nr_csv, out_csv, out_txt)

print("All batch evaluations completely finished!")
