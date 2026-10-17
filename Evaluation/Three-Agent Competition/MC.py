import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition"
MODEL_ID = os.path.join(BASE_DIR, "models", "Meta-Llama-3.1-70B-Instruct")


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

# Updated from your patch script to handle NaN and empty strings safely
def check_nr(ans, real_ans):
    if not str(ans).strip() or str(ans).lower() == "nan" or not str(real_ans).strip() or str(real_ans).lower() == "nan":
        return False
    res = eval_nr(str(ans), str(real_ans)).strip().lower()
    return res.startswith("correct") and not res.startswith("incorrect")

def categorize_answer(ans, ref_red, ref_anch):
    match_result = eval_redefinition(ans, ref_red, ref_anch)
    error_type = "N/A"

    if "first" in match_result: 
        category = "Correct responses"
    elif "second" in match_result: 
        category = "Anchored responses"
    else: 
        category = "Completely wrong responses"
        if len(ans.strip()) < 1 or ans.lower() == "nan": 
            error_type = "Blank"
        else:
            ref_judge = eval_refusal(ans)
            if "refusal" in ref_judge: error_type = "Refusal"
            elif "blank" in ref_judge: error_type = "Blank"
            else: error_type = "Wrong"
            
    return category, error_type


def grade_experiment(main_csv_path, nr_csv_path, out_csv_path, out_txt_path):
    print(f"\n[{main_csv_path.split('/')[-1]}] -> Starting Evaluation...")

    nr_df = pd.read_csv(os.path.join(BASE_DIR, nr_csv_path))
    exp_df = pd.read_csv(os.path.join(BASE_DIR, main_csv_path))

   
    nr_columns_to_keep = ['question', 'nr_judge_final_clean', 'nr_agent_A_initial_clean', 'nr_agent_B_initial_clean']
    nr_columns_available = [col for col in nr_columns_to_keep if col in nr_df.columns]
    nr_df_subset = nr_df[nr_columns_available]

    full_df = pd.merge(exp_df, nr_df_subset, on="question", how="left")

    judge_cats, judge_wrongs = [], []
    agentA_cats, agentA_wrongs = [], []
    agentB_cats, agentB_wrongs = [], []
    nr_judge_correct, nr_agentA_correct, nr_agentB_correct = [], [], []

    for index, row in full_df.iterrows():
       
        ref_redefined = str(row.get('expected_mc_answer', ''))
        ref_anchored = str(row.get('anchored_mc_answer', ''))
        
       
        ref_nr_expected = str(row.get('expected_nr_answer', row.get('expected_answer', '')))
        
        judge_ans = str(row.get('judge_final_clean', ''))
        agentA_ans = str(row.get('agent_A_initial_clean', ''))
        agentB_ans = str(row.get('agent_B_initial_clean', ''))
        
        nr_judge_ans = str(row.get('nr_judge_final_clean', ''))
        nr_agentA_ans = str(row.get('nr_agent_A_initial_clean', ''))
        nr_agentB_ans = str(row.get('nr_agent_B_initial_clean', ''))


        nr_judge_correct.append(check_nr(nr_judge_ans, ref_nr_expected))
        nr_agentA_correct.append(check_nr(nr_agentA_ans, ref_nr_expected))
        nr_agentB_correct.append(check_nr(nr_agentB_ans, ref_nr_expected))

  
        j_cat, j_err = categorize_answer(judge_ans, ref_redefined, ref_anchored)
        judge_cats.append(j_cat)
        judge_wrongs.append(j_err)

        a_cat, a_err = categorize_answer(agentA_ans, ref_redefined, ref_anchored)
        agentA_cats.append(a_cat)
        agentA_wrongs.append(a_err)

        b_cat, b_err = categorize_answer(agentB_ans, ref_redefined, ref_anchored)
        agentB_cats.append(b_cat)
        agentB_wrongs.append(b_err)

        if (index + 1) % 50 == 0: print(f"Evaluated {index + 1} rows...")

   
    full_df['NR_Judge_Correct'] = nr_judge_correct
    full_df['NR_AgentA_Correct'] = nr_agentA_correct
    full_df['NR_AgentB_Correct'] = nr_agentB_correct
    full_df['Judge_Category'] = judge_cats
    full_df['Judge_Wrong_Type'] = judge_wrongs
    full_df['AgentA_Category'] = agentA_cats
    full_df['AgentA_Wrong_Type'] = agentA_wrongs
    full_df['AgentB_Category'] = agentB_cats
    full_df['AgentB_Wrong_Type'] = agentB_wrongs
    
    os.makedirs(os.path.dirname(os.path.join(BASE_DIR, out_csv_path)), exist_ok=True)
    full_df.to_csv(os.path.join(BASE_DIR, out_csv_path), index=False)

  
    total = len(full_df)
    j_counts = full_df['Judge_Category'].value_counts()
    a_counts = full_df['AgentA_Category'].value_counts()
    b_counts = full_df['AgentB_Category'].value_counts()

    summary_text = f"FINAL EVALUATION FOR: {main_csv_path}\n" + "="*50 + "\n\n"
    
    summary_text += "--- NO REDEFINITION (NR) STATS ---\n"
    summary_text += f"Judge NR Correct: {sum(nr_judge_correct)}/{total} ({(sum(nr_judge_correct)/total)*100:.2f}%)\n"
    summary_text += f"Agent A NR Correct: {sum(nr_agentA_correct)}/{total} ({(sum(nr_agentA_correct)/total)*100:.2f}%)\n"
    summary_text += f"Agent B NR Correct: {sum(nr_agentB_correct)}/{total} ({(sum(nr_agentB_correct)/total)*100:.2f}%)\n\n"
    
    summary_text += "--- JUDGE (FINAL OUTPUT) STATS ---\n"
    summary_text += f"Anchored responses: {j_counts.get('Anchored responses', 0)}/{total} ({(j_counts.get('Anchored responses', 0)/total)*100:.2f}%)\n"
    summary_text += f"Correct responses: {j_counts.get('Correct responses', 0)}/{total} ({(j_counts.get('Correct responses', 0)/total)*100:.2f}%)\n"
    summary_text += f"Completely wrong: {j_counts.get('Completely wrong responses', 0)}/{total} ({(j_counts.get('Completely wrong responses', 0)/total)*100:.2f}%)\n\n"

    j_wrong_df = full_df[full_df['Judge_Category'] == "Completely wrong responses"]
    if len(j_wrong_df) > 0:
        summary_text += "- Judge Wrong Breakdown -\n"
        j_total_wrong = len(j_wrong_df)
        for err_type, count in j_wrong_df['Judge_Wrong_Type'].value_counts().items():
            summary_text += f"{err_type}: {count} ({(count / j_total_wrong) * 100:.2f}%)\n"
        summary_text += "\n"

    summary_text += "--- AGENT A (INSTRUCTION FOCUS) STATS ---\n"
    summary_text += f"Anchored responses: {a_counts.get('Anchored responses', 0)}/{total} ({(a_counts.get('Anchored responses', 0)/total)*100:.2f}%)\n"
    summary_text += f"Correct responses: {a_counts.get('Correct responses', 0)}/{total} ({(a_counts.get('Correct responses', 0)/total)*100:.2f}%)\n"
    summary_text += f"Completely wrong: {a_counts.get('Completely wrong responses', 0)}/{total} ({(a_counts.get('Completely wrong responses', 0)/total)*100:.2f}%)\n\n"
    
    a_wrong_df = full_df[full_df['AgentA_Category'] == "Completely wrong responses"]
    if len(a_wrong_df) > 0:
        summary_text += "- Agent A Wrong Breakdown -\n"
        a_total_wrong = len(a_wrong_df)
        for err_type, count in a_wrong_df['AgentA_Wrong_Type'].value_counts().items():
            summary_text += f"{err_type}: {count} ({(count / a_total_wrong) * 100:.2f}%)\n"
        summary_text += "\n"

    summary_text += "--- AGENT B (FACT FOCUS) STATS ---\n"
    summary_text += f"Anchored responses: {b_counts.get('Anchored responses', 0)}/{total} ({(b_counts.get('Anchored responses', 0)/total)*100:.2f}%)\n"
    summary_text += f"Correct responses: {b_counts.get('Correct responses', 0)}/{total} ({(b_counts.get('Correct responses', 0)/total)*100:.2f}%)\n"
    summary_text += f"Completely wrong: {b_counts.get('Completely wrong responses', 0)}/{total} ({(b_counts.get('Completely wrong responses', 0)/total)*100:.2f}%)\n\n"
    
    b_wrong_df = full_df[full_df['AgentB_Category'] == "Completely wrong responses"]
    if len(b_wrong_df) > 0:
        summary_text += "- Agent B Wrong Breakdown -\n"
        b_total_wrong = len(b_wrong_df)
        for err_type, count in b_wrong_df['AgentB_Wrong_Type'].value_counts().items():
            summary_text += f"{err_type}: {count} ({(count / b_total_wrong) * 100:.2f}%)\n"
        summary_text += "\n"

    os.makedirs(os.path.dirname(os.path.join(BASE_DIR, out_txt_path)), exist_ok=True)
    with open(os.path.join(BASE_DIR, out_txt_path), "w") as f: 
        f.write(summary_text)
    print(f"[{main_csv_path.split('/')[-1]}] -> Finished and Saved!\n")


EXPERIMENTS_TO_RUN = [
    # Format: (main_csv, nr_csv, evaluated_csv, stats_txt)
 
]

print(f"\nBatch processing {len(EXPERIMENTS_TO_RUN)} experiments...")
for main_csv, nr_csv, out_csv, out_txt in EXPERIMENTS_TO_RUN:
    grade_experiment(main_csv, nr_csv, out_csv, out_txt)

print("All batch evaluations completely finished!")
