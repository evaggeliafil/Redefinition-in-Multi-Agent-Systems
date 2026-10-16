import os 
import torch 
import pandas as pd 
import ast 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig 


BASE_DIR = "/lustre/fswork/projects/rech/xbk/uai19ky/MASRedefinition" 
MODEL_ID = os.path.join(BASE_DIR, "models", "Meta-Llama-3.1-70B-Instruct") 

ROLE_MAP = {
    "You are an advanced AI capable of performing complex calculations, logical reasoning, and providing efficient solutions to mathematical problems.": "AI",
    "You are an experienced math teacher with a deep understanding of mathematical concepts, methods, and pedagogy. You can offer insights into problem-solving strategies and educational approaches to mathematics.": "Teacher",
    "You are a scientist who specializes in advanced mathematical theories and research. You can contribute deep theoretical knowledge and expertise in cutting-edge mathematical areas, such as topology, abstract algebra, or number theory.": "Scientist",
    "You are a professional engineer with practical experience in applying mathematical concepts to solve real-world problems. You can offer a practical viewpoint on how mathematics is used in designing, analyzing, and improving systems and structures.": "Engineer",
    "You are an expert in computer science, well-versed in algorithms, computational theory, and programming. You can assist in solving problems that involve algorithmic thinking, computational complexity, or numerical methods.": "CS_Expert"
}


print("Loading Llama 3.1 70B Evaluator into VRAM... (This might take a few minutes)") 

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

def get_category_and_error(match_result, model_ans):
    error_type = "N/A"
    if "first" in match_result: 
        category = "Correct responses" 
    elif "second" in match_result: 
        category = "Anchored responses"  
    else: 
        category = "Completely wrong responses" 
        if len(model_ans.strip()) < 1 or model_ans.lower() == "nan": 
            error_type = "Blank" 
        else: 
            ref_judge = eval_refusal(model_ans) 
            if "refusal" in ref_judge: error_type = "Refusal" 
            elif "blank" in ref_judge: error_type = "Blank" 
            else: error_type = "Wrong"
    return category, error_type

def grade_experiment(main_csv_path, nr_csv_path, out_csv_path, out_txt_path): 
    print(f"\n[{main_csv_path.split('/')[-1]}] -> Starting MC Evaluation...") 
    
   
    nr_df = pd.read_csv(os.path.join(BASE_DIR, nr_csv_path)) 
    exp_df = pd.read_csv(os.path.join(BASE_DIR, main_csv_path)) 

    full_df = pd.merge(exp_df, nr_df, on="question", how="left", suffixes=('_exp', '_nr')) 

    results_list = []
    
   
    role_performance = {short_name: {"total": 0, "correct": 0, "anchored": 0} for short_name in ROLE_MAP.values()}
    
    exp_consensus_stats = {"Agreed Correct": 0, "Agreed Anchored": 0, "Agreed Wrong": 0, "Disagreement": 0}
    nr_consensus_stats = {"Agreed Correct": 0, "Agreed Wrong": 0, "Disagreement": 0}

    for index, row in full_df.iterrows(): 
     
        roles_str = str(row.get('roles_used_exp', '[]'))
        try:
            roles = ast.literal_eval(roles_str)
            role_1 = ROLE_MAP.get(roles[0], "Unknown") if len(roles) > 0 else "Unknown"
            role_2 = ROLE_MAP.get(roles[1], "Unknown") if len(roles) > 1 else "Unknown"
        except:
            role_1, role_2 = "Unknown", "Unknown"

        
        ref_redefined = str(row.get('expected_mc_answer_exp', row.get('expected_mc_answer', ''))) 
        ref_anchored = str(row.get('anchored_mc_answer_exp', row.get('anchored_mc_answer', ''))) 

       
        
        # --- Evaluate INITIAL Agent 1 ---
        a1_init_ans = str(row.get('agent1_initial_clean_exp', ''))
        a1_init_match = eval_redefinition(a1_init_ans, ref_redefined, ref_anchored)
        a1_init_cat, a1_init_err = get_category_and_error(a1_init_match, a1_init_ans)

        # --- Evaluate INITIAL Agent 2 ---
        a2_init_ans = str(row.get('agent2_initial_clean_exp', ''))
        a2_init_match = eval_redefinition(a2_init_ans, ref_redefined, ref_anchored)
        a2_init_cat, a2_init_err = get_category_and_error(a2_init_match, a2_init_ans)

        # --- Evaluate FINAL Agent 1 ---
        a1_ans = str(row.get('agent1_final_clean_exp', ''))
        a1_match = eval_redefinition(a1_ans, ref_redefined, ref_anchored)
        a1_cat, a1_err = get_category_and_error(a1_match, a1_ans)

        # --- Evaluate FINAL Agent 2 ---
        a2_ans = str(row.get('agent2_final_clean_exp', ''))
        a2_match = eval_redefinition(a2_ans, ref_redefined, ref_anchored)
        a2_cat, a2_err = get_category_and_error(a2_match, a2_ans)

       
        if role_1 != "Unknown":
            role_performance[role_1]["total"] += 1
            if a1_cat == "Correct responses": role_performance[role_1]["correct"] += 1
            elif a1_cat == "Anchored responses": role_performance[role_1]["anchored"] += 1
                
        if role_2 != "Unknown":
            role_performance[role_2]["total"] += 1
            if a2_cat == "Correct responses": role_performance[role_2]["correct"] += 1
            elif a2_cat == "Anchored responses": role_performance[role_2]["anchored"] += 1

        # --- Evaluate TRUE Exp Consensus ---
        exp_consensus_status = "Disagreement"
        if a1_cat == a2_cat:
            if a1_cat == "Correct responses": exp_consensus_status = "Agreed Correct"
            elif a1_cat == "Anchored responses": exp_consensus_status = "Agreed Anchored"
            else: exp_consensus_status = "Agreed Wrong"
        
        exp_consensus_stats[exp_consensus_status] += 1

     
        # --- Evaluate NR Agent 1 ---
        nr_a1_ans = str(row.get('agent1_final_clean_nr', '')) 
        nr_a1_judge = eval_nr(nr_a1_ans, ref_anchored) 
        nr_a1_correct = True if "correct" in nr_a1_judge and "incorrect" not in nr_a1_judge else False 

        # --- Evaluate NR Agent 2 ---
        nr_a2_ans = str(row.get('agent2_final_clean_nr', '')) 
        nr_a2_judge = eval_nr(nr_a2_ans, ref_anchored) 
        nr_a2_correct = True if "correct" in nr_a2_judge and "incorrect" not in nr_a2_judge else False 

        
        nr_consensus_status = "Disagreement"
        if nr_a1_correct == nr_a2_correct:
            if nr_a1_correct: nr_consensus_status = "Agreed Correct"
            else: nr_consensus_status = "Agreed Wrong"
                
        nr_consensus_stats[nr_consensus_status] += 1

        
        results_list.append({
            "question": row.get("question", ""),
            "difficulty_level": row.get('difficulty_level_exp', row.get('difficulty_level', 'Unknown')),
            "question_difficulty": row.get('question_difficulty_exp', row.get('question_difficulty', 'Unknown')),
            
            "role_1": role_1,
            "role_2": role_2,
            
            "NR_A1_Correct": nr_a1_correct,
            "NR_A2_Correct": nr_a2_correct,
            "NR_Consensus_Status": nr_consensus_status,
            
            "Exp_A1_Initial_Category": a1_init_cat,
            "Exp_A1_Initial_Wrong_Type": a1_init_err,
            "Exp_A2_Initial_Category": a2_init_cat,
            "Exp_A2_Initial_Wrong_Type": a2_init_err,

            "Exp_A1_Category": a1_cat,
            "Exp_A1_Wrong_Type": a1_err,
            "Exp_A2_Category": a2_cat,
            "Exp_A2_Wrong_Type": a2_err,
            "Exp_Consensus_Status": exp_consensus_status
        })

        if (index + 1) % 10 == 0: 
            print(f"Evaluated {index + 1}/{len(full_df)} rows...") 

  
    eval_df = pd.DataFrame(results_list)
    
   
    cols_to_use = eval_df.columns.difference(full_df.columns)
    final_output_df = pd.concat([full_df.reset_index(drop=True), eval_df[cols_to_use].reset_index(drop=True)], axis=1)
    
    os.makedirs(os.path.dirname(os.path.join(BASE_DIR, out_csv_path)), exist_ok=True)
    final_output_df.to_csv(os.path.join(BASE_DIR, out_csv_path), index=False) 

    # Calculate & Write Stats 
    total = len(full_df) 
    nr_a1_correct_count = eval_df['NR_A1_Correct'].sum() 
    nr_a2_correct_count = eval_df['NR_A2_Correct'].sum()
    
    summary_text = f"FINAL MC EVALUATION FOR: {main_csv_path}\n" + "="*50 + "\n\n" 
    
    
    summary_text += "=== NR BASELINE (Single Agent, No Redefinition) ===\n"
    summary_text += f"--- NR AGENT 1 PERFORMANCE ---\nCorrect: {nr_a1_correct_count}/{total} ({(nr_a1_correct_count/total)*100:.2f}%)\n"
    summary_text += f"Incorrect: {total - nr_a1_correct_count}/{total} ({((total - nr_a1_correct_count)/total)*100:.2f}%)\n\n"

    summary_text += f"--- NR AGENT 2 PERFORMANCE ---\nCorrect: {nr_a2_correct_count}/{total} ({(nr_a2_correct_count/total)*100:.2f}%)\n"
    summary_text += f"Incorrect: {total - nr_a2_correct_count}/{total} ({((total - nr_a2_correct_count)/total)*100:.2f}%)\n\n"

    summary_text += "--- NR TRUE CONSENSUS ---\n"
    summary_text += f"Agreed Correct: {nr_consensus_stats['Agreed Correct']}/{total} ({(nr_consensus_stats['Agreed Correct']/total)*100:.2f}%)\n"
    summary_text += f"Agreed Wrong: {nr_consensus_stats['Agreed Wrong']}/{total} ({(nr_consensus_stats['Agreed Wrong']/total)*100:.2f}%)\n"
    summary_text += f"Disagreement: {nr_consensus_stats['Disagreement']}/{total} ({(nr_consensus_stats['Disagreement']/total)*100:.2f}%)\n\n\n"

    
    summary_text += "=== EXPERIMENTAL (Peer Review MAS, With Redefinition) ===\n"
    
    # Agent 1
    summary_text += "--- EXP AGENT 1 PERFORMANCE (Final Answers) ---\n"
    a1_counts = eval_df['Exp_A1_Category'].value_counts()
    a1_wrong = a1_counts.get('Completely wrong responses', 0)
    summary_text += f"[Category 1] Correct responses: {a1_counts.get('Correct responses', 0)}/{total} ({(a1_counts.get('Correct responses', 0)/total)*100:.2f}%)\n" 
    summary_text += f"[Category 2] Anchored responses: {a1_counts.get('Anchored responses', 0)}/{total} ({(a1_counts.get('Anchored responses', 0)/total)*100:.2f}%)\n" 
    summary_text += f"[Category 3] Completely wrong: {a1_wrong}/{total} ({(a1_wrong/total)*100:.2f}%)\n" 
    if a1_wrong > 0:
        a1_w_counts = eval_df[eval_df['Exp_A1_Category'] == 'Completely wrong responses']['Exp_A1_Wrong_Type'].value_counts()
        summary_text += f"    -> Actually Wrong: {a1_w_counts.get('Wrong', 0)}/{a1_wrong} ({(a1_w_counts.get('Wrong', 0)/a1_wrong)*100:.2f}%)\n"
        summary_text += f"    -> Refusals:       {a1_w_counts.get('Refusal', 0)}/{a1_wrong} ({(a1_w_counts.get('Refusal', 0)/a1_wrong)*100:.2f}%)\n"
        summary_text += f"    -> Blanks:         {a1_w_counts.get('Blank', 0)}/{a1_wrong} ({(a1_w_counts.get('Blank', 0)/a1_wrong)*100:.2f}%)\n"
    summary_text += "\n"

    # Agent 2
    summary_text += "--- EXP AGENT 2 PERFORMANCE (Final Answers) ---\n"
    a2_counts = eval_df['Exp_A2_Category'].value_counts()
    a2_wrong = a2_counts.get('Completely wrong responses', 0)
    summary_text += f"[Category 1] Correct responses: {a2_counts.get('Correct responses', 0)}/{total} ({(a2_counts.get('Correct responses', 0)/total)*100:.2f}%)\n" 
    summary_text += f"[Category 2] Anchored responses: {a2_counts.get('Anchored responses', 0)}/{total} ({(a2_counts.get('Anchored responses', 0)/total)*100:.2f}%)\n" 
    summary_text += f"[Category 3] Completely wrong: {a2_wrong}/{total} ({(a2_wrong/total)*100:.2f}%)\n" 
    if a2_wrong > 0:
        a2_w_counts = eval_df[eval_df['Exp_A2_Category'] == 'Completely wrong responses']['Exp_A2_Wrong_Type'].value_counts()
        summary_text += f"    -> Actually Wrong: {a2_w_counts.get('Wrong', 0)}/{a2_wrong} ({(a2_w_counts.get('Wrong', 0)/a2_wrong)*100:.2f}%)\n"
        summary_text += f"    -> Refusals:       {a2_w_counts.get('Refusal', 0)}/{a2_wrong} ({(a2_w_counts.get('Refusal', 0)/a2_wrong)*100:.2f}%)\n"
        summary_text += f"    -> Blanks:         {a2_w_counts.get('Blank', 0)}/{a2_wrong} ({(a2_w_counts.get('Blank', 0)/a2_wrong)*100:.2f}%)\n"
    summary_text += "\n"

    summary_text += "--- EXP TRUE CONSENSUS ---\n"
    summary_text += f"Agreed Correct: {exp_consensus_stats['Agreed Correct']}/{total} ({(exp_consensus_stats['Agreed Correct']/total)*100:.2f}%)\n"
    summary_text += f"Agreed Anchored: {exp_consensus_stats['Agreed Anchored']}/{total} ({(exp_consensus_stats['Agreed Anchored']/total)*100:.2f}%)\n"
    summary_text += f"Agreed Wrong: {exp_consensus_stats['Agreed Wrong']}/{total} ({(exp_consensus_stats['Agreed Wrong']/total)*100:.2f}%)\n"
    summary_text += f"Disagreement: {exp_consensus_stats['Disagreement']}/{total} ({(exp_consensus_stats['Disagreement']/total)*100:.2f}%)\n\n\n"

    
    summary_text += "=== ROLE PERFORMANCE (Success vs Anchored Rates) ===\n"
    for role, stats in role_performance.items():
        if stats["total"] > 0:
            pct_correct = (stats["correct"] / stats["total"]) * 100
            pct_anchored = (stats["anchored"] / stats["total"]) * 100
            summary_text += f"Role [{role}] - Total Assignments: {stats['total']}\n"
            summary_text += f"  -> Correct (Redefined): {stats['correct']}/{stats['total']} ({pct_correct:.2f}%)\n"
            summary_text += f"  -> Anchored (Trap):     {stats['anchored']}/{stats['total']} ({pct_anchored:.2f}%)\n"
            summary_text += f"  -> Completely Wrong:    {stats['total'] - (stats['correct'] + stats['anchored'])}/{stats['total']} ({100 - (pct_correct + pct_anchored):.2f}%)\n\n"

    with open(os.path.join(BASE_DIR, out_txt_path), "w") as f: 
        f.write(summary_text) 
    print(f"[{main_csv_path.split('/')[-1]}] -> Finished and Saved!\n")


EXPERIMENTS_TO_RUN = [
    
    #Add here the results csvs in this form: (redefinition_results_csv, nr_results_csv, evaluated_output_csv, stats_output_txt)
]

print(f"\nBatch processing {len(EXPERIMENTS_TO_RUN)} experiments...") 
for main_csv, nr_csv, out_csv, out_txt in EXPERIMENTS_TO_RUN: 
    grade_experiment(main_csv, nr_csv, out_csv, out_txt)

print("All batch evaluations completely finished!")
