import csv
import math

csv_path = 'analysis/human_llm_check/perfect_audit_sheet.csv'

human_votes = []
judge_votes = []

with open(csv_path, 'r', encoding='utf-8') as f:
    # Read all lines and filter out any math garbage at the start
    lines = [line for line in f if 'dataset' in line or 'verdict' in line or ',' in line]
    
    # Reconstruct the reader from clean lines
    reader = csv.reader(lines)
    
    # Find the actual header row dynamically
    header = None
    for row in reader:
        if 'human_verdict' in row and 'judge_verdict' in row:
            header = row
            break
            
    if not header:
        print("[-] Error: Hard-coding indices because headers are buried in math text.")
        header = ['question_id', 'dataset', 'condition', 'model_name', 'prompt', 'ground_truth', 'answer_type', 'confidence', 'short_explanation', 'model_answer', 'judge_verdict', 'judge_reason', 'human_verdict']
    
    h_idx = header.index('human_verdict')
    j_idx = header.index('judge_verdict')
    for row in reader:
        if len(row) <= max(h_idx, j_idx):
            continue
            
        h = row[h_idx].strip().lower()
        j = row[j_idx].strip().lower()
        
        if h in ['correct', 'incorrect'] and j in ['correct', 'incorrect']:
            human_votes.append(h)
            judge_votes.append(j)

total = len(human_votes)
if total == 0:
    print("[-] Error: No matched rows found with both human and judge verdicts completed.")
    exit(1)

# Get unique categories
categories = list(set(human_votes + judge_votes))

# Calculate Observed Agreement (Po)
matches = sum(1 for h, j in zip(human_votes, judge_votes) if h == j)
p_o = matches / total

# Calculate Expected Agreement (Pe)
p_e = 0.0
for cat in categories:
    h_count = human_votes.count(cat)
    j_count = judge_votes.count(cat)
    p_e += (h_count / total) * (j_count / total)

# Kappa Calculation
if p_e == 1.0:
    kappa = 1.0 if p_o == 1.0 else 0.0
else:
    kappa = (p_o - p_e) / (1.0 - p_e)

if kappa < 0:
    interpretation = "Poor Reliability (Worse than chance)"
elif kappa <= 0.20:
    interpretation = "Slight Agreement"
elif kappa <= 0.40:
    interpretation = "Fair Agreement"
elif kappa <= 0.60:   
    interpretation = "Moderate Agreement"
elif kappa <= 0.80:
    interpretation = "Substantial Agreement"
else:
    interpretation = "Almost Perfect Agreement"

print("=" * 40)
print(f" INTER-RATER RELIABILITY RESULTS ({total} Rows Audited)")
print("=" * 40) 
print(f"Observed Agreement (Po): {p_o:.4f}")
print(f"Expected Agreement  (Pe): {p_e:.4f}")
print(f"Cohen's Kappa     (κ): {kappa:.4f}")
print("-" * 40)
print(f"Interpretation: {interpretation}")
print("=" * 40)

output_path = 'analysis/human_llm_check/audit_results.txt'
with open(output_path, 'w', encoding='utf-8') as out:
    out.write("=" * 40 + "\n")
    out.write(f" INTER-RATER RELIABILITY RESULTS ({total} Rows Audited)\n")
    out.write("=" * 40 + "\n") 
    out.write(f"Observed Agreement (Po): {p_o:.4f}\n")
    out.write(f"Expected Agreement  (Pe): {p_e:.4f}\n")
    out.write(f"Cohen's Kappa     (κ): {kappa:.4f}\n")
    out.write("-" * 40 + "\n")
    out.write(f"Interpretation: {interpretation}\n")
    out.write("=" * 40 + "\n")
