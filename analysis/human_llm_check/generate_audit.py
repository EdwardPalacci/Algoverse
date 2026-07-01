import os
import glob
import json
import random
import csv

base_dirs = ['analysis/llm_as_judge/results/*.jsonl', 'llm_as_judge/results/*.jsonl', 'results/*.jsonl']
files = []
for pattern in base_dirs:
    files.extend(glob.glob(pattern))

if not files:
    files = glob.glob('**/*.jsonl', recursive=True)

valid_lines = []
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as src:
            for line in src:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Filter: Must be parsed successfully AND must have a verdict from the judge
                    if data.get('parse_success') is True and data.get('judge_verdict'):
                        valid_lines.append(data)
                except Exception:
                    continue
    except Exception:
        continue

if not valid_lines:
    print("[-] Error: No records found that have BOTH parse_success=True and a judge_verdict.")
    exit(1)

print(f"[+] Found {len(valid_lines)} fully judged records. Sampling 200...")

sample_size = min(200, len(valid_lines))
sample_data = random.sample(valid_lines, sample_size)

os.makedirs('analysis/human_llm_check', exist_ok=True)

with open('analysis/human_llm_check/raw_200_sample.jsonl', 'w', encoding='utf-8') as out:
    for item in sample_data:
        out.write(json.dumps(item) + '\n')

fields = ['question_id', 'dataset', 'condition', 'model_name', 'prompt', 'ground_truth', 'answer_type', 'confidence', 'short_explanation', 'model_answer', 'judge_verdict', 'judge_reason', 'human_verdict']

csv_path = 'analysis/human_llm_check/perfect_audit_sheet.csv'
with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(fields)
    
    for data in sample_data:
        try:
            row = []
            for field in fields:
                if field == 'human_verdict':
                    row.append('')
                elif field == 'model_answer':
                    ans = data.get('model_answer')
                    if ans is None:
                        ans = data.get('answer')
                    if isinstance(ans, str) and ans.strip().startswith('{'):
                        try:
                            inner = json.loads(ans)
                            ans = inner.get('answer', ans)
                        except Exception:
                            pass
                    row.append(str(ans if ans is not None else ''))
                else:
                    row.append(str(data.get(field, '')))
            writer.writerow(row)
        except Exception:
            continue

print(f"[+] Complete. Every row is guaranteed to have a verdict. Saved to: {csv_path}")
