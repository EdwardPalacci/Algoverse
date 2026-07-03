import glob
import json
import os
import re
import csv


def normalize(text):
    return str(text).strip().lower().replace("$", "").replace(",", "")


def extract_explanation_answer(short_explanation):
    text = str(short_explanation)

    # Try common answer phrases first
    patterns = [
        r"(?:answer is|final answer is|therefore,? the answer is|so the answer is)\s*([^\.\n]+)",
        r"(?:equals|=)\s*([-+]?\d+(?:\.\d+)?%?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: use last number in explanation
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?%?", text)
    if numbers:
        return numbers[-1]

    # Fallback: yes/no
    yes_no = re.findall(r"\b(yes|no|true|false)\b", text, flags=re.IGNORECASE)
    if yes_no:
        return yes_no[-1]

    return ""


def answer_explanation_mismatch(answer, short_explanation):
    answer_norm = normalize(answer)
    explanation_answer = extract_explanation_answer(short_explanation)
    explanation_norm = normalize(explanation_answer)

    if not answer_norm or not explanation_norm:
        return False

    if answer_norm == explanation_norm:
        return False

    # Numeric comparison
    answer_nums = re.findall(r"[-+]?\d+(?:\.\d+)?", answer_norm)
    explanation_nums = re.findall(r"[-+]?\d+(?:\.\d+)?", explanation_norm)

    if answer_nums and explanation_nums:
        return float(answer_nums[-1]) != float(explanation_nums[-1])

    # Yes/no comparison
    if answer_norm in {"yes", "no", "true", "false"} and explanation_norm in {"yes", "no", "true", "false"}:
        return answer_norm != explanation_norm

    return answer_norm not in explanation_norm and explanation_norm not in answer_norm


def analyze_file(file_path):
    model_name = os.path.basename(os.path.dirname(file_path))

    total = 0
    zero_confidence = 0
    saturation = 0
    parsing_failures = 0
    mismatches = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            example = json.loads(line)

            answer = example.get("answer", "")
            confidence = example.get("confidence", None)
            short_explanation = example.get("short_explanation", "")
            parse_success = example.get("parse_success", True)

            if confidence == 0 or confidence == 0.0:
                zero_confidence += 1

            if confidence == 1 or confidence == 1.0:
                saturation += 1

            if parse_success is False:
                parsing_failures += 1

            if answer_explanation_mismatch(answer, short_explanation):
                mismatches += 1

    return {
        "model": model_name,
        "total": total,
        "zero_confidence": zero_confidence,
        "saturation": saturation,
        "parsing_failures": parsing_failures,
        "answer_explanation_mismatch": mismatches,
    }

def main():
    files = (
        glob.glob("analysis/llm_as_judge/results/ar/by_model/*/all_datasets.jsonl")
        + glob.glob("analysis/llm_as_judge/results/dlm/by_model/*/all_datasets.jsonl")
    )

    results = []

    for file_path in files:
        stats = analyze_file(file_path)
        results.append(stats)

    with open(
        "analysis/failure_analysis/failure_statistics.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Saved results to analysis/failure_analysis/failure_statistics.csv")
if __name__ == "__main__":
    main()