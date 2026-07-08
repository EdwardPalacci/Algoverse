import argparse
import csv
import glob
import json
import os
import random


DEFAULT_SEED = 42
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_PATTERNS = [
    "analysis/llm_as_judge/results/*/by_model/*/all_datasets.jsonl",
    "analysis/llm_as_judge/results/*.jsonl",
    "llm_as_judge/results/*.jsonl",
    "results/*.jsonl",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a reproducible sample for human-vs-LLM judge auditing."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--output-dir",
        default="analysis/human_llm_check",
        help="Directory for raw sample and audit-sheet CSVs.",
    )
    return parser.parse_args()


def iter_judged_records():
    files = []
    for pattern in DEFAULT_PATTERNS:
        files.extend(glob.glob(pattern))

    if not files:
        files = glob.glob("**/*.jsonl", recursive=True)

    # Use all_datasets files when present to avoid sampling duplicate by-dataset rows.
    all_dataset_files = [path for path in files if path.endswith("/all_datasets.jsonl")]
    if all_dataset_files:
        files = all_dataset_files

    seen_keys = set()
    for path in sorted(set(files)):
        with open(path, "r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if record.get("parse_success") is not True:
                    continue
                if record.get("judge_verdict") not in {"correct", "incorrect"}:
                    continue

                key = (
                    record.get("model_name"),
                    record.get("dataset"),
                    record.get("question_id"),
                    record.get("condition"),
                    record.get("sample_id"),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                yield record


def extract_model_answer(record):
    answer = record.get("model_answer")
    if answer is None:
        answer = record.get("answer")
    if isinstance(answer, str) and answer.strip().startswith("{"):
        try:
            answer = json.loads(answer).get("answer", answer)
        except json.JSONDecodeError:
            pass
    return "" if answer is None else str(answer)


def write_csv(path, rows, fields, include_human_column=True):
    with open(path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(fields)
        for record in rows:
            row = []
            for field in fields:
                if field == "human_verdict" and include_human_column:
                    row.append("")
                elif field == "model_answer":
                    row.append(extract_model_answer(record))
                else:
                    row.append(str(record.get(field, "")))
            writer.writerow(row)


def main():
    args = parse_args()
    records = list(iter_judged_records())
    if not records:
        raise SystemExit("No parsed judged records found.")

    sample_size = min(args.sample_size, len(records))
    rng = random.Random(args.seed)
    sample = rng.sample(records, sample_size)

    os.makedirs(args.output_dir, exist_ok=True)

    raw_path = os.path.join(args.output_dir, "raw_200_sample.jsonl")
    with open(raw_path, "w", encoding="utf-8") as out:
        for item in sample:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    blind_fields = [
        "question_id",
        "dataset",
        "condition",
        "model_name",
        "prompt",
        "ground_truth",
        "answer_type",
        "confidence",
        "short_explanation",
        "model_answer",
        "human_verdict",
    ]
    review_fields = blind_fields[:-1] + [
        "judge_verdict",
        "judge_reason",
        "human_verdict",
    ]

    write_csv(
        os.path.join(args.output_dir, "blind_audit_sheet.csv"),
        sample,
        blind_fields,
    )
    write_csv(
        os.path.join(args.output_dir, "perfect_audit_sheet.csv"),
        sample,
        review_fields,
    )

    metadata_path = os.path.join(args.output_dir, "sample_metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as out:
        out.write(f"seed={args.seed}\n")
        out.write(f"sample_size={sample_size}\n")
        out.write(f"sample_frame_records={len(records)}\n")
        out.write("sampling_method=simple random sample over parsed judged generations\n")

    print(f"Sampled {sample_size} rows from {len(records)} judged records.")
    print(f"Seed: {args.seed}")
    print(f"Saved raw sample to {raw_path}")


if __name__ == "__main__":
    main()
