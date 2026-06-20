from __future__ import annotations

from grade_answers import auto_grade
from file_io import grouped, read_jsonl
from project_paths import AR_PARSED, AR_RAW, DLM_PARSED, DLM_RAW


def load_all_rows() -> tuple[list[dict], dict[str, int]]:
    """Load AR and DLM generations and add normalized analysis fields."""
    sources = [
        ("outputs/ar_parsed_generations.jsonl", AR_PARSED, AR_RAW),
        ("dlm_outputs/dlm_parsed_generations.jsonl", DLM_PARSED, DLM_RAW),
    ]
    rows = []
    raw_counts = {}
    for source_name, parsed_path, raw_path in sources:
        raw_counts[source_name] = sum(1 for line in raw_path.open() if line.strip())
        for row in read_jsonl(parsed_path):
            row = dict(row)
            row["source_file"] = source_name
            row["model_id"] = row.get("model_name", "")
            row["model_family"] = row.get("model_architecture", "")
            row["prompt_condition"] = row.get("condition", "")
            row["parsed_confidence"] = float(row["confidence"]) if row.get("confidence") is not None else None
            row["correct_auto"], row["grader_rule"] = auto_grade(row)
            rows.append(row)
    return rows, raw_counts


def shared_question_ids(rows: list[dict]) -> set[str]:
    """Return question IDs present for both model families."""
    qsets = {
        family[0]: {row["question_id"] for row in family_rows}
        for family, family_rows in grouped(rows, ("model_family",)).items()
    }
    if {"AR", "DLM"} <= set(qsets):
        return qsets["AR"] & qsets["DLM"]
    return set().union(*qsets.values()) if qsets else set()


def aligned_rows(rows: list[dict]) -> list[dict]:
    """Keep only the shared AR/DLM question set for fair comparison."""
    common = shared_question_ids(rows)
    return [row for row in rows if row["question_id"] in common]
