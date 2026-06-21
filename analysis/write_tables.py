from __future__ import annotations

import json
from collections import Counter

from file_io import grouped, write_csv
from compute_metrics import fmt, metric_row
from project_paths import HIGH_CONFIDENCE_THRESHOLD, METRICS_DIR, PILOT_DATA, TABLE_DIR


def produce_alignment_report(rows: list[dict], aligned_rows: list[dict]) -> None:
    """Write data checks that explain what can and cannot be compared."""
    report = []
    qsets = {
        family[0]: {row["question_id"] for row in family_rows}
        for family, family_rows in grouped(rows, ("model_family",)).items()
    }
    if {"AR", "DLM"} <= set(qsets):
        ar_only = sorted(qsets["AR"] - qsets["DLM"])
        dlm_only = sorted(qsets["DLM"] - qsets["AR"])
        report.append({
            "check_name": "same_question_id_set_raw_sources",
            "status": "pass" if not ar_only and not dlm_only else "fail",
            "n_affected": len(ar_only) + len(dlm_only),
            "details": f"AR-only={len(ar_only)}; DLM-only={len(dlm_only)}",
        })
        write_csv(
            METRICS_DIR / "data_alignment_exclusions.csv",
            [{"question_id": qid, "excluded_from": "comparative_tables", "reason": "AR-only question_id"} for qid in ar_only]
            + [{"question_id": qid, "excluded_from": "comparative_tables", "reason": "DLM-only question_id"} for qid in dlm_only],
            ["question_id", "excluded_from", "reason"],
        )

    aligned_qsets = {
        family[0]: {row["question_id"] for row in family_rows}
        for family, family_rows in grouped(aligned_rows, ("model_family",)).items()
    }
    aligned_pass = len({frozenset(values) for values in aligned_qsets.values()}) <= 1
    shared_count = len(next(iter(aligned_qsets.values()))) if aligned_qsets else 0
    report.append({
        "check_name": "same_question_id_set_comparative_analysis",
        "status": "pass" if aligned_pass else "fail",
        "n_affected": 0 if aligned_pass else 1,
        "details": f"{shared_count} shared question_id values used for comparative tables and figures",
    })

    counts = Counter((row["source_file"], row["question_id"], row["prompt_condition"], row["sample_id"]) for row in rows)
    duplicate_count = sum(value - 1 for value in counts.values() if value > 1)
    report.append({
        "check_name": "duplicate_question_id_within_source",
        "status": "pass" if duplicate_count == 0 else "fail",
        "n_affected": duplicate_count,
        "details": "duplicate full evaluation keys",
    })

    valid_datasets = {row["dataset"] for row in json.loads(PILOT_DATA.read_text())}
    checks = [
        ("valid_dataset", lambda row: row.get("dataset") in valid_datasets),
        ("valid_model_id", lambda row: row.get("model_id") in {
            "qwen/qwen-2.5-7b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "mistralai/ministral-8b-2512",
            "mercury-2",
            "inception/mercury-2",
        }),
        ("valid_model_family", lambda row: row.get("model_family") in {"AR", "DLM"}),
        ("valid_prompt_condition", lambda row: row.get("prompt_condition") in {"neutral", "cautious", "overconfident"}),
        ("confidence_normalized_0_1", lambda row: isinstance(row.get("parsed_confidence"), float) and 0.0 <= row["parsed_confidence"] <= 1.0),
        ("correctness_binary_auto_label", lambda row: isinstance(row.get("correct_auto"), bool)),
    ]
    for name, predicate in checks:
        bad_rows = [row for row in rows if not predicate(row)]
        report.append({"check_name": name, "status": "pass" if not bad_rows else "fail", "n_affected": len(bad_rows), "details": "" if not bad_rows else "invalid rows present"})

    write_csv(METRICS_DIR / "data_alignment_report.csv", report, ["check_name", "status", "n_affected", "details"])


def produce_tables(rows: list[dict], raw_counts: dict[str, int]) -> None:
    table1 = []
    for (dataset, model, family, condition), group_rows in sorted(grouped(rows, ("dataset", "model_id", "model_family", "prompt_condition")).items()):
        table1.append({
            "dataset": dataset,
            "N": len(group_rows),
            "model": model,
            "family": family,
            "prompt_condition": condition,
            "confidence_scale": "verbalized probability in [0, 1]",
            "correctness_grader": "deterministic automatic string/numeric/choice grader",
            "metrics": "accuracy; mean confidence; expected calibration error; Brier score; area under the receiver operating characteristic curve; high-confidence wrong rate; parse success",
        })
    write_csv(TABLE_DIR / "table_1_benchmark_specification.csv", table1, ["dataset", "N", "model", "family", "prompt_condition", "confidence_scale", "correctness_grader", "metrics"])

    table2 = []
    for (model, family, source_file), group_rows in sorted(grouped(rows, ("model_id", "model_family", "source_file")).items()):
        metrics = metric_row(group_rows, raw_counts.get(source_file))
        table2.append({"model": model, "family": family, **{key: fmt(value) for key, value in metrics.items()}})
    write_csv(TABLE_DIR / "table_2_aggregate_metrics.csv", table2, ["model", "family", "N", "accuracy", "mean_confidence", "expected_calibration_error", "brier_score", "area_under_roc", "high_confidence_wrong_rate", "parse_success"])

    table3 = []
    for (dataset, model, family), group_rows in sorted(grouped(rows, ("dataset", "model_id", "model_family")).items()):
        metrics = metric_row(group_rows)
        table3.append({"dataset": dataset, "model": model, "family": family, **{key: fmt(value) for key, value in metrics.items() if key != "parse_success"}})
    write_csv(TABLE_DIR / "table_3_per_dataset_metrics.csv", table3, ["dataset", "model", "family", "N", "accuracy", "mean_confidence", "expected_calibration_error", "brier_score", "area_under_roc", "high_confidence_wrong_rate"])

    table4 = []
    for (model, family, condition), group_rows in sorted(grouped(rows, ("model_id", "model_family", "prompt_condition")).items()):
        metrics = metric_row(group_rows)
        table4.append({"model": model, "family": family, "prompt_condition": condition, **{key: fmt(value) for key, value in metrics.items() if key not in {"area_under_roc", "parse_success"}}})
    write_csv(TABLE_DIR / "table_4_prompt_condition_metrics.csv", table4, ["model", "family", "prompt_condition", "N", "accuracy", "mean_confidence", "expected_calibration_error", "brier_score", "high_confidence_wrong_rate"])


def short(value: object, max_chars: int = 140) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def produce_audit_and_cases(rows: list[dict]) -> None:
    confidence_change_rows = []
    for (_model, _family, _qid, _sample, _answer, _correct), group_rows in grouped(rows, ("model_id", "model_family", "question_id", "sample_id", "answer", "correct_auto")).items():
        conditions = {row["prompt_condition"] for row in group_rows}
        confidences = [row["parsed_confidence"] for row in group_rows if row.get("parsed_confidence") is not None]
        if len(conditions) >= 2 and confidences and max(confidences) - min(confidences) >= 0.20:
            confidence_change_rows.append(max(group_rows, key=lambda row: row["parsed_confidence"]))

    audit_rows = []
    categories = [
        ("high_confidence_wrong", lambda row: not row["correct_auto"] and row["parsed_confidence"] >= HIGH_CONFIDENCE_THRESHOLD),
        ("low_confidence_correct", lambda row: row["correct_auto"] and row["parsed_confidence"] <= 0.5),
        ("answer_and_reasoning_mismatch", lambda row: False),
        ("ambiguous_grading_case", lambda row: row["grader_rule"] in {"string_containment_match", "string_mismatch"}),
        ("random_example", lambda row: True),
    ]
    used = set()
    for category, predicate in categories:
        for row in [row for row in rows if predicate(row) and id(row) not in used][:10]:
            used.add(id(row))
            audit_rows.append({
                "question_id": row["question_id"],
                "dataset": row["dataset"],
                "model": row["model_id"],
                "family": row["model_family"],
                "prompt_condition": row["prompt_condition"],
                "model_answer": short(row.get("answer"), 300),
                "gold_answer": short(row.get("ground_truth"), 300),
                "automatic_grade": int(bool(row["correct_auto"])),
                "manual_grade": "pending",
                "issue_type": category,
                "notes": f"automatic grader rule: {row['grader_rule']}",
            })
    for row in confidence_change_rows[:10]:
        audit_rows.append({
            "question_id": row["question_id"],
            "dataset": row["dataset"],
            "model": row["model_id"],
            "family": row["model_family"],
            "prompt_condition": "cross-condition",
            "model_answer": short(row.get("answer"), 300),
            "gold_answer": short(row.get("ground_truth"), 300),
            "automatic_grade": int(bool(row["correct_auto"])),
            "manual_grade": "pending",
            "issue_type": "confidence_changes_without_correctness_changes",
            "notes": "same answer and rule-based correctness label observed with >=0.20 confidence range across prompt conditions",
        })
    write_csv(METRICS_DIR / "manual_grading_audit.csv", audit_rows, ["question_id", "dataset", "model", "family", "prompt_condition", "model_answer", "gold_answer", "automatic_grade", "manual_grade", "issue_type", "notes"])

    summary = []
    counts = Counter(row["issue_type"] for row in audit_rows)
    for category in ["high_confidence_wrong", "low_confidence_correct", "answer_and_reasoning_mismatch", "confidence_changes_without_correctness_changes", "ambiguous_grading_case", "random_example"]:
        summary.append({"audit_category": category, "N": counts[category], "agreement_rate": "NA", "main_issue": "manual adjudication not yet performed"})
    write_csv(TABLE_DIR / "manual_grading_audit_summary.csv", summary, ["audit_category", "N", "agreement_rate", "main_issue"])

    case_sources = {
        "high_confidence_wrong": [row for row in rows if not row["correct_auto"] and row["parsed_confidence"] >= HIGH_CONFIDENCE_THRESHOLD],
        "low_confidence_correct": [row for row in rows if row["correct_auto"] and row["parsed_confidence"] <= 0.5],
        "confidence_change_without_answer_change": confidence_change_rows,
        "ambiguous_correctness_label": [row for row in rows if row["grader_rule"] in {"string_containment_match", "string_mismatch"}],
    }
    cases = []
    for category, source_rows in case_sources.items():
        for row in source_rows[:3]:
            cases.append({
                "model": row["model_id"],
                "family": row["model_family"],
                "prompt": row["prompt_condition"] if category != "confidence_change_without_answer_change" else "cross-condition",
                "dataset": row["dataset"],
                "question_short": short(row.get("prompt"), 180),
                "answer": short(row.get("answer"), 300),
                "confidence": row["parsed_confidence"],
                "correctness": int(bool(row["correct_auto"])),
                "failure_type": category,
                "short_interpretation": "Selected for manual review; interpretation should be finalized after adjudication.",
            })
    if not case_sources["low_confidence_correct"]:
        cases.append({
            "model": "NA",
            "family": "NA",
            "prompt": "NA",
            "dataset": "NA",
            "question_short": "No low-confidence correct case was found under confidence <= 0.50.",
            "answer": "NA",
            "confidence": "NA",
            "correctness": "NA",
            "failure_type": "low_confidence_correct",
            "short_interpretation": "Not present in the saved aligned outputs under the predefined threshold.",
        })
    cases.append({
        "model": "NA",
        "family": "NA",
        "prompt": "NA",
        "dataset": "NA",
        "question_short": "No reliable automatic detector in saved outputs.",
        "answer": "NA",
        "confidence": "NA",
        "correctness": "NA",
        "failure_type": "answer_and_reasoning_mismatch",
        "short_interpretation": "Requires manual qualitative coding.",
    })
    write_csv(TABLE_DIR / "table_5_representative_failure_cases.csv", cases, ["model", "family", "prompt", "dataset", "question_short", "answer", "confidence", "correctness", "failure_type", "short_interpretation"])
