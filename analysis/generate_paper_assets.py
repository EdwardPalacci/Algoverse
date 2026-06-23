#!/usr/bin/env python3
"""Regenerate paper tables, figures, and documentation from split judge outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from compute_basic_metrics import produce_basic_metric_figures
from render_figures import FIG_CAPTION_DIR, FIG_CSV_DIR, FIG_PNG_DIR, produce_figures


ROOT = Path(__file__).resolve().parents[1]
PILOT_DATA = ROOT / "data" / "PilotDataset.json"
TABLE_DIR = ROOT / "paper_assets" / "tables"
FIG_DIR = ROOT / "paper_assets" / "figures"
QC_DIR = TABLE_DIR / "quality_control"
DOCS_DIR = ROOT / "documentation" / "research_notes"
HIGH_CONFIDENCE_THRESHOLD = 0.90
ECE_BINS = 10


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def grouped(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out = defaultdict(list)
    for row in rows:
        out[tuple(row.get(key, "") for key in keys)].append(row)
    return dict(out)


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def fmt(value: object) -> object:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def all_jsonl(pattern: str) -> list[Path]:
    return sorted(ROOT.glob(pattern))


def source_raw_files(source: str) -> list[Path]:
    return all_jsonl(f"{source}_models/model_outputs/raw_by_model/*.jsonl")


def source_parsed_files(source: str) -> list[Path]:
    return all_jsonl(f"{source}_models/model_outputs/parsed_by_model/*.jsonl")


def judge_files(source: str) -> list[Path]:
    return all_jsonl(f"analysis/llm_as_judge/results/{source}/by_model/*/all_datasets.jsonl")


def judge_result_files(source: str) -> list[Path]:
    return all_jsonl(f"analysis/llm_as_judge/results/{source}/by_model/**/*.jsonl")


def load_all_rows() -> tuple[list[dict], dict[tuple[str, str], int]]:
    raw_counts: dict[tuple[str, str], int] = {}
    for source in ["ar", "dlm"]:
        for path in source_raw_files(source):
            rows = read_jsonl(path)
            if not rows:
                continue
            family = rows[0].get("model_architecture") or source.upper()
            model = rows[0].get("model_name") or path.stem
            raw_counts[(family, model)] = len(rows)

    rows = []
    for source in ["ar", "dlm"]:
        for path in judge_files(source):
            for row in read_jsonl(path):
                row = dict(row)
                row["source_file"] = str(path.relative_to(ROOT))
                row["model_id"] = row.get("model_name", "")
                row["model_family"] = row.get("model_architecture", source.upper())
                row["prompt_condition"] = row.get("condition", "")
                row["parsed_confidence"] = (
                    float(row["confidence"]) if row.get("confidence") is not None else None
                )
                row["correct_auto"] = (
                    True if row.get("CORRECTNESS") == 1
                    else False if row.get("CORRECTNESS") == 0
                    else None
                )
                row["grader_rule"] = row.get("grading_method", "")
                rows.append(row)
    return rows, raw_counts


def shared_question_ids(rows: list[dict]) -> set[str]:
    qsets = {
        family[0]: {row["question_id"] for row in family_rows}
        for family, family_rows in grouped(rows, ("model_family",)).items()
    }
    if {"AR", "DLM"} <= set(qsets):
        return qsets["AR"] & qsets["DLM"]
    return set().union(*qsets.values()) if qsets else set()


def aligned_rows(rows: list[dict]) -> list[dict]:
    common = shared_question_ids(rows)
    return [row for row in rows if row["question_id"] in common]


def ece(rows: list[dict], bins: int = ECE_BINS) -> float | None:
    usable = [
        row for row in rows
        if row.get("parsed_confidence") is not None
        and row.get("correct_auto") is not None
    ]
    if not usable:
        return None
    total = len(usable)
    score = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        if bin_index == bins - 1:
            bucket = [row for row in usable if low <= row["parsed_confidence"] <= high]
        else:
            bucket = [row for row in usable if low <= row["parsed_confidence"] < high]
        if not bucket:
            continue
        bucket_accuracy = mean(1.0 if row["correct_auto"] else 0.0 for row in bucket)
        bucket_confidence = mean(row["parsed_confidence"] for row in bucket)
        score += (len(bucket) / total) * abs(bucket_accuracy - bucket_confidence)
    return score


def brier(rows: list[dict]) -> float | None:
    values = []
    for row in rows:
        confidence = row.get("parsed_confidence")
        if confidence is not None and row.get("correct_auto") is not None:
            target = 1.0 if row["correct_auto"] else 0.0
            values.append((confidence - target) ** 2)
    return mean(values)


def auroc(rows: list[dict]) -> float | None:
    positive = [
        row["parsed_confidence"] for row in rows
        if row.get("parsed_confidence") is not None and row.get("correct_auto") is True
    ]
    negative = [
        row["parsed_confidence"] for row in rows
        if row.get("parsed_confidence") is not None and row.get("correct_auto") is False
    ]
    if not positive or not negative:
        return None
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def aurc(rows: list[dict]) -> float | None:
    """Area under the risk-coverage curve; lower values are better."""
    usable = [
        row for row in rows
        if row.get("parsed_confidence") is not None
        and row.get("correct_auto") is not None
    ]
    if not usable:
        return None
    ranked = sorted(usable, key=lambda row: row["parsed_confidence"], reverse=True)
    errors = 0
    risks = []
    for index, row in enumerate(ranked, start=1):
        if row["correct_auto"] is False:
            errors += 1
        risks.append(errors / index)
    return mean(risks)


def behavioral_alignment_score(rows: list[dict]) -> float | None:
    """Fraction of paired prompt interventions that move confidence in the intended direction."""
    by_key: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("parsed_confidence") is None:
            continue
        key = (
            row.get("model_id", ""),
            row.get("question_id", ""),
            int(row.get("sample_id", 0)),
        )
        by_key[key][row.get("prompt_condition", "")] = row["parsed_confidence"]

    checks = []
    for condition_values in by_key.values():
        neutral = condition_values.get("neutral")
        if neutral is None:
            continue
        if "cautious" in condition_values:
            checks.append(1.0 if condition_values["cautious"] <= neutral else 0.0)
        if "overconfident" in condition_values:
            checks.append(1.0 if condition_values["overconfident"] >= neutral else 0.0)
    return mean(checks)


def metric_row(rows: list[dict], raw_count: int | None = None) -> dict:
    n = len(rows)
    correct = sum(1 for row in rows if row.get("correct_auto") is True)
    confidences = [row["parsed_confidence"] for row in rows if row.get("parsed_confidence") is not None]
    high_confidence_wrong = sum(
        1 for row in rows
        if row.get("correct_auto") is False
        and row.get("parsed_confidence") is not None
        and row["parsed_confidence"] >= HIGH_CONFIDENCE_THRESHOLD
    )
    parsed = sum(1 for row in rows if row.get("source_parse_success") is not False)
    return {
        "N": n,
        "accuracy": correct / n if n else None,
        "mean_confidence": mean(confidences),
        "expected_calibration_error": ece(rows),
        "area_under_risk_coverage": aurc(rows),
        "behavioral_alignment_score": behavioral_alignment_score(rows),
        "brier_score": brier(rows),
        "area_under_roc": auroc(rows),
        "high_confidence_wrong_rate": high_confidence_wrong / n if n else None,
        "parse_success": parsed / raw_count if raw_count else None,
    }


def produce_alignment_report(rows: list[dict], rows_for_comparison: list[dict]) -> None:
    report = []
    qsets = {
        family[0]: {row["question_id"] for row in family_rows}
        for family, family_rows in grouped(rows, ("model_family",)).items()
    }
    ar_only = sorted(qsets.get("AR", set()) - qsets.get("DLM", set()))
    dlm_only = sorted(qsets.get("DLM", set()) - qsets.get("AR", set()))
    report.append({
        "check_name": "same_question_id_set_raw_sources",
        "status": "pass" if not ar_only and not dlm_only else "fail",
        "n_affected": len(ar_only) + len(dlm_only),
        "details": f"AR-only={len(ar_only)}; DLM-only={len(dlm_only)}",
    })
    write_csv(
        QC_DIR / "data_alignment_exclusions.csv",
        [{"question_id": qid, "excluded_from": "comparative_tables", "reason": "AR-only question_id"} for qid in ar_only]
        + [{"question_id": qid, "excluded_from": "comparative_tables", "reason": "DLM-only question_id"} for qid in dlm_only],
        ["question_id", "excluded_from", "reason"],
    )

    shared = len(shared_question_ids(rows))
    report.append({
        "check_name": "same_question_id_set_comparative_analysis",
        "status": "pass",
        "n_affected": 0,
        "details": f"{shared} shared question_id values used for comparative tables and figures",
    })
    duplicate_keys = Counter(
        (row["source_file"], row["question_id"], row["prompt_condition"], row["sample_id"])
        for row in rows
    )
    duplicates = sum(value - 1 for value in duplicate_keys.values() if value > 1)
    report.append({
        "check_name": "duplicate_full_evaluation_keys_within_source",
        "status": "pass" if duplicates == 0 else "fail",
        "n_affected": duplicates,
        "details": "source_file + question_id + condition + sample_id",
    })

    pilot_datasets = {row["dataset"] for row in json.loads(PILOT_DATA.read_text(encoding="utf-8"))}
    checks = [
        ("valid_dataset", lambda row: row.get("dataset") in pilot_datasets),
        ("valid_model_family", lambda row: row.get("model_family") in {"AR", "DLM"}),
        ("valid_prompt_condition", lambda row: row.get("prompt_condition") in {"neutral", "cautious", "overconfident"}),
        ("confidence_normalized_0_1_or_parse_failure", lambda row: row.get("parsed_confidence") is None or 0.0 <= row["parsed_confidence"] <= 1.0),
        ("correctness_binary_llm_judge_label", lambda row: isinstance(row.get("correct_auto"), bool)),
    ]
    for name, predicate in checks:
        bad = [row for row in rows_for_comparison if not predicate(row)]
        report.append({
            "check_name": name,
            "status": "pass" if not bad else "fail",
            "n_affected": len(bad),
            "details": "" if not bad else "invalid rows present",
        })
    write_csv(QC_DIR / "data_alignment_report.csv", report, ["check_name", "status", "n_affected", "details"])


def produce_tables(rows: list[dict], raw_counts: dict[tuple[str, str], int]) -> None:
    table1 = []
    for (dataset, model, family, condition), group_rows in sorted(grouped(rows, ("dataset", "model_id", "model_family", "prompt_condition")).items()):
        table1.append({
            "dataset": dataset,
            "N": len(group_rows),
            "model": model,
            "family": family,
            "prompt_condition": condition,
            "confidence_scale": "verbalized probability in [0, 1]",
            "correctness_grader": "LLM-as-judge with deterministic numeric and multiple-choice checks",
            "metrics": "accuracy; mean confidence; expected calibration error; AURC; behavioral alignment score; Brier score; AUROC; high-confidence wrong rate; parse success",
        })
    write_csv(TABLE_DIR / "table_1_benchmark_specification.csv", table1, ["dataset", "N", "model", "family", "prompt_condition", "confidence_scale", "correctness_grader", "metrics"])

    table2 = []
    for (model, family), group_rows in sorted(grouped(rows, ("model_id", "model_family")).items()):
        metrics = metric_row(group_rows, raw_counts.get((family, model)))
        table2.append({"model": model, "family": family, **{key: fmt(value) for key, value in metrics.items()}})
    write_csv(TABLE_DIR / "table_2_aggregate_metrics.csv", table2, ["model", "family", "N", "accuracy", "mean_confidence", "expected_calibration_error", "area_under_risk_coverage", "behavioral_alignment_score", "brier_score", "area_under_roc", "high_confidence_wrong_rate", "parse_success"])

    table3 = []
    for (dataset, model, family), group_rows in sorted(grouped(rows, ("dataset", "model_id", "model_family")).items()):
        metrics = metric_row(group_rows)
        table3.append({"dataset": dataset, "model": model, "family": family, **{key: fmt(value) for key, value in metrics.items() if key != "parse_success"}})
    write_csv(TABLE_DIR / "table_3_per_dataset_metrics.csv", table3, ["dataset", "model", "family", "N", "accuracy", "mean_confidence", "expected_calibration_error", "area_under_risk_coverage", "behavioral_alignment_score", "brier_score", "area_under_roc", "high_confidence_wrong_rate"])

    table4 = []
    for (model, family, condition), group_rows in sorted(grouped(rows, ("model_id", "model_family", "prompt_condition")).items()):
        metrics = metric_row(group_rows)
        table4.append({"model": model, "family": family, "prompt_condition": condition, **{key: fmt(value) for key, value in metrics.items() if key not in {"area_under_roc", "behavioral_alignment_score", "parse_success"}}})
    write_csv(TABLE_DIR / "table_4_prompt_condition_metrics.csv", table4, ["model", "family", "prompt_condition", "N", "accuracy", "mean_confidence", "expected_calibration_error", "area_under_risk_coverage", "brier_score", "high_confidence_wrong_rate"])


def short(value: object, max_chars: int = 140) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def produce_audit_and_cases(rows: list[dict]) -> None:
    audit_rows = []
    categories = [
        ("high_confidence_wrong", lambda row: row["correct_auto"] is False and row.get("parsed_confidence") is not None and row["parsed_confidence"] >= HIGH_CONFIDENCE_THRESHOLD),
        ("low_confidence_correct", lambda row: row["correct_auto"] is True and row.get("parsed_confidence") is not None and row["parsed_confidence"] <= 0.5),
        ("parse_failure", lambda row: row.get("source_parse_success") is False),
        ("llm_judged_short_answer", lambda row: row.get("grading_method") == "llm_judge"),
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
                "model_answer": short(row.get("model_answer") or row.get("answer"), 300),
                "gold_answer": short(row.get("ground_truth"), 300),
                "automatic_grade": int(bool(row["correct_auto"])),
                "manual_grade": "pending",
                "issue_type": category,
                "notes": f"judge method: {row.get('grading_method')}; reason: {short(row.get('judge_reason'), 180)}",
            })
    write_csv(QC_DIR / "manual_grading_audit.csv", audit_rows, ["question_id", "dataset", "model", "family", "prompt_condition", "model_answer", "gold_answer", "automatic_grade", "manual_grade", "issue_type", "notes"])

    counts = Counter(row["issue_type"] for row in audit_rows)
    summary = [
        {"audit_category": category, "N": counts[category], "agreement_rate": "NA", "main_issue": "manual adjudication not yet performed"}
        for category, _predicate in categories
    ]
    write_csv(TABLE_DIR / "manual_grading_audit_summary.csv", summary, ["audit_category", "N", "agreement_rate", "main_issue"])

    cases = []
    for category, predicate in categories[:4]:
        for row in [row for row in rows if predicate(row)][:3]:
            cases.append({
                "model": row["model_id"],
                "family": row["model_family"],
                "prompt": row["prompt_condition"],
                "dataset": row["dataset"],
                "question_short": short(row.get("prompt"), 180),
                "answer": short(row.get("model_answer") or row.get("answer"), 300),
                "confidence": row.get("parsed_confidence", "NA"),
                "correctness": int(bool(row["correct_auto"])),
                "failure_type": category,
                "short_interpretation": short(row.get("judge_reason"), 220) or "Selected for manual review.",
            })
    write_csv(TABLE_DIR / "table_5_representative_failure_cases.csv", cases, ["model", "family", "prompt", "dataset", "question_short", "answer", "confidence", "correctness", "failure_type", "short_interpretation"])


def produce_table_captions() -> None:
    captions = {
        "table_1_caption.txt": "Table 1. Benchmark specification by dataset, model, model family, and prompt condition. N is the number of judged generations in the aligned comparative analysis set.\n",
        "table_2_caption.txt": "Table 2. Aggregate calibration metrics by model. AURC is area under the risk-coverage curve, where lower is better; behavioral alignment score measures whether cautious and overconfident prompt interventions move confidence in the intended direction relative to neutral prompting.\n",
        "table_3_caption.txt": "Table 3. Dataset-level calibration metrics by model. Metrics are computed on judged generations from the shared question-ID analysis set and include ECE, AURC, and behavioral alignment score.\n",
        "table_4_caption.txt": "Table 4. Prompt-condition calibration metrics. Expected calibration error, AURC, and high-confidence wrong rate quantify sensitivity to cautious, neutral, and overconfident prompting.\n",
        "table_5_caption.txt": "Table 5. Representative cases selected for qualitative audit. Correctness reflects the saved LLM-as-judge result and should be manually adjudicated before being used as final qualitative evidence.\n",
    }
    for filename, caption in captions.items():
        write_text(TABLE_DIR / filename, caption)


def produce_docs(rows: list[dict], raw_counts: dict[tuple[str, str], int]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model_id"] for row in rows})
    conditions = sorted({row["prompt_condition"] for row in rows})
    shared_count = len(shared_question_ids(rows))
    spec = f"""# Benchmark Specification

## Benchmark Objective

Evaluate whether verbalized confidence tracks answer correctness under controlled prompting, comparing autoregressive language models with a diffusion language model.

## Unit of Evaluation

One model generation for one `question_id`, prompt condition, and sample index. Comparative metric tables and figures use the shared AR/DLM question-ID intersection ({shared_count} questions).

## Datasets

{', '.join(datasets)}

## Model Identifiers

{', '.join(models)}

## Prompt Conditions

{', '.join(conditions)}

## Correctness Grader

Correctness labels come from `analysis/llm_as_judge/llm_as_judge.py`. Numeric and multiple-choice answers are graded deterministically; short-answer rows are graded by the configured LLM judge using the raw generation text when available. Empty, malformed, or parse-failed answers are marked incorrect.

## Metric Definitions

Accuracy is the fraction of judged generations marked correct. Mean confidence is the arithmetic mean of verbalized confidence. Expected calibration error (ECE) uses {ECE_BINS} equal-width confidence bins and weights each absolute bin accuracy-confidence gap by bin frequency. Area under the risk-coverage curve (AURC) sorts generations from highest to lowest confidence, computes the cumulative error rate at each coverage level, and averages those risks; lower AURC indicates better confidence-based selective prediction. Behavioral alignment score (BAS) is the fraction of paired prompt interventions where cautious prompting gives confidence less than or equal to neutral confidence and overconfident prompting gives confidence greater than or equal to neutral confidence for the same model, question, and sample index. Brier score is the mean squared error between confidence and correctness. Area under the receiver operating characteristic curve (AUROC) is the Mann-Whitney probability that a correct generation receives higher confidence than an incorrect generation, with half credit for ties. High-confidence wrong rate is the fraction of all evaluated generations that are incorrect with confidence >= {HIGH_CONFIDENCE_THRESHOLD:.2f}. Parse success is parsed rows divided by raw rows for each model.
"""
    write_text(DOCS_DIR / "benchmark_spec.md", spec)

    readme = """# Review Artifact

This artifact contains saved AR and DLM generations, split LLM-as-judge results, benchmark specifications, metric tables, figures, captions, schema documentation, and audit templates for the verbalized confidence calibration benchmark.

## Reproduction

From the repository root, run:

```bash
python3 analysis/generate_paper_assets.py
python3 analysis/check_review_artifacts.py
```

Generation outputs are split by model under `ar_models/model_outputs/` and `dlm_models/model_outputs/`. Judge results are split under `analysis/llm_as_judge/results/` by source, model, and dataset.
"""
    write_text(DOCS_DIR / "artifact_readme.md", readme)

    data_manifest = [{"file_path": "data/PilotDataset.json", "description": "Benchmark item source file", "row_count": len(json.loads(PILOT_DATA.read_text(encoding="utf-8")))}]
    for source in ["ar", "dlm"]:
        for path in source_raw_files(source) + source_parsed_files(source) + judge_result_files(source):
            data_manifest.append({
                "file_path": str(path.relative_to(ROOT)),
                "description": "Generated model or judge output",
                "row_count": len(read_jsonl(path)),
            })
    write_csv(DOCS_DIR / "data_manifest.csv", data_manifest, ["file_path", "description", "row_count"])


def produce_schema_files() -> None:
    schemas = {
        "schema_benchmark_items.json": {
            "type": "object",
            "required": ["question_id", "dataset", "question", "ground_truth", "answer_type"],
        },
        "schema_generations.json": {
            "type": "object",
            "required": ["question_id", "dataset", "condition", "sample_id", "model_name", "model_architecture", "answer", "confidence", "parse_success"],
        },
        "schema_metrics.json": {
            "type": "object",
            "required": ["model", "family", "N", "accuracy", "mean_confidence", "expected_calibration_error", "area_under_risk_coverage", "behavioral_alignment_score", "brier_score", "area_under_roc", "high_confidence_wrong_rate"],
        },
    }
    for filename, schema in schemas.items():
        write_text(DOCS_DIR / filename, json.dumps(schema, indent=2) + "\n")


def remove_stale_figure_files() -> None:
    for suffix in ("*.png", "*.csv", "*.txt"):
        for path in FIG_DIR.glob(suffix):
            path.unlink()
    stale_patterns = [
        "ar_pilot_*",
        "figure_1.png",
        "figure_1_reliability_diagram*",
        "figure_2.png",
        "figure_2_confidence_by_correctness*",
        "figure_2_2_*",
        "figure_3_prompt_sensitivity*",
    ]
    for pattern in stale_patterns:
        for directory in [FIG_DIR, FIG_PNG_DIR, FIG_CSV_DIR, FIG_CAPTION_DIR]:
            for path in directory.glob(pattern):
                path.unlink()


def produce_manifest() -> None:
    definitions = []
    for path in [
        *source_raw_files("ar"),
        *source_parsed_files("ar"),
        *source_raw_files("dlm"),
        *source_parsed_files("dlm"),
        *judge_result_files("ar"),
        *judge_result_files("dlm"),
        *sorted(TABLE_DIR.glob("*.csv")),
        *sorted(TABLE_DIR.glob("*.txt")),
        *sorted(QC_DIR.glob("*.csv")),
        *sorted(FIG_PNG_DIR.glob("*.png")),
        *sorted(FIG_CSV_DIR.glob("*.csv")),
        *sorted(FIG_CAPTION_DIR.glob("*.txt")),
    ]:
        definitions.append({
            "file_path": str(path.relative_to(ROOT)),
            "artifact_type": "artifact",
            "description": "Generated benchmark artifact",
            "paper_section": "Results",
            "anonymous": "true",
            "required_for_reproduction": "true",
        })
    for rel, description in [
        ("analysis/generate_paper_assets.py", "Paper asset generation script"),
        ("analysis/check_review_artifacts.py", "Paper asset validation script"),
        ("analysis/compute_basic_metrics.py", "Figure 1 and Figure 2 metric rendering helpers"),
        ("analysis/render_figures.py", "Figure rendering helpers"),
        ("analysis/llm_as_judge/llm_as_judge.py", "Canonical LLM-as-judge script"),
    ]:
        definitions.append({
            "file_path": rel,
            "artifact_type": "code",
            "description": description,
            "paper_section": "Artifact",
            "anonymous": "true",
            "required_for_reproduction": "true",
        })
    write_csv(DOCS_DIR / "artifact_manifest.csv", definitions, ["file_path", "artifact_type", "description", "paper_section", "anonymous", "required_for_reproduction"])


def main() -> None:
    for directory in [TABLE_DIR, FIG_DIR, FIG_PNG_DIR, FIG_CSV_DIR, FIG_CAPTION_DIR, QC_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    remove_stale_figure_files()
    all_rows, raw_counts = load_all_rows()
    rows_for_comparison = aligned_rows(all_rows)
    produce_alignment_report(all_rows, rows_for_comparison)
    produce_tables(rows_for_comparison, raw_counts)
    produce_figures(rows_for_comparison)
    produce_basic_metric_figures()
    produce_table_captions()
    produce_audit_and_cases(rows_for_comparison)
    produce_docs(all_rows, raw_counts)
    produce_schema_files()
    produce_manifest()

    print(f"generated paper assets from {len(rows_for_comparison)} aligned judged generations")
    print(f"shared question_id values: {len(shared_question_ids(all_rows))}")
    print("paper assets written to paper_assets/ and documentation/research_notes/")


if __name__ == "__main__":
    main()
