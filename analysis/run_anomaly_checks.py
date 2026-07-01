#!/usr/bin/env python3
"""Run anomaly checks on judged generation outputs."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_assets" / "tables" / "quality_control"
DOC_PATH = ROOT / "documentation" / "research_notes" / "anomaly_checks.md"
HIGH_CONFIDENCE_THRESHOLD = 0.90


MODEL_LABELS = {
    "Dream-org/Dream-v0-Instruct-7B": "Dream",
    "google/diffusiongemma-26B-A4B-it": "DiffusionGemma",
    "GSAI-ML/LLaDA-8B-Instruct": "LLaDA",
    "inception/mercury-2": "Mercury-2",
    "google/gemini-2.5-flash": "Gemini Flash",
    "openai/gpt-4.1-mini": "GPT-4.1 mini",
    "x-ai/grok-4.3": "Grok",
}


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
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fmt(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def all_judge_rows() -> list[dict]:
    rows = []
    for source in ("ar", "dlm"):
        for path in sorted((ROOT / "analysis" / "llm_as_judge" / "results" / source / "by_model").glob("*/all_datasets.jsonl")):
            for row in read_jsonl(path):
                row = dict(row)
                row["model_label"] = MODEL_LABELS.get(row.get("model_name"), row.get("model_name", ""))
                rows.append(row)
    return rows


def grouped(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    output = defaultdict(list)
    for row in rows:
        output[tuple(row.get(key, "") for key in keys)].append(row)
    return dict(output)


def model_summary(rows: list[dict]) -> list[dict]:
    output = []
    for (family, model_name, model_label), group_rows in sorted(grouped(rows, ("model_architecture", "model_name", "model_label")).items()):
        confidences = [row.get("confidence") for row in group_rows if isinstance(row.get("confidence"), (int, float))]
        high_confidence = [
            row for row in group_rows
            if isinstance(row.get("confidence"), (int, float))
            and row["confidence"] >= HIGH_CONFIDENCE_THRESHOLD
        ]
        output.append({
            "family": family,
            "model": model_name,
            "model_label": model_label,
            "N": len(group_rows),
            "accuracy": fmt(safe_div(sum(row.get("CORRECTNESS") == 1 for row in group_rows), len(group_rows))),
            "mean_confidence": fmt(sum(confidences) / len(confidences) if confidences else None),
            "parse_failure_count": sum(row.get("source_parse_success") is False for row in group_rows),
            "parse_failure_rate": fmt(safe_div(sum(row.get("source_parse_success") is False for row in group_rows), len(group_rows))),
            "zero_confidence_count": sum(row.get("confidence") == 0 for row in group_rows),
            "zero_confidence_rate": fmt(safe_div(sum(row.get("confidence") == 0 for row in group_rows), len(group_rows))),
            "high_confidence_count": len(high_confidence),
            "high_confidence_rate": fmt(safe_div(len(high_confidence), len(group_rows))),
            "exact_one_confidence_count": sum(row.get("confidence") == 1 for row in group_rows),
            "exact_one_confidence_rate": fmt(safe_div(sum(row.get("confidence") == 1 for row in group_rows), len(group_rows))),
            "high_confidence_wrong_count": sum(row.get("CORRECTNESS") == 0 for row in high_confidence),
            "high_confidence_wrong_rate": fmt(safe_div(sum(row.get("CORRECTNESS") == 0 for row in high_confidence), len(group_rows))),
        })
    return output


def dataset_summary(rows: list[dict]) -> list[dict]:
    output = []
    for (family, model_label, dataset), group_rows in sorted(grouped(rows, ("model_architecture", "model_label", "dataset")).items()):
        confidences = [row.get("confidence") for row in group_rows if isinstance(row.get("confidence"), (int, float))]
        output.append({
            "family": family,
            "model_label": model_label,
            "dataset": dataset,
            "N": len(group_rows),
            "accuracy": fmt(safe_div(sum(row.get("CORRECTNESS") == 1 for row in group_rows), len(group_rows))),
            "mean_confidence": fmt(sum(confidences) / len(confidences) if confidences else None),
            "parse_failure_rate": fmt(safe_div(sum(row.get("source_parse_success") is False for row in group_rows), len(group_rows))),
            "zero_confidence_rate": fmt(safe_div(sum(row.get("confidence") == 0 for row in group_rows), len(group_rows))),
            "high_confidence_rate": fmt(safe_div(sum(isinstance(row.get("confidence"), (int, float)) and row["confidence"] >= HIGH_CONFIDENCE_THRESHOLD for row in group_rows), len(group_rows))),
            "high_confidence_wrong_rate": fmt(safe_div(sum(row.get("CORRECTNESS") == 0 and isinstance(row.get("confidence"), (int, float)) and row["confidence"] >= HIGH_CONFIDENCE_THRESHOLD for row in group_rows), len(group_rows))),
        })
    return output


def family_dataset_summary(rows: list[dict]) -> list[dict]:
    output = []
    for (family, dataset), group_rows in sorted(grouped(rows, ("model_architecture", "dataset")).items()):
        confidences = [row.get("confidence") for row in group_rows if isinstance(row.get("confidence"), (int, float))]
        output.append({
            "family": family,
            "dataset": dataset,
            "N": len(group_rows),
            "accuracy": fmt(safe_div(sum(row.get("CORRECTNESS") == 1 for row in group_rows), len(group_rows))),
            "mean_confidence": fmt(sum(confidences) / len(confidences) if confidences else None),
            "parse_failure_rate": fmt(safe_div(sum(row.get("source_parse_success") is False for row in group_rows), len(group_rows))),
            "high_confidence_wrong_rate": fmt(safe_div(sum(row.get("CORRECTNESS") == 0 and isinstance(row.get("confidence"), (int, float)) and row["confidence"] >= HIGH_CONFIDENCE_THRESHOLD for row in group_rows), len(group_rows))),
        })
    return output


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def numbers(text: object) -> list[float]:
    return [float(match) for match in NUMBER_RE.findall(str(text or "").replace(",", ""))]


def near(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) <= max(1e-6, abs(left) * 1e-6)


def answer_explanation_candidates(rows: list[dict]) -> list[dict]:
    candidates = []
    for row in rows:
        answer = str(row.get("answer") or row.get("model_answer") or "").strip()
        explanation = str(row.get("short_explanation") or "")
        reason = ""
        if row.get("answer_type") == "numeric":
            answer_numbers = numbers(answer)
            explanation_numbers = numbers(explanation)
            if answer_numbers and explanation_numbers and not near(answer_numbers[-1], explanation_numbers[-1]):
                reason = f"numeric answer {answer_numbers[-1]:g} differs from final explanation number {explanation_numbers[-1]:g}"
        elif row.get("answer_type") == "multiple_choice":
            answer_letter = LETTER_RE.search(answer)
            explanation_claim = re.search(r"(?i)(?:answer|option|choice)\s+(?:is\s+)?([A-E])\b", explanation)
            if answer_letter and explanation_claim and answer_letter.group(1).upper() != explanation_claim.group(1).upper():
                reason = f"answer letter {answer_letter.group(1).upper()} differs from explanation claim {explanation_claim.group(1).upper()}"
        if reason:
            candidates.append({
                "family": row.get("model_architecture"),
                "model_label": row.get("model_label"),
                "dataset": row.get("dataset"),
                "condition": row.get("condition"),
                "question_id": row.get("question_id"),
                "sample_id": row.get("sample_id"),
                "answer_type": row.get("answer_type"),
                "answer": answer,
                "confidence": row.get("confidence"),
                "correctness_label": row.get("correctness_label"),
                "disagreement_flag": reason,
                "short_explanation": explanation,
            })
    return candidates


def compact_counter(counter: Counter) -> str:
    return "; ".join(f"{key}={value}" for key, value in counter.most_common())


def produce_markdown(rows: list[dict], summaries: list[dict], candidates: list[dict]) -> str:
    dlm_rows = [row for row in rows if row.get("model_architecture") == "DLM"]
    dream = [row for row in dlm_rows if row.get("model_label") == "Dream"]
    dream_parse_failures = [row for row in dream if row.get("source_parse_success") is False]
    dream_zero = [row for row in dream if row.get("confidence") == 0]
    diffusion = [row for row in dlm_rows if row.get("model_label") == "DiffusionGemma"]
    llada = [row for row in dlm_rows if row.get("model_label") == "LLaDA"]

    def rate(count: int, total: int) -> str:
        return f"{count}/{total} ({safe_div(count, total):.1%})"

    diff_high = [row for row in diffusion if isinstance(row.get("confidence"), (int, float)) and row["confidence"] >= HIGH_CONFIDENCE_THRESHOLD]
    llada_high = [row for row in llada if isinstance(row.get("confidence"), (int, float)) and row["confidence"] >= HIGH_CONFIDENCE_THRESHOLD]
    diff_high_wrong = [row for row in diff_high if row.get("CORRECTNESS") == 0]
    llada_high_wrong = [row for row in llada_high if row.get("CORRECTNESS") == 0]

    family_dataset = family_dataset_summary(rows)
    focused = [row for row in family_dataset if row["dataset"] in {"SimpleQA", "TruthfulQA"}]

    lines = [
        "# Anomaly Checks",
        "",
        "These checks use saved LLM-as-judge rows and parsed confidence values from the fixed 250-question evaluation. Repeated generations are summarized as diagnostic evidence, not as independent question-level evidence.",
        "",
        "## Dream Parse Failures",
        "",
        f"Dream has {rate(len(dream_parse_failures), len(dream))} rows with `source_parse_success == False`. The failures are concentrated by dataset as follows: {compact_counter(Counter(row.get('dataset') for row in dream_parse_failures))}. By prompt condition: {compact_counter(Counter(row.get('condition') for row in dream_parse_failures))}. The judge pipeline retained these rows rather than silently dropping them; {sum(row.get('CORRECTNESS') == 1 for row in dream_parse_failures)} were graded correct and {sum(row.get('CORRECTNESS') == 0 for row in dream_parse_failures)} were graded incorrect after deterministic or judge-based recovery.",
        "",
        "## Dream Zero-Confidence Spike",
        "",
        f"Dream has {rate(len(dream_zero), len(dream))} zero-confidence rows. This is not a parser-only artifact: the raw responses include explicit `\"confidence\": 0.0` fields. The spike appears mostly under cautious and neutral prompting: {compact_counter(Counter(row.get('condition') for row in dream_zero))}. By dataset: {compact_counter(Counter(row.get('dataset') for row in dream_zero))}.",
        "",
        "## DiffusionGemma and LLaDA High-Confidence Saturation",
        "",
        f"DiffusionGemma has {rate(len(diff_high), len(diffusion))} rows with confidence >= {HIGH_CONFIDENCE_THRESHOLD:.2f}; {rate(sum(row.get('confidence') == 1 for row in diffusion), len(diffusion))} are exactly 1.0. Its high-confidence wrong count is {rate(len(diff_high_wrong), len(diffusion))}.",
        f"LLaDA has {rate(len(llada_high), len(llada))} rows with confidence >= {HIGH_CONFIDENCE_THRESHOLD:.2f}; {rate(sum(row.get('confidence') == 1 for row in llada), len(llada))} are exactly 1.0. Its high-confidence wrong count is {rate(len(llada_high_wrong), len(llada))}.",
        "",
        "## Answer/Explanation Disagreement Candidates",
        "",
        f"The heuristic audit flags {len(candidates)} candidate answer/explanation disagreements. These are candidates, not final semantic labels. The current heuristic is conservative for multiple choice and numeric-only for arithmetic rows; most candidates are GSM8K rows where the answer field and the last number in the short explanation differ. Counts by model: {compact_counter(Counter(row.get('model_label') for row in candidates))}.",
        "",
        "## Dataset-Specific Failures",
        "",
        "SimpleQA remains the largest dataset-level failure mode. TruthfulQA also separates AR and DLM behavior. Family-level focused rows:",
        "",
        "| Family | Dataset | N | Accuracy | Mean confidence | Parse failure rate | High-confidence wrong rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in focused:
        lines.append(
            f"| {row['family']} | {row['dataset']} | {row['N']} | {row['accuracy']} | {row['mean_confidence']} | {row['parse_failure_rate']} | {row['high_confidence_wrong_rate']} |"
        )
    lines.extend([
        "",
        "## Follow-up Notes",
        "",
        "Dream parse failures and zero-confidence behavior are model-specific anomalies in the fixed 250-question evaluation, not reasons to discard the current data. DiffusionGemma and LLaDA require explicit high-confidence saturation reporting because confidence values near 1.0 are common even on wrong answers. SimpleQA and TruthfulQA expose the clearest dataset-specific failures and should remain prominent in follow-up evaluation.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    rows = all_judge_rows()
    summaries = model_summary(rows)
    datasets = dataset_summary(rows)
    family_datasets = family_dataset_summary(rows)
    candidates = answer_explanation_candidates([row for row in rows if row.get("model_architecture") == "DLM"])

    write_csv(
        OUT_DIR / "anomaly_model_summary.csv",
        summaries,
        [
            "family", "model", "model_label", "N", "accuracy", "mean_confidence",
            "parse_failure_count", "parse_failure_rate", "zero_confidence_count",
            "zero_confidence_rate", "high_confidence_count", "high_confidence_rate",
            "exact_one_confidence_count", "exact_one_confidence_rate",
            "high_confidence_wrong_count", "high_confidence_wrong_rate",
        ],
    )
    write_csv(
        OUT_DIR / "anomaly_dataset_summary.csv",
        datasets,
        [
            "family", "model_label", "dataset", "N", "accuracy", "mean_confidence",
            "parse_failure_rate", "zero_confidence_rate", "high_confidence_rate",
            "high_confidence_wrong_rate",
        ],
    )
    write_csv(
        OUT_DIR / "anomaly_family_dataset_summary.csv",
        family_datasets,
        [
            "family", "dataset", "N", "accuracy", "mean_confidence",
            "parse_failure_rate", "high_confidence_wrong_rate",
        ],
    )
    write_csv(
        OUT_DIR / "answer_explanation_disagreement_candidates.csv",
        candidates,
        [
            "family", "model_label", "dataset", "condition", "question_id",
            "sample_id", "answer_type", "answer", "confidence", "correctness_label",
            "disagreement_flag", "short_explanation",
        ],
    )
    DOC_PATH.write_text(produce_markdown(rows, summaries, candidates), encoding="utf-8")
    print(f"wrote anomaly checks for {len(rows)} judged rows")
    print(f"model summaries: {OUT_DIR / 'anomaly_model_summary.csv'}")
    print(f"dataset summaries: {OUT_DIR / 'anomaly_dataset_summary.csv'}")
    print(f"report: {DOC_PATH}")


if __name__ == "__main__":
    main()
