#!/usr/bin/env python3
"""Check that generated figures match the saved model outputs.

This script does not inspect every pixel. Instead, each figure writes a small
CSV containing the values it plotted. We recompute those values from the raw
parsed outputs and compare them here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from compute_basic_metrics import confidence_histogram, load_ar_judged_generations
from compute_basic_metrics import reliability_bins_by_condition
from generate_paper_assets import aligned_rows, load_all_rows
from render_figures import confidence_distribution_data, prompt_sensitivity_data
from render_figures import reliability_points


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "documentation" / "research_notes"
FIG_DIR = ROOT / "paper_assets" / "figures"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: object) -> float:
    return float(value)


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(a - b) <= tolerance


def check_png(path: Path) -> None:
    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"not a PNG file: {path}")


def check_distribution_csv(path: Path, expected_rows: list[dict]) -> None:
    actual_rows = read_csv(path)
    if len(actual_rows) != len(expected_rows):
        raise AssertionError(f"{path}: row count mismatch")
    for actual, expected in zip(actual_rows, expected_rows):
        for key in ["group", "model_family", "correct", "bin_low", "bin_high"]:
            if str(actual[key]) != str(expected[key]):
                raise AssertionError(f"{path}: mismatch in {key}: {actual[key]} != {expected[key]}")
        for key in ["count", "group_total"]:
            if int(actual[key]) != int(expected[key]):
                raise AssertionError(f"{path}: mismatch in {key}: {actual[key]} != {expected[key]}")
        if not close(as_float(actual["share"]), as_float(expected["share"])):
            raise AssertionError(f"{path}: share mismatch")


def check_reliability_csv(path: Path, rows: list[dict]) -> None:
    expected = []
    for family in ["AR", "DLM"]:
        for mean_confidence, empirical_accuracy in sorted(reliability_points(rows, family)):
            expected.append({
                "model_family": family,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
            })
    actual = read_csv(path)
    if len(actual) != len(expected):
        raise AssertionError(f"{path}: row count mismatch")
    for actual_row, expected_row in zip(actual, expected):
        if actual_row["model_family"] != expected_row["model_family"]:
            raise AssertionError(f"{path}: family mismatch")
        if not close(as_float(actual_row["mean_confidence"]), expected_row["mean_confidence"]):
            raise AssertionError(f"{path}: mean_confidence mismatch")
        if not close(as_float(actual_row["empirical_accuracy"]), expected_row["empirical_accuracy"]):
            raise AssertionError(f"{path}: empirical_accuracy mismatch")


def check_prompt_sensitivity_csv(path: Path, rows: list[dict]) -> None:
    expected = prompt_sensitivity_data(rows)
    actual = read_csv(path)
    if len(actual) != len(expected):
        raise AssertionError(f"{path}: row count mismatch")
    for actual_row, expected_row in zip(actual, expected):
        for key in ["model_family", "prompt_condition"]:
            if actual_row[key] != expected_row[key]:
                raise AssertionError(f"{path}: mismatch in {key}")
        if int(actual_row["N"]) != expected_row["N"]:
            raise AssertionError(f"{path}: N mismatch")
        if not close(as_float(actual_row["expected_calibration_error"]), expected_row["expected_calibration_error"]):
            raise AssertionError(f"{path}: ECE mismatch")


def check_basic_histogram_csv(path: Path) -> None:
    expected = confidence_histogram(load_ar_judged_generations())
    actual = read_csv(path)
    if len(actual) != len(expected):
        raise AssertionError(f"{path}: row count mismatch")
    for actual_row, expected_row in zip(actual, expected):
        for key in ["prompt_condition", "bin_low", "bin_high", "bin_center"]:
            if actual_row[key] != str(expected_row[key]):
                raise AssertionError(f"{path}: mismatch in {key}")
        for key in ["count", "total"]:
            if int(actual_row[key]) != int(expected_row[key]):
                raise AssertionError(f"{path}: mismatch in {key}")
        if not close(as_float(actual_row["share"]), float(expected_row["share"])):
            raise AssertionError(f"{path}: share mismatch")


def check_basic_reliability_csv(path: Path) -> None:
    expected = reliability_bins_by_condition(load_ar_judged_generations())
    actual = read_csv(path)
    if len(actual) != len(expected):
        raise AssertionError(f"{path}: row count mismatch")
    for actual_row, expected_row in zip(actual, expected):
        for key in ["prompt_condition", "bin_low", "bin_high", "bin_center"]:
            if actual_row[key] != str(expected_row[key]):
                raise AssertionError(f"{path}: mismatch in {key}")
        if int(actual_row["bin_count"]) != int(expected_row["bin_count"]):
            raise AssertionError(f"{path}: bin_count mismatch")
        for key in ["bin_accuracy", "bin_confidence"]:
            if actual_row[key] == "" and expected_row[key] == "":
                continue
            if not close(as_float(actual_row[key]), float(expected_row[key])):
                raise AssertionError(f"{path}: mismatch in {key}")


def check_manifest_files_exist() -> None:
    manifest = read_csv(DOCS_DIR / "artifact_manifest.csv")
    missing = [row["file_path"] for row in manifest if not (ROOT / row["file_path"]).exists()]
    if missing:
        raise AssertionError("manifest points to missing files: " + ", ".join(missing))


def main() -> None:
    all_rows, _raw_counts = load_all_rows()
    rows = aligned_rows(all_rows)

    check_png(FIG_DIR / "figure_1_reliability_diagram.png")
    check_png(FIG_DIR / "figure_2_confidence_by_correctness.png")
    check_png(FIG_DIR / "figure_2_2_confidence_by_correctness_neutral.png")
    check_png(FIG_DIR / "figure_3_prompt_sensitivity.png")
    check_png(FIG_DIR / "ar_pilot_confidence_histogram.png")
    check_png(FIG_DIR / "ar_pilot_reliability_diagram.png")

    check_reliability_csv(FIG_DIR / "figure_1_reliability_diagram_data.csv", rows)
    check_distribution_csv(FIG_DIR / "figure_2_confidence_by_correctness_data.csv", confidence_distribution_data(rows))
    neutral_rows = [row for row in rows if row["prompt_condition"] == "neutral"]
    check_distribution_csv(FIG_DIR / "figure_2_2_confidence_by_correctness_neutral_data.csv", confidence_distribution_data(neutral_rows))
    check_prompt_sensitivity_csv(FIG_DIR / "figure_3_prompt_sensitivity_data.csv", rows)
    check_basic_histogram_csv(FIG_DIR / "ar_pilot_confidence_histogram_data.csv")
    check_basic_reliability_csv(FIG_DIR / "ar_pilot_reliability_diagram_data.csv")

    for schema_path in ["schema_benchmark_items.json", "schema_generations.json", "schema_metrics.json"]:
        json.loads((DOCS_DIR / schema_path).read_text())
    check_manifest_files_exist()
    print("artifact checks passed")


if __name__ == "__main__":
    main()
