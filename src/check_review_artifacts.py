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

from load_generation_data import aligned_rows, load_all_rows
from render_figures import confidence_distribution_data, prompt_sensitivity_data
from compute_metrics import reliability_points


ROOT = Path(__file__).resolve().parents[1]


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


def check_manifest_files_exist() -> None:
    manifest = read_csv(ROOT / "docs" / "artifact_manifest.csv")
    missing = [row["file_path"] for row in manifest if not (ROOT / row["file_path"]).exists()]
    if missing:
        raise AssertionError("manifest points to missing files: " + ", ".join(missing))


def main() -> None:
    all_rows, _raw_counts = load_all_rows()
    rows = aligned_rows(all_rows)

    check_png(ROOT / "fig_tabs" / "figure_1_reliability_diagram.png")
    check_png(ROOT / "fig_tabs" / "figure_2_confidence_by_correctness.png")
    check_png(ROOT / "fig_tabs" / "figure_2_2_confidence_by_correctness_neutral.png")
    check_png(ROOT / "fig_tabs" / "figure_3_prompt_sensitivity.png")

    check_reliability_csv(ROOT / "fig_tabs" / "figure_1_reliability_diagram_data.csv", rows)
    check_distribution_csv(ROOT / "fig_tabs" / "figure_2_confidence_by_correctness_data.csv", confidence_distribution_data(rows))
    neutral_rows = [row for row in rows if row["prompt_condition"] == "neutral"]
    check_distribution_csv(ROOT / "fig_tabs" / "figure_2_2_confidence_by_correctness_neutral_data.csv", confidence_distribution_data(neutral_rows))
    check_prompt_sensitivity_csv(ROOT / "fig_tabs" / "figure_3_prompt_sensitivity_data.csv", rows)

    for schema_path in ["schema_benchmark_items.json", "schema_generations.json", "schema_metrics.json"]:
        json.loads((ROOT / "docs" / schema_path).read_text())
    check_manifest_files_exist()
    print("artifact checks passed")


if __name__ == "__main__":
    main()
