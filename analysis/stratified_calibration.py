"""Accuracy-stratified calibration analysis.

Motivation: the aggregate AR-vs-DLM calibration gap is confounded with model
capability. The AR family is more accurate, and less accurate models are
generally more overconfident, so an aggregate ECE gap does not by itself show
that the families differ in calibration behaviour. This script compares the
families *within* strata of question difficulty, which holds task difficulty
roughly fixed across the comparison.

Difficulty is defined leave-one-model-out: for model m, the difficulty of
question q is the fraction of correct answers that the OTHER six models
produced on q (54 generations). Without the leave-one-out step a model's own
answers would help define the stratum it is then evaluated in, which biases
each model toward looking well matched to its own bin.

Outputs land in paper_assets/tables/stratified/.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_paper_assets as G

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_assets" / "tables" / "stratified"

# Fixed-width difficulty bins on [0, 1]. Fixed width rather than quantile bins
# so that the strata mean the same thing across models and datasets.
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

REPORT_METRICS = [
    "accuracy",
    "mean_confidence",
    "expected_calibration_error",
    "area_under_risk_coverage",
    "area_under_roc",
    "high_confidence_wrong_rate",
]


def bin_index(value: float) -> int:
    for index in range(len(BIN_EDGES) - 1):
        low, high = BIN_EDGES[index], BIN_EDGES[index + 1]
        if index == len(BIN_EDGES) - 2:
            if low <= value <= high:
                return index
        elif low <= value < high:
            return index
    return len(BIN_LABELS) - 1


def leave_one_model_out_difficulty(rows: list[dict]) -> dict[tuple[str, str], float]:
    """(model_id, question_id) -> fraction correct among the other models."""
    correct_by_qm: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        if row.get("correct_auto") is None:
            continue
        correct_by_qm[(row["question_id"], row["model_id"])].append(
            1 if row["correct_auto"] else 0
        )

    totals: dict[str, tuple[int, int]] = {}
    for (question_id, _model), values in correct_by_qm.items():
        hits, count = totals.get(question_id, (0, 0))
        totals[question_id] = (hits + sum(values), count + len(values))

    difficulty: dict[tuple[str, str], float] = {}
    for (question_id, model_id), values in correct_by_qm.items():
        hits, count = totals[question_id]
        other_hits = hits - sum(values)
        other_count = count - len(values)
        if other_count:
            difficulty[(model_id, question_id)] = other_hits / other_count
    return difficulty


def annotate(rows: list[dict]) -> list[dict]:
    difficulty = leave_one_model_out_difficulty(rows)
    annotated = []
    for row in rows:
        key = (row["model_id"], row["question_id"])
        if key not in difficulty:
            continue
        row = dict(row)
        row["lomo_difficulty"] = difficulty[key]
        row["difficulty_bin"] = BIN_LABELS[bin_index(difficulty[key])]
        annotated.append(row)
    return annotated


def summarize(rows: list[dict], group_keys: tuple[str, ...], with_ci: bool) -> list[dict]:
    output = []
    for key, group in sorted(G.grouped(rows, group_keys).items()):
        point = G.metric_row(group)
        record = {name: value for name, value in zip(group_keys, key)}
        record["N"] = point["N"]
        record["questions"] = len({row["question_id"] for row in group})
        for metric in REPORT_METRICS:
            record[metric] = G.fmt(point.get(metric))
        if with_ci:
            intervals = G.bootstrap_metric_intervals(group, point)
            for metric in REPORT_METRICS:
                low, high = intervals.get(metric, (None, None))
                record[f"{metric}_ci_low"] = G.fmt(low)
                record[f"{metric}_ci_high"] = G.fmt(high)
        output.append(record)
    return output


def paired_family_gap(rows: list[dict]) -> list[dict]:
    """Per-bin AR-vs-DLM contrast, so the gap at matched difficulty is explicit."""
    by_bin = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_bin[row["difficulty_bin"]][row["model_family"]].append(row)

    output = []
    for label in BIN_LABELS:
        families = by_bin.get(label)
        if not families or not {"AR", "DLM"} <= set(families):
            continue
        ar, dlm = G.metric_row(families["AR"]), G.metric_row(families["DLM"])
        record = {"difficulty_bin": label, "N_ar": ar["N"], "N_dlm": dlm["N"]}
        for metric in REPORT_METRICS:
            a, d = ar.get(metric), dlm.get(metric)
            record[f"ar_{metric}"] = G.fmt(a)
            record[f"dlm_{metric}"] = G.fmt(d)
            record[f"gap_{metric}"] = (
                G.fmt(d - a) if isinstance(a, (int, float)) and isinstance(d, (int, float)) else None
            )
        output.append(record)
    return output


def write(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    G.write_csv(OUT_DIR / name, rows, list(rows[0].keys()))
    print(f"wrote {name} ({len(rows)} rows)")


def main() -> None:
    rows, _raw_counts = G.load_all_rows()
    rows = annotate(rows)
    print(f"annotated {len(rows)} generations with leave-one-model-out difficulty")

    write("difficulty_bin_populations.csv", summarize(rows, ("difficulty_bin",), False))
    write("family_by_difficulty.csv", summarize(rows, ("model_family", "difficulty_bin"), True))
    write("model_by_difficulty.csv", summarize(rows, ("model_id", "difficulty_bin"), False))
    write("family_gap_by_difficulty.csv", paired_family_gap(rows))
    write(
        "family_by_difficulty_by_dataset.csv",
        summarize(rows, ("dataset", "model_family", "difficulty_bin"), False),
    )
    write(
        "family_by_difficulty_neutral.csv",
        summarize(
            [row for row in rows if row["prompt_condition"] == "neutral"],
            ("model_family", "difficulty_bin"),
            True,
        ),
    )


if __name__ == "__main__":
    main()
